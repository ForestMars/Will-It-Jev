use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use super::registry::TierId;
use super::task::TaskId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CascadeGraphError {
    UnknownTier { task: TaskId, tier: TierId },
    DuplicateTier { task: TaskId, tier: TierId },
    Cycle { tiers: Vec<TierId> },
}

impl fmt::Display for CascadeGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTier { task, tier } =>
                write!(f, "task {task} cascade references unregistered tier {tier}"),
            Self::DuplicateTier { task, tier } =>
                write!(f, "task {task} cascade lists tier {tier} more than once"),
            Self::Cycle { tiers } => write!(
                f,
                "task cascades create a cycle among tiers: {}",
                tiers.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")
            ),
        }
    }
}
impl Error for CascadeGraphError {}

/// Union of task-specific ordered paths. Adjacent TierIds create dependency
/// edges; shared TierIds are single nodes and are evaluated once for all tasks
/// that still need a reading at that node.
#[derive(Debug, Clone, Default)]
pub struct CascadeGraph {
    paths: BTreeMap<TaskId, Vec<TierId>>,
    order: Vec<TierId>,
    tasks_by_tier: BTreeMap<TierId, BTreeSet<TaskId>>,
}

impl CascadeGraph {
    pub fn from_task_paths(
        paths: BTreeMap<TaskId, Vec<TierId>>,
        registered: &BTreeSet<TierId>,
    ) -> Result<Self, CascadeGraphError> {
        let mut successors: BTreeMap<TierId, BTreeSet<TierId>> = BTreeMap::new();
        let mut indegree: BTreeMap<TierId, usize> = BTreeMap::new();
        let mut tasks_by_tier: BTreeMap<TierId, BTreeSet<TaskId>> = BTreeMap::new();

        for (task, path) in &paths {
            let mut seen = BTreeSet::new();
            for tier in path {
                if !registered.contains(tier) {
                    return Err(CascadeGraphError::UnknownTier { task: task.clone(), tier: tier.clone() });
                }
                if !seen.insert(tier.clone()) {
                    return Err(CascadeGraphError::DuplicateTier { task: task.clone(), tier: tier.clone() });
                }
                successors.entry(tier.clone()).or_default();
                indegree.entry(tier.clone()).or_insert(0);
                if !first_seen.contains_key(tier) {
                    first_seen.insert(tier.clone(), next_rank);
                    next_rank += 1;
                }
                tasks_by_tier.entry(tier.clone()).or_default().insert(task.clone());
            }
            for edge in path.windows(2) {
                if successors.entry(edge[0].clone()).or_default().insert(edge[1].clone()) {
                    *indegree.entry(edge[1].clone()).or_insert(0) += 1;
                }
            }
        }

        // Preserve first appearance in the configured task paths whenever
        // dependency edges leave nodes unordered. This is deterministic and
        // avoids making lexical TierId order an accidental scheduling policy.
        let mut ready: BTreeSet<(usize, TierId)> = indegree.iter()
            .filter_map(|(tier, degree)| (*degree == 0).then_some((
                *first_seen.get(tier).expect("registered graph node has a rank"),
                tier.clone(),
            )))
            .collect();
        let mut order = Vec::with_capacity(indegree.len());
        while let Some((_, tier)) = ready.pop_first() {
            order.push(tier.clone());
            if let Some(next) = successors.get(&tier) {
                for dependent in next {
                    let degree = indegree.get_mut(dependent).expect("edge target has indegree");
                    *degree -= 1;
                    if *degree == 0 {
                        ready.insert((
                            *first_seen.get(dependent).expect("registered graph node has a rank"),
                            dependent.clone(),
                        ));
                    }
                }
            }
        }

        if order.len() != indegree.len() {
            let tiers = indegree.into_iter()
                .filter_map(|(tier, degree)| (degree > 0).then_some(tier))
                .collect();
            return Err(CascadeGraphError::Cycle { tiers });
        }

        Ok(Self { paths, order, tasks_by_tier })
    }

    /// Remove disabled/unavailable TierIds from each path and rebuild the DAG.
    /// Relative order among remaining TierIds is preserved.
    pub fn retain_available(&self, available: &BTreeSet<TierId>) -> Result<Self, CascadeGraphError> {
        let paths = self.paths.iter().map(|(task, path)| (
            task.clone(),
            path.iter().filter(|tier| available.contains(*tier)).cloned().collect(),
        )).collect();
        Self::from_task_paths(paths, available)
    }

    pub fn order(&self) -> &[TierId] { &self.order }
    pub fn paths(&self) -> &BTreeMap<TaskId, Vec<TierId>> { &self.paths }

    pub fn task_uses(&self, task: &TaskId, tier: &TierId) -> bool {
        self.tasks_by_tier.get(tier).is_some_and(|tasks| tasks.contains(task))
    }

    pub fn tasks_using(&self, tier: &TierId) -> Option<&BTreeSet<TaskId>> {
        self.tasks_by_tier.get(tier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn id(s: &str) -> TierId { TierId::new(s).unwrap() }
    fn task(s: &str) -> TaskId { TaskId::new(s) }

    #[test]
    fn shared_tier_ids_form_shared_dag_nodes() {
        let paths = [
            (task("route"), vec![id("rules"), id("modernbert"), id("harrier")]),
            (task("tag"), vec![id("modernbert"), id("harrier")]),
        ].into();
        let registered = ["rules", "modernbert", "harrier"].into_iter().map(id).collect();
        let graph = CascadeGraph::from_task_paths(paths, &registered).unwrap();
        assert_eq!(graph.order(), &[id("rules"), id("modernbert"), id("harrier")]);
        assert_eq!(graph.tasks_using(&id("modernbert")).unwrap().len(), 2);
    }

    #[test]
    fn conflicting_task_paths_are_rejected_as_a_cycle() {
        let paths = [
            (task("route"), vec![id("a"), id("b")]),
            (task("tag"), vec![id("b"), id("a")]),
        ].into();
        let registered = [id("a"), id("b")].into();
        assert!(matches!(CascadeGraph::from_task_paths(paths, &registered), Err(CascadeGraphError::Cycle { .. })));
    }

    #[test]
    fn task_paths_require_registered_unique_tier_ids() {
        let paths = [(task("route"), vec![id("a"), id("missing")])].into();
        let registered = [id("a")].into();
        assert!(matches!(CascadeGraph::from_task_paths(paths, &registered), Err(CascadeGraphError::UnknownTier { .. })));
        let paths = [(task("route"), vec![id("a"), id("a")])].into();
        let registered = [id("a")].into();
        assert!(matches!(CascadeGraph::from_task_paths(paths, &registered), Err(CascadeGraphError::DuplicateTier { .. })));
    }
}
