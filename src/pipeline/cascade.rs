use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use super::tier_types::TierId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CascadeGraphError {
    UnknownTier { task: String, tier: TierId },
    DuplicateTier { task: String, tier: TierId },
    Cycle { tiers: Vec<TierId> },
}

impl fmt::Display for CascadeGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTier { task, tier } =>
                write!(f, "task {task} cascade references unregistered TierId {tier}"),
            Self::DuplicateTier { task, tier } =>
                write!(f, "task {task} cascade lists TierId {tier} more than once"),
            Self::Cycle { tiers } => write!(
                f,
                "task cascades create a cycle among TierIds: {}",
                tiers.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")
            ),
        }
    }
}
impl Error for CascadeGraphError {}

/// One task-specific fallback transition. Shared transitions retain every task
/// that contributed them; they never become global AND-dependencies.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CascadeEdge {
    pub from: TierId,
    pub to: TierId,
    pub task: String,
}

#[derive(Debug, Clone, Default)]
pub struct ExecutionPlan {
    pub requested_tasks: BTreeSet<String>,
    pub required_nodes: BTreeSet<TierId>,
    pub task_entries: BTreeMap<String, TierId>,
}

/// Persistent aggregate graph of task-specific fallback routes. The topological
/// order is a derived projection; nodes and task-labelled edges remain canonical.
#[derive(Debug, Clone, Default)]
pub struct CascadeGraph {
    nodes: BTreeSet<TierId>,
    edges: BTreeSet<CascadeEdge>,
    order: Vec<TierId>,
    task_entries: BTreeMap<String, TierId>,
    tasks_by_tier: BTreeMap<TierId, BTreeSet<String>>,
}

impl CascadeGraph {
    pub fn from_task_paths(
        paths: BTreeMap<String, Vec<TierId>>,
        registered: &BTreeSet<TierId>,
    ) -> Result<Self, CascadeGraphError> {
        let mut nodes = BTreeSet::new();
        let mut edges = BTreeSet::new();
        let mut successors: BTreeMap<TierId, BTreeSet<TierId>> = BTreeMap::new();
        let mut indegree: BTreeMap<TierId, usize> = BTreeMap::new();
        let mut tasks_by_tier: BTreeMap<TierId, BTreeSet<String>> = BTreeMap::new();
        let mut task_entries = BTreeMap::new();
        let mut first_seen: BTreeMap<TierId, usize> = BTreeMap::new();
        let mut next_rank = 0usize;

        for (task, path) in &paths {
            let mut seen = BTreeSet::new();
            for tier in path {
                if !registered.contains(tier) {
                    return Err(CascadeGraphError::UnknownTier { task: task.clone(), tier: *tier });
                }
                if !seen.insert(*tier) {
                    return Err(CascadeGraphError::DuplicateTier { task: task.clone(), tier: *tier });
                }
                nodes.insert(*tier);
                successors.entry(*tier).or_default();
                indegree.entry(*tier).or_insert(0);
                if !first_seen.contains_key(tier) {
                    first_seen.insert(*tier, next_rank);
                    next_rank += 1;
                }
                tasks_by_tier.entry(*tier).or_default().insert(task.clone());
            }
            if let Some(first) = path.first() {
                task_entries.insert(task.clone(), *first);
            }
            for pair in path.windows(2) {
                edges.insert(CascadeEdge { from: pair[0], to: pair[1], task: task.clone() });
                if successors.entry(pair[0]).or_default().insert(pair[1]) {
                    *indegree.entry(pair[1]).or_insert(0) += 1;
                }
            }
        }

        // Topological order is a deterministic projection of the retained graph.
        // It validates aggregate ordering; it is not a global task barrier.
        let mut ready: BTreeSet<(usize, TierId)> = indegree.iter()
            .filter_map(|(tier, degree)| (*degree == 0).then_some((
                *first_seen.get(tier).expect("registered graph node has a rank"),
                *tier,
            )))
            .collect();
        let mut order = Vec::with_capacity(indegree.len());
        while let Some((_, tier)) = ready.pop_first() {
            order.push(tier);
            if let Some(next) = successors.get(&tier) {
                for dependent in next {
                    let degree = indegree.get_mut(dependent).expect("edge target has indegree");
                    *degree -= 1;
                    if *degree == 0 {
                        ready.insert((
                            *first_seen.get(dependent).expect("registered graph node has a rank"),
                            *dependent,
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

        Ok(Self { nodes, edges, order, task_entries, tasks_by_tier })
    }

    /// Rebuild the graph after filtering disabled/unavailable instances.
    /// Each task follows only its own fallback transitions.
    pub fn retain_available(&self, available: &BTreeSet<TierId>) -> Result<Self, CascadeGraphError> {
        let paths = self.task_entries.keys().map(|task| {
            let path = self.task_path(task).into_iter().filter(|tier| available.contains(tier)).collect();
            (task.clone(), path)
        }).collect();
        Self::from_task_paths(paths, available)
    }

    pub fn nodes(&self) -> &BTreeSet<TierId> { &self.nodes }
    pub fn edges(&self) -> &BTreeSet<CascadeEdge> { &self.edges }
    pub fn order(&self) -> &[TierId] { &self.order }
    pub fn task_entries(&self) -> &BTreeMap<String, TierId> { &self.task_entries }

    pub fn task_path(&self, task: &str) -> Vec<TierId> {
        let mut path = Vec::new();
        let Some(mut current) = self.task_entries.get(task).copied() else { return path };
        path.push(current);
        while let Some(next) = self.next(task, &current).copied() {
            path.push(next);
            current = next;
        }
        path
    }

    pub fn next(&self, task: &str, tier: &TierId) -> Option<&TierId> {
        self.edges.iter()
            .find(|edge| edge.task == task && &edge.from == tier)
            .map(|edge| &edge.to)
    }

    pub fn task_uses(&self, task: &str, tier: &TierId) -> bool {
        self.tasks_by_tier.get(tier).is_some_and(|tasks| tasks.contains(task))
    }

    pub fn tasks_using(&self, tier: &TierId) -> Option<&BTreeSet<String>> {
        self.tasks_by_tier.get(tier)
    }

    pub fn plan(&self, requested_tasks: &BTreeSet<String>) -> ExecutionPlan {
        let mut required_nodes = BTreeSet::new();
        let mut task_entries = BTreeMap::new();
        for task in requested_tasks {
            if let Some(entry) = self.task_entries.get(task) {
                task_entries.insert(task.clone(), *entry);
                required_nodes.extend(self.task_path(task));
            }
        }
        ExecutionPlan { requested_tasks: requested_tasks.clone(), required_nodes, task_entries }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(byte: u8) -> TierId { TierId::from_digest(&[byte; 32]) }
    fn task(name: &str) -> String { name.to_string() }

    #[test]
    fn graph_retains_task_labelled_edges_and_shared_nodes() {
        let paths = [
            (task("route"), vec![id(1), id(2), id(3)]),
            (task("tag"), vec![id(2), id(3)]),
        ].into();
        let registered = [id(1), id(2), id(3)].into();
        let graph = CascadeGraph::from_task_paths(paths, &registered).unwrap();
        assert_eq!(graph.order(), &[id(1), id(2), id(3)]);
        assert_eq!(graph.nodes().len(), 3);
        assert_eq!(graph.edges().len(), 3);
        assert_eq!(graph.tasks_using(&id(2)).unwrap().len(), 2);
        assert_eq!(graph.next("route", &id(1)), Some(&id(2)));
        assert_eq!(graph.next("tag", &id(1)), None);
    }

    #[test]
    fn conflicting_task_orders_are_rejected_as_a_cycle() {
        let paths = [
            (task("route"), vec![id(1), id(2)]),
            (task("tag"), vec![id(2), id(1)]),
        ].into();
        let registered = [id(1), id(2)].into();
        assert!(matches!(CascadeGraph::from_task_paths(paths, &registered), Err(CascadeGraphError::Cycle { .. })));
    }

    #[test]
    fn task_paths_require_registered_unique_tier_ids() {
        let paths = [(task("route"), vec![id(1), id(3)])].into();
        let registered = [id(1)].into();
        assert!(matches!(CascadeGraph::from_task_paths(paths, &registered), Err(CascadeGraphError::UnknownTier { .. })));
        let paths = [(task("route"), vec![id(1), id(1)])].into();
        let registered = [id(1)].into();
        assert!(matches!(CascadeGraph::from_task_paths(paths, &registered), Err(CascadeGraphError::DuplicateTier { .. })));
    }

    #[test]
    fn execution_plan_contains_only_requested_routes() {
        let paths = [
            (task("route"), vec![id(1), id(2)]),
            (task("tag"), vec![id(3), id(4)]),
        ].into();
        let registered = [id(1), id(2), id(3), id(4)].into();
        let graph = CascadeGraph::from_task_paths(paths, &registered).unwrap();
        let plan = graph.plan(&BTreeSet::from([task("tag")]));
        assert_eq!(plan.required_nodes, BTreeSet::from([id(3), id(4)]));
        assert!(!plan.required_nodes.contains(&id(1)));
    }
}
