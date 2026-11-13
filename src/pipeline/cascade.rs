use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use super::identity::TierName;
use super::task::String;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CascadeGraphError {
    UnknownTier { task: String, tier: TierName },
    DuplicateTier { task: String, tier: TierName },
    Cycle { tiers: Vec<TierName> },
}

impl fmt::Display for CascadeGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTier { task, tier } => write!(f, "task {task} cascade references unregistered tier {tier}"),
            Self::DuplicateTier { task, tier } => write!(f, "task {task} cascade lists tier {tier} more than once"),
            Self::Cycle { tiers } => write!(f, "task cascades create a cycle among tiers: {}", tiers.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")),
        }
    }
}
impl Error for CascadeGraphError {}

/// One task-specific fallback transition. Shared transitions retain every task
/// that contributed them; they never become global AND-dependencies.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CascadeEdge {
    pub from: TierName,
    pub to: TierName,
    pub task: String,
}

#[derive(Debug, Clone, Default)]
pub struct ExecutionPlan {
    pub requested_tasks: BTreeSet<String>,
    pub required_nodes: BTreeSet<TierName>,
    pub task_entries: BTreeMap<String, TierName>,
}

/// Persistent aggregate graph of task-specific fallback routes. The topological
/// order is a derived projection; nodes and labelled edges remain authoritative.
#[derive(Debug, Clone, Default)]
pub struct CascadeGraph {
    nodes: BTreeSet<TierName>,
    edges: BTreeSet<CascadeEdge>,
    order: Vec<TierName>,
    task_entries: BTreeMap<String, TierName>,
    tasks_by_tier: BTreeMap<TierName, BTreeSet<String>>,
}

impl CascadeGraph {
    pub fn from_task_paths(
        paths: BTreeMap<String, Vec<TierName>>,
        registered: &BTreeSet<TierName>,
    ) -> Result<Self, CascadeGraphError> {
        let mut nodes = BTreeSet::new();
        let mut edges = BTreeSet::new();
        let mut successors: BTreeMap<TierName, BTreeSet<TierName>> = BTreeMap::new();
        let mut indegree: BTreeMap<TierName, usize> = BTreeMap::new();
        let mut tasks_by_tier: BTreeMap<TierName, BTreeSet<String>> = BTreeMap::new();
        let mut task_entries = BTreeMap::new();
        let mut first_seen: BTreeMap<TierName, usize> = BTreeMap::new();
        let mut next_rank = 0usize;

        for (task, path) in &paths {
            let mut seen = BTreeSet::new();
            for tier in path {
                if !registered.contains(tier) {
                    return Err(CascadeGraphError::UnknownTier { task: task.clone(), tier: tier.clone() });
                }
                if !seen.insert(tier.clone()) {
                    return Err(CascadeGraphError::DuplicateTier { task: task.clone(), tier: tier.clone() });
                }
                nodes.insert(tier.clone());
                successors.entry(tier.clone()).or_default();
                indegree.entry(tier.clone()).or_insert(0);
                if !first_seen.contains_key(tier) {
                    first_seen.insert(tier.clone(), next_rank);
                    next_rank += 1;
                }
                tasks_by_tier.entry(tier.clone()).or_default().insert(task.clone());
            }
            if let Some(first) = path.first() {
                task_entries.insert(task.clone(), first.clone());
            }
            for pair in path.windows(2) {
                edges.insert(CascadeEdge { from: pair[0].clone(), to: pair[1].clone(), task: task.clone() });
                if successors.entry(pair[0].clone()).or_default().insert(pair[1].clone()) {
                    *indegree.entry(pair[1].clone()).or_insert(0) += 1;
                }
            }
        }

        // Topological order is a deterministic projection of the retained graph.
        // Shared transitions contribute one ordering edge regardless of task count.
        let mut ready: BTreeSet<(usize, TierName)> = indegree.iter()
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
            let tiers = indegree.into_iter().filter_map(|(tier, degree)| (degree > 0).then_some(tier)).collect();
            return Err(CascadeGraphError::Cycle { tiers });
        }

        Ok(Self { nodes, edges, order, task_entries, tasks_by_tier })
    }

    /// Rebuild after disabling instances. Each route is reconstructed from its
    /// own labelled transitions before unavailable nodes are removed.
    pub fn retain_available(&self, available: &BTreeSet<TierName>) -> Result<Self, CascadeGraphError> {
        let paths = self.task_entries.keys().map(|task| {
            let path = self.task_path(task).into_iter().filter(|tier| available.contains(tier)).collect();
            (task.clone(), path)
        }).collect();
        Self::from_task_paths(paths, available)
    }

    pub fn nodes(&self) -> &BTreeSet<TierName> { &self.nodes }
    pub fn edges(&self) -> &BTreeSet<CascadeEdge> { &self.edges }
    pub fn order(&self) -> &[TierName] { &self.order }
    pub fn task_entries(&self) -> &BTreeMap<String, TierName> { &self.task_entries }

    pub fn task_path(&self, task: &String) -> Vec<TierName> {
        let mut path = Vec::new();
        let Some(mut current) = self.task_entries.get(task).cloned() else { return path };
        path.push(current.clone());
        while let Some(next) = self.next(task, &current).cloned() {
            path.push(next.clone());
            current = next;
        }
        path
    }

    pub fn next(&self, task: &String, tier: &TierName) -> Option<&TierName> {
        self.edges.iter().find(|edge| &edge.task == task && &edge.from == tier).map(|edge| &edge.to)
    }

    pub fn task_uses(&self, task: &String, tier: &TierName) -> bool {
        self.tasks_by_tier.get(tier).is_some_and(|tasks| tasks.contains(task))
    }

    pub fn tasks_using(&self, tier: &TierName) -> Option<&BTreeSet<String>> {
        self.tasks_by_tier.get(tier)
    }

    pub fn plan(&self, requested_tasks: &BTreeSet<String>) -> ExecutionPlan {
        let mut required_nodes = BTreeSet::new();
        let mut task_entries = BTreeMap::new();
        for task in requested_tasks {
            if let Some(entry) = self.task_entries.get(task) {
                task_entries.insert(task.clone(), entry.clone());
                required_nodes.extend(self.task_path(task));
            }
        }
        ExecutionPlan { requested_tasks: requested_tasks.clone(), required_nodes, task_entries }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn id(s: &str) -> TierName { TierName::new(s).unwrap() }
    fn task(s: &str) -> String { s.to_string() }

    #[test]
    fn graph_retains_task_labelled_edges_and_shared_nodes() {
        let paths = [
            (task("route"), vec![id("rules"), id("modernbert"), id("harrier")]),
            (task("tag"), vec![id("modernbert"), id("harrier")]),
        ].into();
        let registered = ["rules", "modernbert", "harrier"].into_iter().map(id).collect();
        let graph = CascadeGraph::from_task_paths(paths, &registered).unwrap();
        assert_eq!(graph.order(), &[id("rules"), id("modernbert"), id("harrier")]);
        assert_eq!(graph.nodes().len(), 3);
        assert_eq!(graph.edges().len(), 3);
        assert_eq!(graph.tasks_using(&id("modernbert")).unwrap().len(), 2);
        assert_eq!(graph.next(&task("route"), &id("rules")), Some(&id("modernbert")));
        assert_eq!(graph.next(&task("tag"), &id("rules")), None);
    }

    #[test]
    fn conflicting_task_orders_are_rejected_as_a_cycle() {
        let paths = [
            (task("route"), vec![id("a"), id("b")]),
            (task("tag"), vec![id("b"), id("a")]),
        ].into();
        let registered = [id("a"), id("b")].into();
        assert!(matches!(CascadeGraph::from_task_paths(paths, &registered), Err(CascadeGraphError::Cycle { .. })));
    }

    #[test]
    fn task_paths_require_registered_unique_tier_names() {
        let paths = [(task("route"), vec![id("a"), id("missing")])].into();
        let registered = [id("a")].into();
        assert!(matches!(CascadeGraph::from_task_paths(paths, &registered), Err(CascadeGraphError::UnknownTier { .. })));
        let paths = [(task("route"), vec![id("a"), id("a")])].into();
        let registered = [id("a")].into();
        assert!(matches!(CascadeGraph::from_task_paths(paths, &registered), Err(CascadeGraphError::DuplicateTier { .. })));
    }

    #[test]
    fn execution_plan_contains_only_requested_routes() {
        let paths = [
            (task("route"), vec![id("a"), id("b")]),
            (task("tag"), vec![id("c"), id("d")]),
        ].into();
        let registered = ["a", "b", "c", "d"].into_iter().map(id).collect();
        let graph = CascadeGraph::from_task_paths(paths, &registered).unwrap();
        let plan = graph.plan(&BTreeSet::from([task("tag")]));
        assert_eq!(plan.required_nodes, BTreeSet::from([id("c"), id("d")]));
        assert!(!plan.required_nodes.contains(&id("a")));
    }
}
