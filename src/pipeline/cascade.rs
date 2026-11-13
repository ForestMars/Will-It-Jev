// src/pipeline/cascade.rs
//
// Persistent task-labelled execution graph. Authored TierNames are resolved
// against TierRegistry here; TierIds are assigned to graph nodes here.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use super::registry::TierRegistry;
use super::tier::TierInstance;
use super::tier_types::{Slot, TierId, TierName};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CascadeGraphError {
    UnknownTier { task: String, tier: TierName },
    DuplicateTier { task: String, tier: TierName },
    IdentityCollision { id: TierId, first: TierName, second: TierName },
    Cycle { tiers: Vec<TierId> },
}

impl fmt::Display for CascadeGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTier { task, tier } =>
                write!(f, "task {task} cascade references unavailable tier {tier}"),
            Self::DuplicateTier { task, tier } =>
                write!(f, "task {task} cascade lists tier {tier} more than once"),
            Self::IdentityCollision { id, first, second } =>
                write!(f, "tiers {first} and {second} generated the same TierId {id}"),
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

/// Persistent graph nodes and task-labelled edges are canonical. Topological
/// order and positional slots are derived projections.
#[derive(Debug, Clone, Default)]
pub struct CascadeGraph {
    nodes: BTreeSet<TierId>,
    names_by_id: BTreeMap<TierId, TierName>,
    ids_by_name: BTreeMap<TierName, TierId>,
    edges: BTreeSet<CascadeEdge>,
    order: Vec<TierId>,
    task_entries: BTreeMap<String, TierId>,
    tasks_by_tier: BTreeMap<TierId, BTreeSet<String>>,
}

impl CascadeGraph {
    /// Resolve configured names against the available registry and assign a
    /// stable TierId to each graph node.
    pub fn from_task_paths(
        paths: BTreeMap<String, Vec<TierName>>,
        registry: &TierRegistry,
    ) -> Result<Self, CascadeGraphError> {
        let mut graph = Self::default();
        let mut successors: BTreeMap<TierId, BTreeSet<TierId>> = BTreeMap::new();
        let mut indegree: BTreeMap<TierId, usize> = BTreeMap::new();
        let mut first_seen: BTreeMap<TierId, usize> = BTreeMap::new();
        let mut next_rank = 0usize;

        for (task, path) in &paths {
            let mut seen_names = BTreeSet::new();
            let mut resolved_path = Vec::with_capacity(path.len());

            for name in path {
                if !seen_names.insert(name.clone()) {
                    return Err(CascadeGraphError::DuplicateTier { task: task.clone(), tier: name.clone() });
                }
                let instance = registry.get(name)
                    .ok_or_else(|| CascadeGraphError::UnknownTier { task: task.clone(), tier: name.clone() })?;
                let id = TierInstance::derive_id(
                    instance.name(), instance.tier(), instance.params(), instance.artifact_hash(),
                );

                if let Some(existing) = graph.names_by_id.get(&id) {
                    if existing != name {
                        return Err(CascadeGraphError::IdentityCollision {
                            id, first: existing.clone(), second: name.clone(),
                        });
                    }
                } else {
                    graph.nodes.insert(id);
                    graph.names_by_id.insert(id, name.clone());
                    graph.ids_by_name.insert(name.clone(), id);
                }

                successors.entry(id).or_default();
                indegree.entry(id).or_insert(0);
                if !first_seen.contains_key(&id) {
                    first_seen.insert(id, next_rank);
                    next_rank += 1;
                }
                graph.tasks_by_tier.entry(id).or_default().insert(task.clone());
                resolved_path.push(id);
            }

            if let Some(first) = resolved_path.first() {
                graph.task_entries.insert(task.clone(), *first);
            }
            for pair in resolved_path.windows(2) {
                graph.edges.insert(CascadeEdge { from: pair[0], to: pair[1], task: task.clone() });
                if successors.entry(pair[0]).or_default().insert(pair[1]) {
                    *indegree.entry(pair[1]).or_insert(0) += 1;
                }
            }
        }

        let mut ready: BTreeSet<(usize, TierId)> = indegree.iter()
            .filter_map(|(tier, degree)| (*degree == 0).then_some((
                *first_seen.get(tier).expect("graph node has a first-seen rank"), *tier,
            )))
            .collect();
        while let Some((_, tier)) = ready.pop_first() {
            graph.order.push(tier);
            if let Some(next) = successors.get(&tier) {
                for dependent in next {
                    let degree = indegree.get_mut(dependent).expect("edge target has indegree");
                    *degree -= 1;
                    if *degree == 0 {
                        ready.insert((
                            *first_seen.get(dependent).expect("graph node has a first-seen rank"),
                            *dependent,
                        ));
                    }
                }
            }
        }
        if graph.order.len() != indegree.len() {
            let tiers = indegree.into_iter().filter_map(|(tier, degree)| (degree > 0).then_some(tier)).collect();
            return Err(CascadeGraphError::Cycle { tiers });
        }

        Ok(graph)
    }

    pub fn nodes(&self) -> &BTreeSet<TierId> { &self.nodes }
    pub fn edges(&self) -> &BTreeSet<CascadeEdge> { &self.edges }
    pub fn order(&self) -> &[TierId] { &self.order }
    pub fn task_entries(&self) -> &BTreeMap<String, TierId> { &self.task_entries }
    pub fn tier_name(&self, id: &TierId) -> Option<&TierName> { self.names_by_id.get(id) }
    pub fn id_for_name(&self, name: &TierName) -> Option<TierId> { self.ids_by_name.get(name).copied() }
    pub fn slot(&self, id: &TierId) -> Option<Slot> {
        self.order.iter().position(|candidate| candidate == id).and_then(Slot::from_index)
    }

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
        self.edges.iter().find(|edge| edge.task == task && &edge.from == tier).map(|edge| &edge.to)
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
    use crate::pipeline::tier_types::tname;
    use crate::pipeline::tier::{FakeEvaluate, TierContext, TierOutput};
    use async_trait::async_trait;
    use std::sync::Arc;

    struct Silent;
    #[async_trait]
    impl FakeEvaluate for Silent {
        async fn evaluate(&self, ctx: &TierContext<'_>) -> TierOutput {
            TierOutput::abstain_all("silent", ctx.tasks, "test")
        }
    }

    fn registry(names: &[&str]) -> TierRegistry {
        TierRegistry::of(names.iter().map(|name| Arc::new(TierInstance::fake(name, Arc::new(Silent)))).collect())
    }
    fn name(value: &str) -> TierName { tname(value) }

    #[test]
    fn graph_retains_task_labelled_edges_and_shared_nodes() {
        let registry = registry(&["rules", "modernbert", "harrier", "unused"]);
        let paths = [
            ("route".to_string(), vec![name("rules"), name("modernbert"), name("harrier")]),
            ("tag".to_string(), vec![name("modernbert"), name("harrier")]),
        ].into();
        let graph = CascadeGraph::from_task_paths(paths, &registry).unwrap();
        let rules = graph.id_for_name(&name("rules")).unwrap();
        let modernbert = graph.id_for_name(&name("modernbert")).unwrap();
        let harrier = graph.id_for_name(&name("harrier")).unwrap();
        assert_eq!(graph.order(), &[rules, modernbert, harrier]);
        assert_eq!(graph.nodes().len(), 3);
        assert_eq!(graph.edges().len(), 3);
        assert_eq!(graph.tasks_using(&modernbert).unwrap().len(), 2);
        assert_eq!(graph.next("route", &rules), Some(&modernbert));
        assert_eq!(graph.next("tag", &rules), None);
        assert!(graph.id_for_name(&name("unused")).is_none());
    }

    #[test]
    fn conflicting_task_orders_are_rejected_as_a_cycle() {
        let registry = registry(&["a", "b"]);
        let paths = [
            ("route".to_string(), vec![name("a"), name("b")]),
            ("tag".to_string(), vec![name("b"), name("a")]),
        ].into();
        assert!(matches!(CascadeGraph::from_task_paths(paths, &registry), Err(CascadeGraphError::Cycle { .. })));
    }

    #[test]
    fn task_paths_require_available_unique_tier_names() {
        let registry = registry(&["a"]);
        let paths = [("route".to_string(), vec![name("a"), name("missing")])].into();
        assert!(matches!(CascadeGraph::from_task_paths(paths, &registry), Err(CascadeGraphError::UnknownTier { .. })));
        let paths = [("route".to_string(), vec![name("a"), name("a")])].into();
        assert!(matches!(CascadeGraph::from_task_paths(paths, &registry), Err(CascadeGraphError::DuplicateTier { .. })));
    }

    #[test]
    fn execution_plan_contains_only_requested_routes() {
        let registry = registry(&["a", "b", "c", "d"]);
        let paths = [
            ("route".to_string(), vec![name("a"), name("b")]),
            ("tag".to_string(), vec![name("c"), name("d")]),
        ].into();
        let graph = CascadeGraph::from_task_paths(paths, &registry).unwrap();
        let plan = graph.plan(&BTreeSet::from(["tag".to_string()]));
        let route_id = graph.id_for_name(&name("a")).unwrap();
        assert_eq!(plan.required_nodes.len(), 2);
        assert!(!plan.required_nodes.contains(&route_id));
    }
}
