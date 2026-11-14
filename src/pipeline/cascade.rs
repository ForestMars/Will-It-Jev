// Task-owned ordered cascades and their compiled execution graph.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use super::registry::TierRegistry;
use super::task::Task;
use super::tier::{TierId, TierInstance, TierName};

/// One task's ordered path through concrete tier instances.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Cascade {
    tiers: Vec<TierId>,
}

impl Cascade {
    pub fn from_names(names: &[TierName], registry: &TierRegistry) -> Result<Self, CascadeError> {
        let mut tiers = Vec::with_capacity(names.len());
        let mut seen = BTreeSet::new();
        for name in names {
            if !seen.insert(name.clone()) {
                return Err(CascadeError::DuplicateTier(name.clone()));
            }
            let instance = registry.get(name).ok_or_else(|| CascadeError::UnknownTier(name.clone()))?;
            tiers.push(TierInstance::derive_id(
                instance.name(), instance.tier(), instance.params(), instance.artifact_hash(),
            ));
        }
        Ok(Self { tiers })
    }

    pub fn tiers(&self) -> &[TierId] { &self.tiers }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CascadeError {
    UnknownTier(TierName),
    DuplicateTier(TierName),
}
impl fmt::Display for CascadeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTier(tier) => write!(f, "cascade references unavailable tier {tier}"),
            Self::DuplicateTier(tier) => write!(f, "cascade lists tier {tier} more than once"),
        }
    }
}
impl Error for CascadeError {}

/// Positional address in the compiled graph, not tier identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Slot(u16);
impl Slot {
    pub fn from_index(index: usize) -> Option<Self> { u16::try_from(index).ok().map(Self) }
    pub fn index(self) -> usize { self.0 as usize }
}
impl fmt::Display for Slot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}", self.0) }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CascadeGraphError {
    UnknownTier { task: String, tier: TierId },
    DuplicateTier { task: String, tier: TierId },
    DuplicateTask(String),
    TierIdCollision { id: TierId, first: TierName, second: TierName },
    Cycle { tiers: Vec<TierId> },
}
impl fmt::Display for CascadeGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTier { task, tier } => write!(f, "task {task} cascade references unavailable tier {tier}"),
            Self::DuplicateTier { task, tier } => write!(f, "task {task} cascade lists tier {tier} more than once"),
            Self::DuplicateTask(task) => write!(f, "task {task} is defined more than once"),
            Self::TierIdCollision { id, first, second } =>
                write!(f, "tiers {first} and {second} generated the same TierId {id}"),
            Self::Cycle { tiers } => write!(f, "task cascades create a cycle among tiers: {}",
                tiers.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")),
        }
    }
}
impl Error for CascadeGraphError {}

/// The computed execution graph of all Task.cascade paths.
#[derive(Debug, Clone, Default)]
pub struct CascadeGraph {
    paths: BTreeMap<String, Cascade>,
    order: Vec<TierId>,
    tasks_by_tier: BTreeMap<TierId, BTreeSet<String>>,
    names_by_id: BTreeMap<TierId, TierName>,
    ids_by_name: BTreeMap<TierName, TierId>,
}

impl CascadeGraph {
    pub fn from_tasks(tasks: &[Task], registry: &TierRegistry) -> Result<Self, CascadeGraphError> {
        let mut graph = Self::default();
        for name in registry.names() {
            let instance = registry.get(&name).expect("registry name must resolve");
            let id = TierInstance::derive_id(
                instance.name(), instance.tier(), instance.params(), instance.artifact_hash(),
            );
            if let Some(first) = graph.names_by_id.insert(id, name.clone()) {
                if first != name {
                    return Err(CascadeGraphError::TierIdCollision { id, first, second: name });
                }
            }
            graph.ids_by_name.insert(name, id);
        }

        let mut successors: BTreeMap<TierId, BTreeSet<TierId>> = BTreeMap::new();
        let mut indegree: BTreeMap<TierId, usize> = BTreeMap::new();
        let mut first_seen: BTreeMap<TierId, usize> = BTreeMap::new();
        let mut next_rank = 0usize;

        for task in tasks {
            if graph.paths.contains_key(&task.name) {
                return Err(CascadeGraphError::DuplicateTask(task.name.clone()));
            }
            let path = task.cascade.tiers();
            let mut seen = BTreeSet::new();
            for tier in path {
                if !graph.names_by_id.contains_key(tier) {
                    return Err(CascadeGraphError::UnknownTier { task: task.name.clone(), tier: *tier });
                }
                if !seen.insert(*tier) {
                    return Err(CascadeGraphError::DuplicateTier { task: task.name.clone(), tier: *tier });
                }
                successors.entry(*tier).or_default();
                indegree.entry(*tier).or_insert(0);
                if !first_seen.contains_key(tier) {
                    first_seen.insert(*tier, next_rank);
                    next_rank += 1;
                }
                graph.tasks_by_tier.entry(*tier).or_default().insert(task.name.clone());
            }
            for edge in path.windows(2) {
                if successors.entry(edge[0]).or_default().insert(edge[1]) {
                    *indegree.entry(edge[1]).or_insert(0) += 1;
                }
            }
            graph.paths.insert(task.name.clone(), task.cascade.clone());
        }

        // First appearance is the deterministic tie-break for unordered nodes.
        let mut ready: BTreeSet<(usize, TierId)> = indegree.iter()
            .filter_map(|(tier, degree)| (*degree == 0).then_some((
                *first_seen.get(tier).expect("graph node has rank"), *tier,
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
                            *first_seen.get(dependent).expect("graph node has rank"),
                            *dependent,
                        ));
                    }
                }
            }
        }
        if order.len() != indegree.len() {
            let tiers = indegree.into_iter()
                .filter_map(|(tier, degree)| (degree > 0).then_some(tier)).collect();
            return Err(CascadeGraphError::Cycle { tiers });
        }
        graph.order = order;
        Ok(graph)
    }

    pub fn order(&self) -> &[TierId] { &self.order }
    pub fn task_uses(&self, task: &str, tier: &TierId) -> bool {
        self.tasks_by_tier.get(tier).is_some_and(|tasks| tasks.contains(task))
    }
    pub fn tasks_using(&self, tier: &TierId) -> Option<&BTreeSet<String>> { self.tasks_by_tier.get(tier) }
    pub fn tier_name(&self, id: &TierId) -> Option<&TierName> { self.names_by_id.get(id) }
    pub fn id_for_name(&self, name: &TierName) -> Option<TierId> { self.ids_by_name.get(name).copied() }
    pub fn slot(&self, id: &TierId) -> Option<Slot> {
        self.order.iter().position(|candidate| candidate == id).and_then(Slot::from_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::task::{FallbackPolicy, OutputKind};
    use crate::pipeline::tier::{tname, FakeEvaluate, TierContext, TierOutput};
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
        TierRegistry::of(names.iter().map(|name|
            Arc::new(TierInstance::fake(name, Arc::new(Silent)))).collect())
    }
    fn task(name: &str, registry: &TierRegistry, path: &[&str]) -> Task {
        Task {
            name: name.into(),
            cascade: Cascade::from_names(&path.iter().map(|n| tname(n)).collect::<Vec<_>>(), registry).unwrap(),
            output_kind: OutputKind::Binary,
            labels: None,
            label_metadata: Vec::new(),
            taxonomy_version: None,
            fallback: FallbackPolicy::Defer,
        }
    }

    #[test]
    fn task_cascades_compile_into_shared_graph_nodes_and_ordering_edges() {
        let registry = registry(&["rules", "modernbert", "harrier", "unused"]);
        let tasks = vec![
            task("route", &registry, &["rules", "modernbert", "harrier"]),
            task("tag", &registry, &["modernbert", "harrier"]),
        ];
        let graph = CascadeGraph::from_tasks(&tasks, &registry).unwrap();
        let names: Vec<_> = graph.order().iter().map(|id|
            graph.tier_name(id).unwrap().as_str()).collect();
        assert_eq!(names, ["rules", "modernbert", "harrier"]);
        let modernbert = graph.id_for_name(&tname("modernbert")).unwrap();
        assert_eq!(graph.tasks_using(&modernbert).unwrap().len(), 2);
        assert!(graph.id_for_name(&tname("unused")).is_some());
        assert_eq!(tasks[0].cascade.tiers().len(), 3);
        assert_eq!(tasks[1].cascade.tiers().len(), 2);
    }

    #[test]
    fn unrelated_paths_preserve_each_order_without_adding_edges_between_them() {
        let registry = registry(&["alpha", "beta", "gamma"]);
        let tasks = vec![
            task("first", &registry, &["alpha", "gamma"]),
            task("second", &registry, &["beta", "gamma"]),
        ];
        let graph = CascadeGraph::from_tasks(&tasks, &registry).unwrap();
        let names: Vec<_> = graph.order().iter().map(|id|
            graph.tier_name(id).unwrap().as_str()).collect();
        assert_eq!(names, ["alpha", "beta", "gamma"]);
    }

    #[test]
    fn contradictory_task_paths_are_rejected_as_a_cycle() {
        let registry = registry(&["alpha", "beta"]);
        let tasks = vec![
            task("first", &registry, &["alpha", "beta"]),
            task("second", &registry, &["beta", "alpha"]),
        ];
        assert!(matches!(CascadeGraph::from_tasks(&tasks, &registry),
            Err(CascadeGraphError::Cycle { .. })));
    }

    #[test]
    fn cascade_rejects_unknown_and_duplicate_tier_names() {
        let registry = registry(&["alpha"]);
        assert!(matches!(Cascade::from_names(&[tname("missing")], &registry),
            Err(CascadeError::UnknownTier(_))));
        assert!(matches!(Cascade::from_names(&[tname("alpha"), tname("alpha")], &registry),
            Err(CascadeError::DuplicateTier(_))));
    }

    #[test]
    fn slots_are_graph_positions() {
        assert_eq!(Slot::from_index(3).map(Slot::index), Some(3));
        assert_eq!(Slot::from_index(70_000), None);
    }
}
