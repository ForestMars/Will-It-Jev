// src/pipeline/cascade.rs
//
// Compiled execution topology. Task configuration refers to tiers by TierName;
// this module resolves those names against the available-tier registry and
// assigns TierIds to the graph nodes.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::registry::TierRegistry;
use super::tier::{TierInstance, TierName};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TierId([u8; 16]);

impl TierId {
    pub const LEN: usize = 16;

    pub fn from_digest(digest: &[u8]) -> Self {
        let mut bytes = [0u8; Self::LEN];
        bytes.copy_from_slice(&digest[..Self::LEN]);
        Self(bytes)
    }

    pub fn parse(hex: &str) -> Result<Self, TierIdParseError> {
        let bad = || TierIdParseError(format!("invalid tier id {hex:?}: expected {} hex characters", Self::LEN * 2));
        if hex.len() != Self::LEN * 2 || !hex.is_ascii() {
            return Err(bad());
        }
        let mut bytes = [0u8; Self::LEN];
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).map_err(|_| bad())?;
        }
        Ok(Self(bytes))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierIdParseError(String);

impl fmt::Display for TierIdParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.0) }
}
impl Error for TierIdParseError {}

impl fmt::Display for TierId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 { write!(f, "{byte:02x}")?; }
        Ok(())
    }
}

impl<'de> Deserialize<'de> for TierId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        Self::parse(&text).map_err(serde::de::Error::custom)
    }
}

/// Position of a graph node in the compiled topological execution order.
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

/// A graph node's stable identity and its authored registry name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierInfo {
    pub name: TierName,
    pub id: TierId,
    pub slot: Slot,
}

/// Execution-graph projection from stable node identity to its topological slot.
#[derive(Debug, Clone, Default)]
pub struct TierDirectory {
    infos: Vec<TierInfo>,
}

impl TierDirectory {
    pub fn infos(&self) -> &[TierInfo] { &self.infos }
    pub fn info(&self, id: &TierId) -> Option<&TierInfo> {
        self.infos.iter().find(|info| &info.id == id)
    }
    pub fn info_by_name(&self, name: &TierName) -> Option<&TierInfo> {
        self.infos.iter().find(|info| &info.name == name)
    }
    pub fn slot(&self, id: &TierId) -> Option<Slot> { self.info(id).map(|info| info.slot) }
    pub fn slot_by_name(&self, name: &TierName) -> Option<Slot> {
        self.info_by_name(name).map(|info| info.slot)
    }

    pub fn from_graph(graph: &CascadeGraph) -> Self {
        let infos = graph.order.iter().enumerate().filter_map(|(index, id)| {
            Some(TierInfo {
                name: graph.tier_name(id)?.clone(),
                id: *id,
                slot: Slot::from_index(index)?,
            })
        }).collect();
        Self { infos }
    }
}

/// Persistent aggregate graph of task-specific fallback routes. The topological
/// order is a derived projection; nodes and task-labelled edges remain canonical.
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
    /// Resolve configured names against the available registry and assign
    /// execution identities while compiling the graph.
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
                let id = *graph.ids_by_name.entry(name.clone()).or_insert_with(|| {
                    TierInstance::derive_id(
                        instance.name(),
                        instance.tier(),
                        instance.params(),
                        instance.artifact_hash(),
                    )
                });

                if let Some(existing) = graph.names_by_id.get(&id) {
                    if existing != name {
                        return Err(CascadeGraphError::IdentityCollision {
                            id, first: existing.clone(), second: name.clone(),
                        });
                    }
                } else {
                    graph.names_by_id.insert(id, name.clone());
                }

                graph.nodes.insert(id);
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

        // Topological order is a deterministic projection of the retained graph.
        // It validates aggregate ordering; it is not a global task barrier.
        let mut ready: BTreeSet<(usize, TierId)> = indegree.iter()
            .filter_map(|(tier, degree)| (*degree == 0).then_some((
                *first_seen.get(tier).expect("graph node has a first-seen rank"),
                *tier,
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
            let tiers = indegree.into_iter()
                .filter_map(|(tier, degree)| (degree > 0).then_some(tier))
                .collect();
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
        TierRegistry::of(names.iter().map(|name| {
            Arc::new(TierInstance::fake(name, Arc::new(Silent)))
        }).collect())
    }

    fn name(value: &str) -> TierName { TierName::new(value).unwrap() }

    #[test]
    fn graph_retains_task_labelled_edges_and_shared_nodes() {
        let registry = registry(&["rules", "modernbert", "harrier"]);
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

    #[test]
    fn directory_is_a_projection_of_graph_order_not_registry_order() {
        let registry = registry(&["zulu", "alpha", "mike", "unused"]);
        let paths = [("route".to_string(), vec![name("zulu"), name("alpha"), name("mike")])].into();
        let graph = CascadeGraph::from_task_paths(paths, &registry).unwrap();
        let directory = TierDirectory::from_graph(&graph);
        assert_eq!(directory.slot_by_name(&name("zulu")).map(Slot::index), Some(0));
        assert_eq!(directory.slot_by_name(&name("alpha")).map(Slot::index), Some(1));
        assert!(directory.info_by_name(&name("unused")).is_none());
    }

    #[test]
    fn tier_id_round_trips_through_hex() {
        let id = TierId::from_digest(&[0xab; 32]);
        assert_eq!(id.to_string(), "ab".repeat(16));
        assert_eq!(TierId::parse(&id.to_string()), Ok(id));
        assert!(TierId::parse("xyz").is_err());
        assert!(TierId::parse(&"g".repeat(32)).is_err());
    }

    #[test]
    fn graph_assigns_stable_ids_from_instance_identity() {
        let make_registry = |tier_name: &str| {
            TierRegistry::of(vec![Arc::new(TierInstance::fake(
                tier_name, Arc::new(Silent),
            ))])
        };
        let first_registry = make_registry("encoder");
        let second_registry = make_registry("encoder");
        let changed_registry = make_registry("encoder_v2");
        let paths = |tier_name: &str| {
            [("route".to_string(), vec![name(tier_name)])].into()
        };

        let first = CascadeGraph::from_task_paths(paths("encoder"), &first_registry).unwrap();
        let second = CascadeGraph::from_task_paths(paths("encoder"), &second_registry).unwrap();
        let changed = CascadeGraph::from_task_paths(paths("encoder_v2"), &changed_registry).unwrap();
        let first_id = first.id_for_name(&name("encoder")).unwrap();
        assert_eq!(first_id, second.id_for_name(&name("encoder")).unwrap());
        assert_ne!(first_id, changed.id_for_name(&name("encoder_v2")).unwrap());
    }

    #[test]
    fn slot_is_an_index_and_nothing_more() {
        assert_eq!(Slot::from_index(3).map(Slot::index), Some(3));
        assert_eq!(Slot::from_index(70_000), None);
    }
}
