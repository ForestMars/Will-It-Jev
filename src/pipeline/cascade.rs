// Task-owned named cascades and their compiled, instance-bearing execution graph.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use super::registry::TierRegistry;
use super::task::Task;
use super::tier::{TierId, TierInstance, TierName};

/// One task's ordered path through the tier ontology. It contains names, never
/// materialized instance IDs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Cascade {
    tiers: Vec<TierName>,
}

impl Cascade {
    pub fn from_names(names: &[TierName]) -> Result<Self, CascadeError> {
        let mut seen = BTreeSet::new();
        for name in names {
            if !seen.insert(name.clone()) {
                return Err(CascadeError::DuplicateTier(name.clone()));
            }
        }
        Ok(Self { tiers: names.to_vec() })
    }

    pub fn tiers(&self) -> &[TierName] { &self.tiers }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CascadeError {
    DuplicateTier(TierName),
}
impl fmt::Display for CascadeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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

/// A precedence constraint contributed by one or more task-owned cascade paths.
/// The task set preserves the provenance of an edge in the union execution DAG.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CascadeDependency {
    pub from: TierId,
    pub to: TierId,
    pub tasks: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CascadeGraphError {
    UnknownTier { task: String, tier: TierName },
    DuplicateTier { task: String, tier: TierName },
    DuplicateTask(String),
    TierBuild { tier: TierName, reason: String },
    TierIdCollision { id: TierId, first: TierName, second: TierName },
    Cycle {
        tiers: Vec<TierId>,
        blocked_dependencies: Vec<CascadeDependency>,
    },
}
impl fmt::Display for CascadeGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTier { task, tier } =>
                write!(f, "task {task} cascade references unavailable tier {tier}"),
            Self::DuplicateTier { task, tier } =>
                write!(f, "task {task} cascade lists tier {tier} more than once"),
            Self::DuplicateTask(task) => write!(f, "task {task} is defined more than once"),
            Self::TierBuild { tier, reason } =>
                write!(f, "failed to materialize tier {tier} on CascadeGraph: {reason}"),
            Self::TierIdCollision { id, first, second } =>
                write!(f, "tiers {first} and {second} generated the same TierId {id}"),
            Self::Cycle { tiers, blocked_dependencies } => {
                let tiers = tiers.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ");
                let dependencies = blocked_dependencies.iter().map(|edge| format!(
                    "{} -> {} required by [{}]",
                    edge.from,
                    edge.to,
                    edge.tasks.iter().cloned().collect::<Vec<_>>().join(", "),
                )).collect::<Vec<_>>().join("; ");
                write!(
                    f,
                    "task cascades create a cycle among tiers: {tiers}; blocked precedence constraints: {dependencies}"
                )
            },
        }
    }
}
impl Error for CascadeGraphError {}

/// The graph owns the concrete TierInstances. Each task path remains a sequence
/// of TierNames; graph nodes receive hash-derived TierIds only during compilation.
/// The union graph has one global topological order, and each precedence edge
/// retains the task paths that introduced it.
#[derive(Clone, Default)]
pub struct CascadeGraph {
    paths: BTreeMap<String, Cascade>,
    order: Vec<TierId>,
    dependencies: Vec<CascadeDependency>,
    tasks_by_tier: BTreeMap<TierId, BTreeSet<String>>,
    names_by_id: BTreeMap<TierId, TierName>,
    ids_by_name: BTreeMap<TierName, TierId>,
    instances: BTreeMap<TierId, Arc<TierInstance>>,
}

impl CascadeGraph {
    pub fn from_tasks(
        tasks: &[Task],
        registry: &TierRegistry,
    ) -> Result<Self, CascadeGraphError> {
        let mut graph = Self::default();
        let mut successors: BTreeMap<TierId, BTreeMap<TierId, BTreeSet<String>>> = BTreeMap::new();
        let mut indegree: BTreeMap<TierId, usize> = BTreeMap::new();
        let mut first_seen: BTreeMap<TierId, usize> = BTreeMap::new();
        let mut next_rank = 0usize;

        for task in tasks {
            if graph.paths.contains_key(&task.name) {
                return Err(CascadeGraphError::DuplicateTask(task.name.clone()));
            }

            let mut path_ids = Vec::with_capacity(task.cascade.tiers().len());
            let mut seen_names = BTreeSet::new();

            for name in task.cascade.tiers() {
                if !seen_names.insert(name.clone()) {
                    return Err(CascadeGraphError::DuplicateTier {
                        task: task.name.clone(),
                        tier: name.clone(),
                    });
                }

                let id = if let Some(id) = graph.ids_by_name.get(name) {
                    *id
                } else {
                    let (definition, params) = registry.get(name).ok_or_else(|| {
                        CascadeGraphError::UnknownTier {
                            task: task.name.clone(),
                            tier: name.clone(),
                        }
                    })?;

                    #[cfg(test)]
                    let instance = if let Some(fake) = registry.fake(name) {
                        TierInstance::fake_with_definition(name.clone(), *definition, params.clone(), fake)
                    } else {
                        TierInstance::build(name.clone(), *definition, params.clone())
                            .map_err(|error| CascadeGraphError::TierBuild {
                                tier: name.clone(),
                                reason: error.to_string(),
                            })?
                    };

                    #[cfg(not(test))]
                    let instance = TierInstance::build(name.clone(), *definition, params.clone())
                        .map_err(|error| CascadeGraphError::TierBuild {
                            tier: name.clone(),
                            reason: error.to_string(),
                        })?;

                    let instance = Arc::new(instance);
                    let id = TierInstance::derive_id(
                        instance.name(),
                        instance.tier(),
                        instance.params(),
                        instance.artifact_hash(),
                    );

                    if let Some(first) = graph.names_by_id.get(&id) {
                        if first != name {
                            return Err(CascadeGraphError::TierIdCollision {
                                id,
                                first: first.clone(),
                                second: name.clone(),
                            });
                        }
                    }

                    graph.names_by_id.insert(id, name.clone());
                    graph.ids_by_name.insert(name.clone(), id);
                    graph.instances.insert(id, instance);
                    id
                };

                successors.entry(id).or_default();
                indegree.entry(id).or_insert(0);
                if !first_seen.contains_key(&id) {
                    first_seen.insert(id, next_rank);
                    next_rank += 1;
                }
                graph.tasks_by_tier.entry(id).or_default().insert(task.name.clone());
                path_ids.push(id);
            }

            for edge in path_ids.windows(2) {
                let owners = successors.entry(edge[0]).or_default().entry(edge[1]).or_default();
                if owners.is_empty() {
                    *indegree.entry(edge[1]).or_insert(0) += 1;
                }
                owners.insert(task.name.clone());
            }

            graph.paths.insert(task.name.clone(), task.cascade.clone());
        }

        // Materialize the union edges with their task-path provenance. Keep the
        // ordering relation global; provenance explains which paths require it.
        graph.dependencies = successors.iter()
            .flat_map(|(from, next)| next.iter().map(|(to, tasks)| CascadeDependency {
                from: *from,
                to: *to,
                tasks: tasks.clone(),
            }))
            .collect();

        // Preserve first appearance in task cascades as the deterministic tie-break.
        let mut ready: BTreeSet<(usize, TierId)> = indegree.iter()
            .filter_map(|(tier, degree)| (*degree == 0).then_some((
                *first_seen.get(tier).expect("every graph node has a rank"),
                *tier,
            )))
            .collect();

        let mut order = Vec::with_capacity(indegree.len());
        while let Some((_, tier)) = ready.pop_first() {
            order.push(tier);
            if let Some(next) = successors.get(&tier) {
                for dependent in next.keys() {
                    let degree = indegree.get_mut(dependent).expect("edge target has indegree");
                    *degree -= 1;
                    if *degree == 0 {
                        ready.insert((
                            *first_seen.get(dependent).expect("every graph node has a rank"),
                            *dependent,
                        ));
                    }
                }
            }
        }

        if order.len() != indegree.len() {
            let blocked: BTreeSet<TierId> = indegree.iter()
                .filter_map(|(tier, degree)| (*degree > 0).then_some(*tier))
                .collect();
            let tiers = blocked.iter().copied().collect();
            // Kahn's residual graph can include nodes downstream of a cycle,
            // so report every dependency still blocked by that residual graph.
            let blocked_dependencies = graph.dependencies.iter()
                .filter(|edge| blocked.contains(&edge.from) && blocked.contains(&edge.to))
                .cloned()
                .collect();
            return Err(CascadeGraphError::Cycle { tiers, blocked_dependencies });
        }

        graph.order = order;
        Ok(graph)
    }

    pub fn order(&self) -> &[TierId] { &self.order }

    /// All precedence constraints in the union DAG, with the task paths that
    /// contributed each edge. The returned order is deterministic.
    pub fn dependencies(&self) -> &[CascadeDependency] { &self.dependencies }

    pub fn task_uses(&self, task: &str, tier: &TierId) -> bool {
        self.tasks_by_tier.get(tier).is_some_and(|tasks| tasks.contains(task))
    }

    pub fn tasks_using(&self, tier: &TierId) -> Option<&BTreeSet<String>> {
        self.tasks_by_tier.get(tier)
    }

    pub fn tier_name(&self, id: &TierId) -> Option<&TierName> {
        self.names_by_id.get(id)
    }

    pub fn id_for_name(&self, name: &TierName) -> Option<TierId> {
        self.ids_by_name.get(name).copied()
    }

    pub fn instance(&self, id: &TierId) -> Option<Arc<TierInstance>> {
        self.instances.get(id).cloned()
    }

    pub fn instance_for_name(&self, name: &TierName) -> Option<Arc<TierInstance>> {
        self.id_for_name(name).and_then(|id| self.instance(&id))
    }

    pub fn instances(&self) -> impl Iterator<Item = (&TierId, &Arc<TierInstance>)> {
        self.instances.iter()
    }

    pub fn slot(&self, id: &TierId) -> Option<Slot> {
        self.order.iter().position(|candidate| candidate == id).and_then(Slot::from_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::task::{FallbackPolicy, OutputKind};
    use crate::pipeline::tier::{
        tname, FakeEvaluate, ModelFormat, Params, RequestShape, Role, TierContext, TierOutput,
        Transport,
    };
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
        let mut registry = TierRegistry::new();
        for name in names {
            registry.register_fake(
                tname(name),
                super::super::tier::Tier {
                    exex: Role::Encoder,
                    transport: Transport::Grpc,
                    request_shape: Some(RequestShape::Score),
                },
                Params {
                    model_format: Some(ModelFormat::Onnx),
                    model: None,
                    config: None,
                    endpoint: Some("http://127.0.0.1:9".into()),
                    timeout_ms: Some(250),
                    enabled: None,
                },
                Arc::new(Silent),
            ).unwrap();
        }
        registry
    }

    fn task(name: &str, path: &[&str]) -> Task {
        Task {
            name: name.into(),
            cascade: Cascade::from_names(&path.iter().map(|name| tname(name)).collect::<Vec<_>>()).unwrap(),
            output_kind: OutputKind::Binary,
            labels: None,
            label_metadata: Vec::new(),
            taxonomy_version: None,
            fallback: FallbackPolicy::Defer,
        }
    }

    #[test]
    fn task_cascades_remain_names_and_graph_materializes_only_referenced_instances() {
        let registry = registry(&["rules", "encoder", "unused"]);
        let tasks = vec![task("route", &["rules", "encoder"]), task("tag", &["encoder"])];
        assert_eq!(tasks[0].cascade.tiers(), &[tname("rules"), tname("encoder")]);

        let graph = CascadeGraph::from_tasks(&tasks, &registry).unwrap();
        let names: Vec<_> = graph.order().iter()
            .map(|id| graph.tier_name(id).unwrap().as_str()).collect();
        assert_eq!(names, ["rules", "encoder"]);
        let encoder = graph.id_for_name(&tname("encoder")).unwrap();
        assert_eq!(graph.tasks_using(&encoder).unwrap().len(), 2);
        assert!(registry.get(&tname("unused")).is_some());
        assert!(graph.id_for_name(&tname("unused")).is_none());
        assert!(graph.instance_for_name(&tname("unused")).is_none());
    }

    #[test]
    fn unrelated_paths_preserve_each_order_without_adding_edges_between_them() {
        let registry = registry(&["alpha", "beta", "gamma"]);
        let tasks = vec![task("first", &["alpha", "gamma"]), task("second", &["beta", "gamma"])];
        let graph = CascadeGraph::from_tasks(&tasks, &registry).unwrap();
        let names: Vec<_> = graph.order().iter()
            .map(|id| graph.tier_name(id).unwrap().as_str()).collect();
        assert_eq!(names, ["alpha", "beta", "gamma"]);
    }

    #[test]
    fn shared_dependencies_retain_every_contributing_task() {
        let registry = registry(&["alpha", "beta"]);
        let tasks = vec![
            task("first", &["alpha", "beta"]),
            task("second", &["alpha", "beta"]),
        ];
        let graph = CascadeGraph::from_tasks(&tasks, &registry).unwrap();

        assert_eq!(graph.dependencies().len(), 1);
        assert_eq!(
            graph.dependencies()[0].tasks,
            ["first".to_string(), "second".to_string()].into_iter().collect(),
        );
    }

    #[test]
    fn contradictory_task_paths_report_the_owners_of_blocked_dependencies() {
        let registry = registry(&["alpha", "beta"]);
        let tasks = vec![task("first", &["alpha", "beta"]), task("second", &["beta", "alpha"])];
        let error = match CascadeGraph::from_tasks(&tasks, &registry) {
            Err(error) => error,
            Ok(_) => panic!("contradictory task paths must fail graph compilation"),
        };

        let CascadeGraphError::Cycle { tiers, blocked_dependencies } = error else {
            panic!("contradictory task paths must fail graph compilation");
        };
        assert_eq!(tiers.len(), 2);
        assert_eq!(blocked_dependencies.len(), 2);
        let owners: BTreeSet<_> = blocked_dependencies.iter()
            .flat_map(|edge| edge.tasks.iter().cloned())
            .collect();
        assert_eq!(owners, ["first".to_string(), "second".to_string()].into_iter().collect());
        assert!(blocked_dependencies.iter().all(|edge| edge.tasks.len() == 1));
    }

    #[test]
    fn unknown_and_duplicate_tier_names_are_rejected() {
        let registry = registry(&["alpha"]);
        let unknown = task("first", &["missing"]);
        assert!(matches!(CascadeGraph::from_tasks(&[unknown], &registry),
            Err(CascadeGraphError::UnknownTier { .. })));
        assert!(Cascade::from_names(&[tname("alpha"), tname("alpha")]).is_err());
    }
}
