// Unordered catalog of tier definitions. Concrete instances belong to CascadeGraph.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use super::tier::{Params, Tier, TierName};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    DuplicateName(TierName),
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateName(name) => write!(f, "tier {name} is registered twice"),
        }
    }
}

impl Error for RegistryError {}

/// Available tier definitions and their declarative parameters.
/// This catalog deliberately contains no TierInstance and assigns no TierId.
#[derive(Default)]
pub struct TierRegistry {
    entries: BTreeMap<TierName, (Tier, Params)>,
    #[cfg(test)]
    fakes: BTreeMap<TierName, std::sync::Arc<dyn super::tier::FakeEvaluate>>,
}

impl TierRegistry {
    pub fn new() -> Self { Self::default() }

    pub fn register(
        &mut self,
        name: TierName,
        tier: Tier,
        params: Params,
    ) -> Result<(), RegistryError> {
        if self.entries.contains_key(&name) {
            return Err(RegistryError::DuplicateName(name));
        }
        self.entries.insert(name, (tier, params));
        Ok(())
    }

    pub fn get(&self, name: &TierName) -> Option<(&Tier, &Params)> {
        self.entries.get(name).map(|(tier, params)| (tier, params))
    }

    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    #[cfg(test)]
    pub fn len(&self) -> usize { self.entries.len() }

    // Test adapters are injected in place of materialized production instances.
    #[cfg(test)]
    pub fn register_fake(
        &mut self,
        name: TierName,
        tier: Tier,
        params: Params,
        fake: std::sync::Arc<dyn super::tier::FakeEvaluate>,
    ) -> Result<(), RegistryError> {
        self.register(name.clone(), tier, params)?;
        self.fakes.insert(name, fake);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn fake(
        &self,
        name: &TierName,
    ) -> Option<std::sync::Arc<dyn super::tier::FakeEvaluate>> {
        self.fakes.get(name).cloned()
    }

    #[cfg(test)]
    pub fn of(
        entries: Vec<(
            TierName,
            Tier,
            Params,
            std::sync::Arc<dyn super::tier::FakeEvaluate>,
        )>,
    ) -> Self {
        let mut registry = Self::new();
        for (name, tier, params, fake) in entries {
            registry.register_fake(name, tier, params, fake)
                .expect("test tier names must be unique");
        }
        registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::tier::{tname, ModelFormat, RequestShape, Role, Transport};

    fn params() -> Params {
        Params {
            model_format: Some(ModelFormat::Onnx),
            model: Some("opaque-model-config".into()),
            config: None,
            endpoint: Some("http://127.0.0.1:9".into()),
            timeout_ms: Some(250),
            enabled: None,
        }
    }

    #[test]
    fn registry_is_an_unordered_catalog_of_definitions() {
        let mut registry = TierRegistry::new();
        let encoder = Tier {
            exex: Role::Encoder,
            transport: Transport::Grpc,
            request_shape: Some(RequestShape::Score),
        };
        registry.register(tname("encoder"), encoder, params()).unwrap();
        registry.register(tname("unused"), encoder, params()).unwrap();

        assert_eq!(registry.len(), 2);
        assert!(registry.get(&tname("encoder")).is_some());
        assert!(registry.get(&tname("unused")).is_some());
    }

    #[test]
    fn duplicate_tier_names_are_rejected() {
        let mut registry = TierRegistry::new();
        let tier = Tier {
            exex: Role::Rules,
            transport: Transport::InProcess,
            request_shape: None,
        };
        let make_params = || Params {
            model_format: None, model: None, config: Some("config/tier1.toml".into()),
            endpoint: None, timeout_ms: None, enabled: None,
        };
        registry.register(tname("rules"), tier, make_params()).unwrap();
        assert_eq!(
            registry.register(tname("rules"), tier, make_params()),
            Err(RegistryError::DuplicateName(tname("rules"))),
        );
    }
}
