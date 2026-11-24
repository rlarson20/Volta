//! Atomic state-dict loading: collect every write, check them all, then make them (#65).

use super::Module;
use crate::io::{StateDict, TensorData};
use crate::tensor::Tensor;

/// The writes a state-dict load would make, collected before any is made.
///
/// A [`Module`] fills one in [`Module::plan_load`]: it names each tensor it can load
/// with [`Self::param`] and hands each sub-module to [`Self::child`]. The plan holds
/// handles to the destination tensors and references to the entries of the state dict,
/// so collecting copies no weights. [`Module::load_state_dict_partial`] and
/// [`Module::load_state_dict`] then check every write before making any, so a bad entry
/// leaves the module unchanged (#65).
pub struct LoadPlan<'a> {
    /// The state dict being loaded.
    state: &'a StateDict,
    /// The key prefix of the module being visited (`"2.ffn."`), empty at the top.
    prefix: String,
    /// Every key the visited modules can load, present in `state` or not, in visit order.
    wanted: Vec<String>,
    /// The writes: the entry's full key, its destination, and the entry.
    writes: Vec<(String, Tensor, &'a TensorData)>,
}

/// The prefix being visited and the keys planned, not the entries' bytes.
impl std::fmt::Debug for LoadPlan<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let writes: Vec<&str> = self.writes.iter().map(|(k, _, _)| k.as_str()).collect();
        f.debug_struct("LoadPlan")
            .field("prefix", &self.prefix)
            .field("wanted", &self.wanted)
            .field("writes", &writes)
            .finish_non_exhaustive()
    }
}

impl<'a> LoadPlan<'a> {
    /// An empty plan for loading `state`.
    pub(crate) const fn new(state: &'a StateDict) -> Self {
        Self {
            state,
            prefix: String::new(),
            wanted: Vec::new(),
            writes: Vec::new(),
        }
    }

    /// Plans the load of `module` from `state`.
    pub(crate) fn of<M: Module + ?Sized>(module: &M, state: &'a StateDict) -> Self {
        let mut plan = Self::new(state);
        module.plan_load(&mut plan);
        plan
    }

    /// Names `tensor` as the destination of the entry `name` of the module being
    /// visited (`"weight"`; the plan adds the prefix of any containers around it). A
    /// strict load expects the entry; a lenient one loads it if `state` has it.
    pub fn param(&mut self, name: &str, tensor: &Tensor) {
        let key = format!("{}{name}", self.prefix);
        if let Some(entry) = self.state.get(&key) {
            self.writes.push((key.clone(), tensor.clone(), entry));
        }
        self.wanted.push(key);
    }

    /// Visits `child`, a sub-module whose keys sit under `prefix`, which ends in `.`
    /// (`"ffn."` turns the child's `expand.weight` into `ffn.expand.weight`).
    pub fn child<M: Module + ?Sized>(&mut self, prefix: &str, child: &M) {
        let outer = self.prefix.len();
        self.prefix.push_str(prefix);
        child.plan_load(self);
        self.prefix.truncate(outer);
    }

    /// Whether `state` has an entry under `prefix` inside the module being visited, not
    /// counting a key that is the bare prefix. `Sequential` uses it to read a named
    /// layer's entries under its index when there are none under its name.
    #[must_use]
    pub fn has_entries_under(&self, prefix: &str) -> bool {
        let full = format!("{}{prefix}", self.prefix);
        // Keys sort lexicographically, so the ones under `full` are one contiguous run.
        self.state
            .range(full.clone()..)
            .take_while(|(key, _)| key.starts_with(full.as_str()))
            .any(|(key, _)| key.len() > full.len())
    }

    /// `Err(StateKeyMismatch)` unless the keys of `state` are exactly the keys the
    /// visited modules can load: the check of a strict load.
    pub(crate) fn check_keys(&self) -> crate::Result<()> {
        let wanted: std::collections::BTreeSet<&str> =
            self.wanted.iter().map(String::as_str).collect();
        let missing: Vec<String> = wanted
            .iter()
            .filter(|k| !self.state.contains_key(**k))
            .map(|k| (*k).to_string())
            .collect();
        let unexpected: Vec<String> = self
            .state
            .keys()
            .filter(|k| !wanted.contains(k.as_str()))
            .cloned()
            .collect();
        if missing.is_empty() && unexpected.is_empty() {
            Ok(())
        } else {
            Err(crate::VoltaError::StateKeyMismatch {
                missing,
                unexpected,
            })
        }
    }

    /// Drops the writes whose key `skip` returns true for
    /// ([`crate::io::load_state_dict_checked`] skips shape mismatches).
    pub(crate) fn skip(&mut self, mut skip: impl FnMut(&str) -> bool) {
        self.writes.retain(|(key, _, _)| !skip(key));
    }

    /// Checks every write, then makes them all, in visit order. A tensor named under two
    /// keys (a tied weight) is written once per key, so the last one wins.
    ///
    /// # Errors
    ///
    /// The first write's error from [`Tensor::check_load`]; nothing is written then.
    ///
    /// # Panics
    ///
    /// If a write fails after every check passed (a GPU upload failure), the module is
    /// left partly loaded and this panics rather than return an `Err` that would claim
    /// otherwise.
    pub(crate) fn apply(self) -> crate::Result<()> {
        for (key, tensor, entry) in &self.writes {
            tensor.check_load(key, entry)?;
        }
        for (key, tensor, entry) in &self.writes {
            tensor
                .load_weight(key, entry)
                .unwrap_or_else(|e| panic!("load_state_dict: '{key}' passed its checks: {e}"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Linear, Sequential};

    /// An `f32` entry of `shape` filled with `v`.
    fn entry(v: f32, shape: &[usize]) -> TensorData {
        TensorData::from_f32(vec![v; shape.iter().product()], shape.to_vec())
    }

    #[test]
    fn has_entries_under_ignores_the_bare_prefix_and_lookalikes() {
        let state: StateDict = ["a", "a.", "ab.w", "a-w", "b.w"]
            .into_iter()
            .map(|k| (k.to_string(), entry(1.0, &[1])))
            .collect();
        let plan = LoadPlan::new(&state);
        assert!(!plan.has_entries_under("a."));
        assert!(plan.has_entries_under("b."));
        assert!(!plan.has_entries_under("c."));
    }

    #[test]
    fn a_plan_wants_exactly_the_state_dict_keys_of_every_layer() {
        use crate::nn::layers::*;
        use crate::nn::transformer::*;
        let modules: Vec<Box<dyn Module>> = vec![
            Box::new(Linear::new(3, 2, true)),
            Box::new(Linear::new(3, 2, false)),
            Box::new(Conv2d::new(2, 3, 3, 1, 1, true)),
            Box::new(ConvTranspose2d::new(2, 3, 3, 1, 1, false)),
            Box::new(Embedding::new(5, 4)),
            Box::new(LayerNorm::new(vec![4])),
            Box::new(BatchNorm1d::new(3)),
            Box::new(BatchNorm2d::new(3)),
            Box::new(LSTMCell::new(3, 4, true)),
            Box::new(MultiHeadAttention::new(8, 2, 0.0)),
            Box::new(FeedForward::new(8, 2, 0.0)),
            Box::new(TransformerBlock::new(8, 2, 2, 0.0)),
            Box::new(PositionalEncoding::new(
                8,
                6,
                PositionalEncodingType::Learned,
            )),
            Box::new(PositionalEncoding::new(
                8,
                6,
                PositionalEncodingType::Sinusoidal,
            )),
            Box::new(
                Sequential::builder()
                    .add_named("fc", Box::new(Linear::new(2, 2, true)))
                    .add_unnamed(Box::new(ReLU))
                    .add_unnamed(Box::new(Linear::new(2, 1, true)))
                    .build(),
            ),
        ];
        for m in &modules {
            // Against the module's own state dict, so `Sequential` reads `fc` by name.
            let own = m.state_dict();
            let plan = LoadPlan::of(m.as_ref(), &own);
            let mut wanted = plan.wanted.clone();
            wanted.sort();
            let keys: Vec<&String> = own.keys().collect();
            assert_eq!(wanted.iter().collect::<Vec<_>>(), keys);
            assert_eq!(plan.writes.len(), keys.len(), "every own entry is written");
            plan.check_keys()
                .expect("a module's own state dict loads strictly");
        }
    }

    #[test]
    fn a_bad_entry_anywhere_leaves_every_layer_unchanged() {
        let mut model = Sequential::new(vec![
            Box::new(Linear::new(2, 2, true)),
            Box::new(Linear::new(2, 2, true)),
        ]);
        let before = model.state_dict();
        let mut state = StateDict::new();
        state.insert("0.weight".into(), entry(7.0, &[2, 2]));
        state.insert("0.bias".into(), entry(7.0, &[2]));
        state.insert("1.weight".into(), entry(7.0, &[2, 2]));
        state.insert("1.bias".into(), entry(7.0, &[1, 2])); // wrong: Linear's bias is [2]
        let err = model.load_state_dict_partial(&state).unwrap_err();
        assert!(
            matches!(&err, crate::VoltaError::StateShapeMismatch { key, .. } if key == "1.bias"),
            "{err:?}"
        );
        assert_eq!(model.state_dict(), before);
    }
}
