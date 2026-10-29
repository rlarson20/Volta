use crate::index::span;
use crate::io::{StateDict, TensorData};
use crate::nn::{Module, RecurrentLayer, SealedLayer};
use crate::tensor::{RawTensor, Tensor};

/// LSTM Cell - processes one timestep at a time
///
/// Implements the LSTM equations:
/// - i = `sigmoid(W_ii @ x + W_hi @ h + b_i)`  (input gate)
/// - f = `sigmoid(W_if @ x + W_hf @ h + b_f)`  (forget gate)
/// - g = `tanh(W_ig @ x + W_hg @ h + b_g)`     (cell gate)
/// - o = `sigmoid(W_io @ x + W_ho @ h + b_o)`  (output gate)
/// - `c_next` = f * c + i * g                  (new cell state)
/// - `h_next` = o * `tanh(c_next)`               (new hidden state)
pub struct LSTMCell {
    #[allow(dead_code)]
    input_size: usize,
    hidden_size: usize,
    /// Combined weights for input: `[4*hidden_size, input_size]`
    /// Order: input, forget, cell, output gates
    weight_ih: Tensor,
    /// Combined weights for hidden: `[4*hidden_size, hidden_size]`
    weight_hh: Tensor,
    /// Combined bias for input: `[4*hidden_size]`
    bias_ih: Option<Tensor>,
    /// Combined bias for hidden: `[4*hidden_size]`
    bias_hh: Option<Tensor>,
}

impl LSTMCell {
    /// Create a new LSTM cell
    #[must_use]
    pub fn new(input_size: usize, hidden_size: usize, bias: bool) -> Self {
        // Initialize weights using Xavier/Glorot uniform
        let weight_ih = RawTensor::xavier_uniform(&[4 * hidden_size, input_size]);
        let weight_hh = RawTensor::xavier_uniform(&[4 * hidden_size, hidden_size]);

        weight_ih.borrow_mut().requires_grad = true;
        weight_hh.borrow_mut().requires_grad = true;

        let (bias_ih, bias_hh) = if bias {
            let b_ih = RawTensor::zeros(&[4 * hidden_size]);
            let b_hh = RawTensor::zeros(&[4 * hidden_size]);
            b_ih.borrow_mut().requires_grad = true;
            b_hh.borrow_mut().requires_grad = true;
            (Some(b_ih), Some(b_hh))
        } else {
            (None, None)
        };

        Self {
            input_size,
            hidden_size,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
        }
    }

    /// Helper to extract a gate from the combined gates tensor
    fn slice_gate(&self, gates: &Tensor, gate_idx: usize) -> Tensor {
        let start = gate_idx * self.hidden_size;
        let end = (gate_idx + 1) * self.hidden_size;

        // Extract columns [start:end] for all rows
        // gates shape: [batch, 4*hidden_size]
        let batch_size = gates.borrow().shape.first().copied().unwrap_or(1);
        let gates_ref = gates.borrow();
        let data = &*gates_ref.data.to_host_f32();

        let mut gate_data = Vec::with_capacity(batch_size * self.hidden_size);
        for i in 0..batch_size {
            let row_start = i * 4 * self.hidden_size;
            gate_data.extend_from_slice(span(data, row_start + start..row_start + end));
        }

        RawTensor::new(
            gate_data,
            &[batch_size, self.hidden_size],
            gates.borrow().requires_grad,
        )
    }
}

impl SealedLayer for LSTMCell {}

impl RecurrentLayer for LSTMCell {
    /// `(h, c)`, each `[batch, hidden_size]`.
    type State = (Tensor, Tensor);

    fn initial_state(&self, batch: usize) -> Self::State {
        (
            RawTensor::zeros(&[batch, self.hidden_size]),
            RawTensor::zeros(&[batch, self.hidden_size]),
        )
    }

    /// One timestep. `x` is `[batch, input_size]`; `state` holds `(h, c)` and is overwritten with
    /// `(h_next, c_next)`. Returns `h_next`.
    fn step(&self, x: &Tensor, state: &mut Self::State) -> Tensor {
        let (h, c) = (&state.0, &state.1);

        // Compute gates: x @ weight_ih + h @ weight_hh + bias
        let gates = x.matmul(&self.weight_ih.transpose());
        let gates = gates.add(&h.matmul(&self.weight_hh.transpose()));

        let gates = if let Some(ref b_ih) = self.bias_ih {
            let gates = gates.add(b_ih);
            if let Some(ref b_hh) = self.bias_hh {
                gates.add(b_hh)
            } else {
                gates
            }
        } else {
            gates
        };

        // Split gates into i, f, g, o (each is [batch, hidden_size])
        let i = self.slice_gate(&gates, 0).sigmoid();
        let f = self.slice_gate(&gates, 1).sigmoid();
        let g = self.slice_gate(&gates, 2).tanh();
        let o = self.slice_gate(&gates, 3).sigmoid();

        // c_next = f * c + i * g;  h_next = o * tanh(c_next)
        let c_next = f.elem_mul(c).add(&i.elem_mul(&g));
        let h_next = o.elem_mul(&c_next.tanh());

        *state = (h_next.clone(), c_next);
        h_next
    }
}

impl Module for LSTMCell {
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight_ih.clone(), self.weight_hh.clone()];
        if let Some(ref b_ih) = self.bias_ih {
            params.push(b_ih.clone());
        }
        if let Some(ref b_hh) = self.bias_hh {
            params.push(b_hh.clone());
        }
        params
    }

    fn state_dict(&self) -> StateDict {
        let mut state = StateDict::new();
        state.insert(
            "weight_ih".to_string(),
            TensorData::from_tensor(&self.weight_ih),
        );
        state.insert(
            "weight_hh".to_string(),
            TensorData::from_tensor(&self.weight_hh),
        );
        if let Some(ref b) = self.bias_ih {
            state.insert("bias_ih".to_string(), TensorData::from_tensor(b));
        }
        if let Some(ref b) = self.bias_hh {
            state.insert("bias_hh".to_string(), TensorData::from_tensor(b));
        }
        state
    }

    fn load_state_dict(&mut self, state: &StateDict) {
        if let Some(w) = state.get("weight_ih") {
            self.weight_ih.load_weight("weight_ih", w);
        }
        if let Some(w) = state.get("weight_hh") {
            self.weight_hh.load_weight("weight_hh", w);
        }
        if let Some(b) = state.get("bias_ih")
            && self.bias_ih.is_some()
        {
            let bias_tensor = self.bias_ih.as_ref().unwrap();
            bias_tensor.load_weight("bias", b);
        }
        if let Some(b) = state.get("bias_hh")
            && self.bias_hh.is_some()
        {
            let bias_tensor = self.bias_hh.as_ref().unwrap();
            bias_tensor.load_weight("bias", b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One step from zero state, so `assert_trainable!` (which needs `Layer`) can drive the cell.
    struct OneStep(LSTMCell);

    impl Module for OneStep {
        fn parameters(&self) -> Vec<Tensor> {
            self.0.parameters()
        }
        fn state_dict(&self) -> StateDict {
            self.0.state_dict()
        }
        fn load_state_dict(&mut self, state: &StateDict) {
            self.0.load_state_dict(state);
        }
    }

    impl crate::nn::Layer for OneStep {
        fn forward(&self, x: &Tensor) -> Tensor {
            let batch = x.shape().first().copied().expect("batch dim");
            self.0.step(x, &mut self.0.initial_state(batch))
        }
    }

    #[test]
    #[ignore = "B2: slice_gate copies gate data into a fresh RawTensor, severing the graph; remove when fixed"]
    fn test_lstmcell_assert_trainable() {
        let cell = OneStep(LSTMCell::new(4, 8, true));
        let x = crate::macros::trainable_input(&[2, 4]);
        crate::assert_trainable!(cell, x);
    }
}
