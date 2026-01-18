use crate::Storage;
use crate::io::{StateDict, TensorData};
use crate::nn::Module;
use crate::tensor::{RawTensor, Tensor, TensorOps};

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

    /// Forward pass for a single timestep
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape `[batch, input_size]`
    /// * `state` - Optional tuple of (h, c) where h and c are `[batch, hidden_size]`
    ///
    /// # Returns
    /// Tuple of `(h_next, c_next)` both of shape `[batch, hidden_size]`
    pub fn forward_step(
        &self,
        input: &Tensor,
        state: Option<(&Tensor, &Tensor)>,
    ) -> (Tensor, Tensor) {
        let batch_size = input.borrow().shape.first().copied().unwrap_or(1);

        // Initialize hidden and cell states if not provided
        let (h, c) = if let Some((h, c)) = state {
            (h.clone(), c.clone())
        } else {
            let h = RawTensor::zeros(&[batch_size, self.hidden_size]);
            let c = RawTensor::zeros(&[batch_size, self.hidden_size]);
            (h, c)
        };

        // Compute gates: input @ weight_ih + h @ weight_hh + bias
        let gates = input.matmul(&self.weight_ih.transpose());
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
        let i_gate = self.slice_gate(&gates, 0); // input gate
        let f_gate = self.slice_gate(&gates, 1); // forget gate
        let g_gate = self.slice_gate(&gates, 2); // cell gate
        let o_gate = self.slice_gate(&gates, 3); // output gate

        // Apply activations
        let i = i_gate.sigmoid();
        let f = f_gate.sigmoid();
        let g = g_gate.tanh();
        let o = o_gate.sigmoid();

        // Update cell state: c_next = f * c + i * g
        let c_next = f.elem_mul(&c).add(&i.elem_mul(&g));

        // Update hidden state: h_next = o * tanh(c_next)
        let h_next = o.elem_mul(&c_next.tanh());

        (h_next, c_next)
    }

    /// Helper to extract a gate from the combined gates tensor
    fn slice_gate(&self, gates: &Tensor, gate_idx: usize) -> Tensor {
        let start = gate_idx * self.hidden_size;
        let end = (gate_idx + 1) * self.hidden_size;

        // Extract columns [start:end] for all rows
        // gates shape: [batch, 4*hidden_size]
        let batch_size = gates.borrow().shape.first().copied().unwrap_or(1);
        let data = &gates.borrow().data;

        let mut gate_data = Vec::with_capacity(batch_size * self.hidden_size);
        for i in 0..batch_size {
            let row_start = i * 4 * self.hidden_size;
            gate_data.extend_from_slice(&data[row_start + start..row_start + end]);
        }

        RawTensor::new(
            gate_data,
            &[batch_size, self.hidden_size],
            gates.borrow().requires_grad,
        )
    }
}

impl Module for LSTMCell {
    fn forward(&self, x: &Tensor) -> Tensor {
        // For Module trait, just return hidden state
        let (h, _c) = self.forward_step(x, None);
        h
    }

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
            let mut t = self.weight_ih.borrow_mut();
            t.data = Storage::cpu(w.data.clone());
            t.shape.clone_from(&w.shape);
        }
        if let Some(w) = state.get("weight_hh") {
            let mut t = self.weight_hh.borrow_mut();
            t.data = Storage::cpu(w.data.clone());
            t.shape.clone_from(&w.shape);
        }
        if let Some(b) = state.get("bias_ih")
            && self.bias_ih.is_some()
        {
            let bias_tensor = self.bias_ih.as_ref().unwrap();
            let mut t = bias_tensor.borrow_mut();
            t.data = Storage::cpu(b.data.clone());
            t.shape.clone_from(&b.shape);
        }
        if let Some(b) = state.get("bias_hh")
            && self.bias_hh.is_some()
        {
            let bias_tensor = self.bias_hh.as_ref().unwrap();
            let mut t = bias_tensor.borrow_mut();
            t.data = Storage::cpu(b.data.clone());
            t.shape.clone_from(&b.shape);
        }
    }
}
