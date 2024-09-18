use crate::graph::Op;
use ndarray::ArrayD;
use std::cell::RefCell;
use std::rc::Rc;

/// Represents a node in the computational graph.
pub struct Node {
    pub value: RefCell<Option<ArrayD<f64>>>, // The value computed at this node.
    pub grad: RefCell<Option<ArrayD<f64>>>,  // The gradient of the output w.r.t this node.
    op: Option<Box<dyn Op>>,                 // The operation that computes this node's value.
    pub inputs: Vec<Rc<Node>>,               // The input nodes to this operation.
}

impl Node {
    /// Creates a new operation node with the given operation and input nodes.
    pub fn new(op: Box<dyn Op>, inputs: Vec<Rc<Node>>) -> Self {
        Self {
            value: RefCell::new(None),
            grad: RefCell::new(None),
            op: Some(op),
            inputs,
        }
    }

    /// Creates a new leaf node (e.g., input data or trainable parameters).
    pub fn new_leaf(value: ArrayD<f64>) -> Self {
        Self {
            value: RefCell::new(Some(value)),
            grad: RefCell::new(None),
            op: None,
            inputs: Vec::new(),
        }
    }

    /// Sets the value of a leaf node.
    pub fn set_value(&self, value: ArrayD<f64>) {
        *self.value.borrow_mut() = Some(value);
    }

    /// Resets the gradient stored in this node.
    pub fn zero_grad(&self) {
        *self.grad.borrow_mut() = None;
    }

    /// Performs the forward pass to compute the value of this node.
    pub fn forward(&self) {
        if let Some(ref op) = self.op {
            // Ensure all inputs have computed their values.
            for input in &self.inputs {
                input.forward();
            }

            // Collect input values.
            let input_values: Vec<&ArrayD<f64>> = self
                .inputs
                .iter()
                .map(|input| input.value.borrow().as_ref().unwrap())
                .collect();

            // Compute this node's value.
            let output = op.forward(&input_values);
            *self.value.borrow_mut() = Some(output);
        }
        // Leaf nodes already have their value set.
    }

    /// Performs the backward pass to compute gradients.
    pub fn backward(&self, grad: Option<ArrayD<f64>>) {
        // Accumulate gradients.
        {
            let mut self_grad = self.grad.borrow_mut();
            if let Some(ref existing_grad) = *self_grad {
                if let Some(new_grad) = grad {
                    *self_grad = Some(existing_grad + new_grad);
                }
            } else {
                *self_grad = grad;
            }
        }

        if let Some(ref op) = self.op {
            // Collect input values.
            let input_values: Vec<&ArrayD<f64>> = self
                .inputs
                .iter()
                .map(|input| input.value.borrow().as_ref().unwrap())
                .collect();

            // Get the gradient w.r.t. this node's output.
            let grad_output = self.grad.borrow().as_ref().unwrap();

            // Compute gradients w.r.t inputs.
            let input_grads = op.backward(&input_values, grad_output);

            // Backpropagate to input nodes.
            for (input, input_grad) in self.inputs.iter().zip(input_grads.into_iter()) {
                input.backward(Some(input_grad));
            }
        }
    }
}
