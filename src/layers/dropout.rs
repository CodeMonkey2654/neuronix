// src/layers/dropout.rs

use std::rc::Rc;
use crate::graph::{Node, op::Op};
use crate::layers::Layer;
use ndarray::ArrayD;
use rand::Rng;

pub struct Dropout {
    pub rate: f64, // Dropout rate
}

impl Dropout {
    pub fn new(rate: f64) -> Self {
        assert!(rate >= 0.0 && rate < 1.0, "Dropout rate must be in [0, 1)");
        Self { rate }
    }

    pub fn forward(&self, input: Rc<Node>, training: bool) -> Rc<Node> {
        if training {
            let mask = Node::new_op(DropoutMaskOp::new(self.rate), vec![input.clone()]);
            Node::new_op(MultiplyOp, vec![input, mask])
        } else {
            // During evaluation, scale the activations
            let scale = 1.0 - self.rate;
            Node::new_op(ScalarMulOp::new(scale), vec![input])
        }
    }
}

impl Layer for Dropout {
    fn as_op_node(&self, input: Rc<Node>) -> Rc<Node> {
        // You might need to pass the training flag from the model or global state
        let training = true; // Placeholder
        self.forward(input, training)
    }
}
