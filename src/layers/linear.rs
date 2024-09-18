use crate::graph::{Node};
use crate::graph::ops::LinearOp;
use crate::utils::initialization::initialize_parameters;
use ndarray::prelude::*;
use std::rc::Rc;

/// Linear (fully connected) layer
pub struct Linear {
    pub weight: Rc<Node>,
    pub bias: Rc<Node>,
}

impl Linear {
    /// Creates a new Linear layer with the specified input and output dimensions
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        // Initialize weights and biases using a utility function
        let weight_array = initialize_parameters((input_dim, output_dim));
        let bias_array = Array2::<f64>::zeros((1, output_dim));

        let weight = Rc::new(Node::new_leaf(weight_array.into_dyn()));
        let bias = Rc::new(Node::new_leaf(bias_array.into_dyn()));

        Self { weight, bias }
    }

    /// Applies the Linear layer to an input node
    pub fn forward(&self, input: Rc<Node>) -> Rc<Node> {
        let linear_op = Box::new(LinearOp);
        Rc::new(Node::new(
            linear_op,
            vec![input, self.weight.clone(), self.bias.clone()],
        ))
    }
}



