// src/models/mlp.rs

use std::rc::Rc;
use crate::graph::Node;
use crate::layers::{linear::Linear, activation::relu::ReLU};
use crate::layers::layer::Layer;

pub struct MLP {
    layers: Vec<Linear>,
    activations: Vec<ReLU>,
}

impl MLP {
    pub fn new(input_dim: usize, hidden_dims: Vec<usize>, output_dim: usize) -> Self {
        let mut layers = Vec::new();
        let mut activations = Vec::new();

        let mut prev_dim = input_dim;
        for &hidden_dim in &hidden_dims {
            layers.push(Linear::new(prev_dim, hidden_dim));
            activations.push(ReLU::new());
            prev_dim = hidden_dim;
        }
        layers.push(Linear::new(prev_dim, output_dim));

        Self { layers, activations }
    }

    pub fn forward(&self, input: Rc<Node>) -> Rc<Node> {
        let mut x = input;
        for (layer, activation) in self.layers.iter().zip(self.activations.iter()) {
            x = activation.forward(layer.forward(x));
        }
        // Apply the last layer without activation
        x = self.layers.last().unwrap().forward(x);
        x
    }
}
