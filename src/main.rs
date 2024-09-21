pub mod layer;
pub mod optimizer;
pub mod op;
pub mod tensor;
pub mod variable;
pub mod node;
pub mod graph;
use crate::tensor::Tensor;
use crate::variable::Variable;
use std::rc::Rc;
use crate::layer::{Layer, Sequential, Dense};
use crate::optimizer::{Optimizer, SGD};
use crate::op::{Op, MeanSquaredError};
use crate::graph::ComputationGraph;
use ndarray::Array2;
use rand::Rng;

fn generate_linear_noise(num_samples: usize, slope: f32, intercept: f32, noise_std: f32) -> (Tensor, Tensor) {
    let mut rng = rand::thread_rng();
    let x: Vec<f32> = (0..num_samples).map(|i| i as f32 / num_samples as f32).collect();
    let y: Vec<f32> = x.iter()
        .map(|&xi| slope * xi + intercept + rng.gen_range(-noise_std..noise_std))
        .collect();

    let x_tensor = Tensor::new_f32(Array2::from_shape_vec((num_samples, 1), x).unwrap().into_dyn());
    let y_tensor = Tensor::new_f32(Array2::from_shape_vec((num_samples, 1), y).unwrap().into_dyn());

    (x_tensor, y_tensor)
}

fn main() -> Result<(), String> {
    let mut graph = ComputationGraph::new();

    // Generate linear noise data
    let (x, y) = generate_linear_noise(100, 2.0, 1.0, 0.1);

    // Create model - 3 Layers and Loss Function
    let mut layer1 = Dense::new(1, 10, &mut graph);
    let mut layer2 = Dense::new(10, 10, &mut graph);
    let mut layer3 = Dense::new(10, 1, &mut graph);
    let loss_op = MeanSquaredError;

    let _ = layer1.initialize(&mut graph);
    let _ = layer2.initialize(&mut graph);
    let _ = layer3.initialize(&mut graph);

    // Forward Pass
    let output = graph.forward(x)?;
    let loss = loss_op.forward(&[output.clone(), y.clone()]);
    // Tensor to printable string
    output.clone().display();
    loss.unwrap().display();
    Ok(())
}