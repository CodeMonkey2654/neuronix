mod tensor;
mod layer;
mod optimizer;
mod neural_network;

use crate::tensor::Tensor;
use crate::layer::DenseLayer;
use crate::optimizer::{Optimizer, SGDOptimizer};
use crate::neural_network::NeuralNetwork;

fn main() {
    let mut model = NeuralNetwork::new(vec![
        Box::new(DenseLayer::new(3, 4)),
        Box::new(DenseLayer::new(4, 2)),
    ]);

    let input = Tensor::new(vec![1.0, 2.0, 3.0], &[1, 3]);
    let target_output = Tensor::new(vec![0.0, 1.0], &[1, 2]);

    // Forward pass
    let output = model.forward(&input);
    println!("Output: {:?}", output.data());

    // Compute loss (e.g., MSE)
    let loss = (output.clone() - target_output.clone()).pow(2.0).sum();

    // Backward pass (dummy gradient of loss w.r.t. output)
    let grad_output = output - target_output;
    model.backward(&input, &grad_output);

    // Get parameters and their gradients
    let (params, grads) = model.get_params_and_grads();

    // Optimizer step (update parameters using gradients)
    let mut optimizer = SGDOptimizer::new(0.01);
    optimizer.step(params.as_mut_slice(), grads.as_slice());

    println!("Updated Weights: {:?}", params[0].data());
}
