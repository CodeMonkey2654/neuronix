mod graph;
mod layers;
mod losses;
mod optimizers;
mod models;
mod utils;

use std::rc::Rc;
use crate::graph::Node;
use crate::layers::{linear::Linear, activation::relu::ReLU};
use crate::losses::cross_entropy::CrossEntropyLoss;
use crate::layers::layer::Layer;
use crate::optimizers::optimizer::Optimizer;
use crate::optimizers::sgd::SGD;
use crate::losses::loss::Loss;
use ndarray::ArrayD;

fn main() {
    // Create x and y to be a simple linear function [y = 2x + 1]
    let input_dim = 1;
    let hidden_dim = 100;
    let output_dim = 1;

    // Input data
    let x = Rc::new(Node::new_leaf(ArrayD::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap()));
    let y = Rc::new(Node::new_leaf(ArrayD::from_shape_vec((5, 1), vec![3.0, 5.0, 7.0, 9.0, 11.0]).unwrap()));

    // Initialize layers
    let linear1 = Linear::new(input_dim, hidden_dim);
    let relu = ReLU::new();
    let linear2 = Linear::new(hidden_dim, output_dim);

    // Initialize optimizer
    let mut optimizer = SGD::new(0.01, 0.9); // Learning rate and momentum

    // Training loop
    for epoch in 0..1000 {
        // Forward pass
        let linear1_output = linear1.forward(x.clone());
        let relu_output = relu.forward(linear1_output);
        let linear2_output = linear2.forward(relu_output);

        // Compute loss
        let loss_fn = CrossEntropyLoss::new();
        let loss = loss_fn.compute(linear2_output.clone(), y.clone());

        // Backward pass
        loss.backward(None); // Start backpropagation from the loss node

        // Update parameters
        optimizer.step(&[linear1.weight.clone(), linear1.bias.clone(), linear2.weight.clone(), linear2.bias.clone()]);

        // Print loss every 100 epochs
        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {:?}", epoch, loss.value.borrow());
        }
    }

    // Final output after training
    let final_output = linear2.forward(relu.forward(linear1.forward(x.clone())));
    println!("Final Output: {:?}", final_output.value.borrow());
}
