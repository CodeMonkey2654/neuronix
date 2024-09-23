mod tensor;
mod variable;
mod graph;
mod op;
mod optimizer;
mod node;

use std::rc::Rc;
use crate::tensor::Tensor;
use crate::graph::ComputationGraph;
use crate::op::{Add, Multiply, Linear};
use crate::optimizer::{Optimizer, SGD};

fn main() {
    // Create a computation graph
    let mut graph = ComputationGraph::new();

    // Create input variables
    let x = graph.create_variable(Tensor::new(&[1, 1], vec![1.0].as_slice(), "f32").unwrap(), true);
    let y_true = graph.create_variable(Tensor::new(&[1, 1], vec![2.0].as_slice(), "f32").unwrap(), true);

    // Create weights and bias
    let w = graph.create_variable(Tensor::new(&[1, 1], vec![0.5].as_slice(), "f32").unwrap(), true);
    let b = graph.create_variable(Tensor::new(&[1], vec![0.0].as_slice(), "f32").unwrap(), true);

    // Create a linear operation
    let linear_op = Rc::new(Linear);
    let y_pred = graph.add_node_with_variables(linear_op, vec![&x], vec![w.clone(), b.clone()]);

    // Define loss function (mean squared error)
    let diff = graph.add_node(Rc::new(Add), vec![&y_pred, &y_true]);
    let loss = graph.add_node(Rc::new(Multiply), vec![&diff, &diff]);

    // Create an optimizer
    let mut optimizer = SGD::new(0.01);

    // Training loop
    for _ in 0..100 {
        // Forward pass
        graph.forward(&loss.node).unwrap();

        // Backward pass
        graph.backward(&loss.node).unwrap();

        // Update weights and bias
        optimizer.step(&mut graph).unwrap();

        // Zero gradients
        optimizer.zero_grad();
    }

    // Print final weights and bias
    println!("Final weights: {:?}", w.value().unwrap());
    println!("Final bias: {:?}", b.value().unwrap());
}
