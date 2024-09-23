pub mod tensor;
pub mod graph;
pub mod variable;
pub mod node;
pub mod op;
pub mod layer;
pub mod optimizer;
use crate::layer::{Layer, Dense};
use crate::optimizer::{Optimizer, SGD};
use crate::op::MeanSquaredError;
use crate::graph::ComputationGraph;
use crate::tensor::Tensor;
use std::rc::Rc;
use crate::tensor::TensorError;
use std::error::Error;
use std::fmt;

// Wrapper error type
#[derive(Debug)]
struct WrapperError(TensorError);

impl fmt::Display for WrapperError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl Error for WrapperError {}

impl From<TensorError> for WrapperError {
    fn from(err: TensorError) -> Self {
        WrapperError(err)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Define the true linear function: y = 2x + 1
    let true_weight = 2.0;
    let true_bias = 1.0;

    // Create a computation graph
    let mut graph = ComputationGraph::new();

    // Initialize the linear layer
    let mut linear_layer = Dense::new(1, 1, None);
    let _ = linear_layer.initialize(&mut graph);

    // Create an optimizer
    let mut optimizer = SGD::new(0.01); // Learning rate of 0.01

    // Training loop
    let num_epochs = 1000;
    let batch_size = 32;

    for epoch in 0..num_epochs {
        // Generate random input data
        let x = Tensor::randn(&[batch_size, 1], 0.0, 1.0).map_err(WrapperError)?;
        let x_var = graph.create_variable(x.clone(), false);

        // Compute true y values
        let y_true = &(&x * &Tensor::scalar(true_weight)) + &Tensor::scalar(true_bias);
        let y_true_var = graph.create_variable(y_true.clone(), false);

        // Forward pass
        let y_pred = linear_layer.forward(&x_var, &mut graph);
        // Compute loss
        let mse_op = Rc::new(MeanSquaredError);
        let loss_node = graph.add_node(mse_op, vec![&y_pred.unwrap(), &y_true_var]);
        let loss = graph.forward(&loss_node.node);

        // Backward pass
        let _ = graph.backward(&loss_node.node);

        // Update parameters
        optimizer.step(&mut graph)?;

        // Zero gradients
        optimizer.zero_grad();

        // Print progress
        if (epoch + 1) % 100 == 0 {
            println!("Epoch [{}/{}], Loss: {:.4}", epoch + 1, num_epochs, loss.unwrap().to_scalar().unwrap());
        }
    }

    // Evaluate the trained model
    let test_x = Tensor::randn(&[1, 1], 0.0, 1.0).unwrap();
    let test_x_var = graph.create_variable(test_x.clone(), false);
    let test_y_pred = linear_layer.forward(&test_x_var, &mut graph).unwrap();

    println!("True function: y = {}x + {}", true_weight, true_bias);
    println!("Learned function: y = {:.4}x + {:.4}", 
             linear_layer.weights.value().unwrap().to_scalar().unwrap(),
             linear_layer.bias.value().unwrap().to_scalar().unwrap());
    println!("Test input: {:.4}", test_x.to_scalar().unwrap());
    println!("Predicted output: {:.4}", test_y_pred.value().unwrap().to_scalar().unwrap());

    Ok(())
}