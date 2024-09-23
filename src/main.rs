mod tensor;
mod variable;
mod graph;
mod op;
mod optimizer;
mod node;
mod regularization;
mod loss;

use crate::tensor::Tensor;
use crate::graph::ComputationGraph;
use crate::op::{Add, Linear, ReLU, Softmax};
use crate::variable::Variable;
use crate::regularization::L2Regularization;
use crate::loss::MeanSquaredError;
use crate::optimizer::{Optimizer, SGD};
use std::rc::Rc;

fn main() {
    // define variables to train
    let x = Tensor::new(&[2], &[1.0, 2.0], "f32").unwrap();
    let y = Tensor::new(&[2], &[3.0, 4.0], "f32").unwrap(); 
    let variable_y = Variable::from_tensor(3, y, false);

    let w = Variable::from_tensor(1, Tensor::new(&[2, 2], &[1.0, 1.0, 1.0, 1.0], "f32").unwrap(), true);
    let b = Variable::from_tensor(2, Tensor::new(&[2], &[0.0, 0.0], "f32").unwrap(), true);
    let mut optimizer = SGD::new(0.01);
    let mut graph = ComputationGraph::new();

    // create linear node
    let linear_op = Rc::new(Linear { w: w.clone(), b: b.clone() });
    let linear_node = graph.add_node(linear_op, vec![&w, &b]);

    // have w go to regularization
    let l2_reg_op = Rc::new(L2Regularization { lambda: 0.01 });
    let reg_node = graph.add_node(l2_reg_op, vec![&w]);

    // have linear go to relu
    let relu_op = Rc::new(ReLU);
    let relu_node = graph.add_node(relu_op, vec![&linear_node]);

    // have relu go to a new linear node that takes in M and c as weights and biases
    let m = Variable::from_tensor(3, Tensor::new(&[2, 2], &[1.0, 1.0, 1.0, 1.0], "f32").unwrap(), true);
    let c = Variable::from_tensor(4, Tensor::new(&[2], &[0.0, 0.0], "f32").unwrap(), true);
    let linear_op2 = Rc::new(Linear { w: m.clone(), b: c.clone() });
    let linear_node2 = graph.add_node(linear_op2, vec![&relu_node, &m, &c]);

    // have linear go to softmax and M go to regularization again (l2)
    let l2_reg_op2 = Rc::new(L2Regularization { lambda: 0.01 });
    let reg_node2 = graph.add_node(l2_reg_op2, vec![&m]);
    let softmax_op = Rc::new(Softmax);
    let softmax_node = graph.add_node(softmax_op, vec![&linear_node2]);

    // have both regularization nodes go to Add or sum
    let add_op = Rc::new(Add);
    let add_node = graph.add_node(add_op.clone(), vec![&reg_node, &reg_node2]);

    // have softmax go to MSE
    let mse_op = Rc::new(MeanSquaredError);
    let mse_node = graph.add_node(mse_op, vec![&softmax_node, &variable_y]);

    // add regularization outputs and loss through add
    let add_node2 = graph.add_node(add_op, vec![&add_node, &mse_node]);

    // training loop
    for epoch in 0..100 {
        // forward pass
        let result = graph.forward(&add_node2.node).unwrap();
        println!("Epoch {}: Loss: {:?}", epoch, result);

        // backward pass
        graph.backward(&add_node2.node).unwrap();

        // update weights
        optimizer.step(&mut graph);
    }
}
