use std::rc::Rc;
use crate::graph::{Node};
use crate::layers::layer::Layer;
use ndarray::ArrayD;
use crate::graph::ops::{SquareOp, SumOp, ScalarMulOp, AddOp};
use ndarray::prelude::*;

pub struct L2Regularizer {
    pub lambda: f64, // Regularization strength
}

impl L2Regularizer {
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }
    fn backward(&self, parameter: Rc<Node>) -> ArrayD<f64> {
        parameter.into().value.mapv(|x| self.lambda * x)
    }
}

impl Layer for L2Regularizer {
    fn forward(&self, parameters: [Rc<Node>]) -> Rc<Node> {
        let mut reg_loss = Node::new_leaf(ArrayD::<f64>::zeros(IxDyn(&[1, 1])).into_dyn());
        for param in parameters {
            let square = Node::new(Box::new(SquareOp), vec![param.clone()]);
            let sum = Node::new(Box::new(SumOp::new(None)), vec![square.into()]);
            let scaled = Node::new(Box::new(ScalarMulOp::new(self.lambda / 2.0)), vec![sum.into()]);
            reg_loss = Node::new(Box::new(AddOp), vec![reg_loss.into(), scaled.into()]);
        }
        reg_loss.into()
    }
}