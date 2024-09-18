use std::rc::Rc;
use crate::graph::{Node};
use crate::layers::layer::Layer;
use ndarray::ArrayD;
use crate::graph::ops::{AbsOp, SumOp, ScalarMulOp, AddOp};
use ndarray::IxDyn;

pub struct L1Regularizer {
    pub lambda: f64, // Regularization strength
}

impl L1Regularizer {
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }

    fn backward(&self, parameter: &Rc<Node>) -> ArrayD<f64> {
        parameter.into().value.mapv(|x| self.lambda * x.signum())
    }
}

impl Layer for L1Regularizer {
    fn forward(&self, parameters: [Rc<Node>]) -> Rc<Node> {
        let mut reg_loss = Node::new_leaf(ArrayD::<f64>::zeros(IxDyn(&[1, 1])).into_dyn());
        for param in parameters {
            let abs_param = Node::new(Box::new(AbsOp), vec![param.clone()]);
            let sum = Node::new(Box::new(SumOp::new(None)), vec![abs_param.into()]);
            let scaled = Node::new(Box::new(ScalarMulOp::new(self.lambda)), vec![sum.into()]);
            reg_loss = Node::new(Box::new(AddOp), vec![reg_loss.into(), scaled.into()]);
        }
        reg_loss.into()
    }
}
