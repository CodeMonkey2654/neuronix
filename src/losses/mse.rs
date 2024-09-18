// src/losses/mse.rs

use std::rc::Rc;
use crate::graph::Node;
use crate::losses::loss::Loss;
use crate::graph::ops::SubtractOp;
use crate::graph::ops::SquareOp;
use crate::graph::ops::MeanOp;

pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self {
        Self
    }
}

impl Loss for MSELoss {
    fn compute(&self, prediction: Rc<Node>, target: Rc<Node>) -> Rc<Node> {
        let diff = Node::new(Box::new(SubtractOp), vec![prediction.into(), target.into()]);
        let sq_diff = Node::new(Box::new(SquareOp), vec![diff.into()]);
        Node::new(Box::new(MeanOp::new(None)), vec![sq_diff.into()]).into()
    }
}

// Assuming SubtractOp, SquareOp, and MeanOp are defined in your op.rs
