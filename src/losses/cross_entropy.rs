// src/losses/cross_entropy.rs

use std::rc::Rc;
use crate::graph::Node;
use crate::losses::loss::Loss;
use crate::graph::ops::{LogOp, MultiplyOp, SumOp, NegateOp};

pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self
    }
}

impl Loss for CrossEntropyLoss {
    fn compute(&self, prediction: Rc<Node>, target: Rc<Node>) -> Rc<Node> {
        let log_pred = Node::new(Box::new(LogOp), vec![prediction.clone()]);
        let mul = Node::new(Box::new(MultiplyOp), vec![target.into(), log_pred.into()]);
        let sum = Node::new(Box::new(SumOp::new(None)), vec![mul.into()]);
        let neg = Node::new(Box::new(NegateOp), vec![sum.into()]);
        neg.into()
    }
}

// Assuming LogOp, MultiplyOp, SumOp, and NegateOp are defined in your op.rs
