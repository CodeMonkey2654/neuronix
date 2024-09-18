// src/optimizers/optimizer.rs

use std::rc::Rc;
use crate::graph::Node;

pub trait Optimizer {
    fn step(&mut self, parameters: &[Rc<Node>]);
    fn zero_grad(&mut self, parameters: &[Rc<Node>]);
}
