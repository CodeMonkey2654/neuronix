use crate::graph::{Node};
use crate::graph::ops::SigmoidOp;
use crate::layers::layer::Layer;
use std::rc::Rc;

/// Sigmoid activation function
pub struct Sigmoid;

impl Sigmoid {
    /// Creates a new Sigmoid activation function
    pub fn new() -> Self {
        Sigmoid
    }
}

impl Layer for Sigmoid {
    fn forward(&self, input: Rc<Node>) -> Rc<Node> {
        let sigmoid_op = Box::new(SigmoidOp);
        Rc::new(Node::new(sigmoid_op, vec![input]))
    }
}
