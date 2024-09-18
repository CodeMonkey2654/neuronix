use crate::graph::{Node};
use crate::graph::ops::ReLUOp;
use crate::layers::layer::Layer;
use std::rc::Rc;

/// ReLU activation function
pub struct ReLU;

impl ReLU {
    /// Creates a new ReLU activation function
    pub fn new() -> Self {
        ReLU
    }
}

impl Layer for ReLU {
    fn forward(&self, input: Rc<Node>) -> Rc<Node> {
        let relu_op = Box::new(ReLUOp);
        Rc::new(Node::new(relu_op, vec![input]))
    }
}
