use crate::graph::{Node};
use crate::graph::ops::TanhOp;
use crate::layers::layer::Layer;
use std::rc::Rc;

/// Tanh activation function
pub struct Tanh;

impl Tanh {
    /// Creates a new Tanh activation function
    pub fn new() -> Self {
        Tanh
    }
}

impl Layer for Tanh {
    fn forward(&self, input: Rc<Node>) -> Rc<Node> {
        let tanh_op = Box::new(TanhOp);
        Rc::new(Node::new(tanh_op, vec![input]))
    }
}
