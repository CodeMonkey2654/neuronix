use crate::Node;
use std::rc::Rc;
use crate::Linear;

pub trait Layer {
    fn forward(&self, input: Rc<Node>) -> Rc<Node>;
}

impl Layer for Linear {
    fn forward(&self, input: Rc<Node>) -> Rc<Node> {
        self.forward(input)
    }
}