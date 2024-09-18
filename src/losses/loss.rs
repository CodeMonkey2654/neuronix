use std::rc::Rc;
use crate::graph::Node;

pub trait Loss {
    fn compute(&self, prediction: Rc<Node>, target: Rc<Node>) -> Rc<Node>;
}
