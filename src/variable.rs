use std::rc::Rc;
use std::cell::RefCell;
use crate::node::Node;
use crate::tensor::Tensor;
use crate::op::Op;
use crate::op::Identity;
use crate::tensor::TensorError;

#[derive(Debug, Clone)] 
pub struct Variable {
    pub node: Rc<RefCell<Node>>,
    pub requires_grad: bool,
}

impl Variable {
    pub fn new(id: usize, op: Rc<dyn Op>, requires_grad: bool) -> Self {
        Self {
            node: Node::new(id, op),
            requires_grad,
        }
    }

    pub fn from_tensor(id: usize, tensor: Tensor, requires_grad: bool) -> Self {
        let node = Node::new(id, Rc::new(Identity));
        node.borrow_mut().set_value(tensor);
        Self {
            node,
            requires_grad,
        }
    }

    pub fn to_tensor(&self) -> Tensor {
        self.node.borrow().value().unwrap().clone()
    }

    pub fn set_value(&self, value: Tensor) {
        self.node.borrow_mut().set_value(value);
    }

    pub fn value(&self) -> Option<Tensor> {
        self.node.borrow().value().cloned()
    }

    pub fn gradient(&self) -> Option<Tensor> {
        if self.requires_grad {
            self.node.borrow().gradient().cloned()
        } else {
            None
        }
    }

    pub fn backward(&self) -> Result<Vec<Tensor>, TensorError> {
        self.node.borrow_mut().backward()
    }

    pub fn update(&self, grad: &Variable, learning_rate: f32) -> Result<(), TensorError> {
        if self.requires_grad {
            let grad_tensor = grad.value().unwrap();
            let value_tensor = self.value().unwrap();
            let updated_value = &value_tensor - &(&Tensor::scalar(learning_rate) * &grad_tensor);
            self.set_value(updated_value);
            Ok(())
        } else {
            Err(TensorError::InvalidInputSize("Variable does not require gradient".to_string()))
        }
    }

    pub fn id(&self) -> usize {
        self.node.borrow().id()
    }
}
