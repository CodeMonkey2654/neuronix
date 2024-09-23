use std::rc::Rc;
use std::cell::RefCell;
use crate::node::Node;
use crate::tensor::Tensor;
use crate::op::Op;
use crate::tensor::TensorError;
use std::any::Any;

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
        let node = Node::new(id, Rc::new(NoOp));
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

    pub fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug, Clone)]
struct NoOp;

impl Op for NoOp {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 1 {
            return Err(TensorError::InvalidInputSize("NoOp operation requires 1 input".to_string()));
        }
        Ok(inputs[0].clone())
    }

    fn backward(&self, _inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        Ok(vec![output_gradient.clone()])
    }

    fn name(&self) -> &str {
        "NoOp"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::op::Add;
    use std::rc::Rc;

    #[test]
    fn test_variable_new() {
        let op = Rc::new(NoOp);
        let var = Variable::new(1, op, true);
        assert_eq!(var.id(), 1);
        assert!(var.requires_grad);
        assert!(var.value().is_none());
    }

    #[test]
    fn test_variable_from_tensor() {
        let tensor = Tensor::scalar(5.0);
        let var = Variable::from_tensor(2, tensor.clone(), false);
        assert_eq!(var.id(), 2);
        assert!(!var.requires_grad);
        assert_eq!(var.value().unwrap(), tensor);
    }

    #[test]
    fn test_to_tensor() {
        let tensor = Tensor::scalar(3.0);
        let var = Variable::from_tensor(3, tensor.clone(), true);
        assert_eq!(var.to_tensor(), tensor);
    }

    #[test]
    fn test_set_value() {
        let var = Variable::new(4, Rc::new(NoOp), true);
        let tensor = Tensor::scalar(7.0);
        var.set_value(tensor.clone());
        assert_eq!(var.value().unwrap(), tensor);
    }

    #[test]
    fn test_value() {
        let tensor = Tensor::scalar(9.0);
        let var = Variable::from_tensor(5, tensor.clone(), false);
        assert_eq!(var.value().unwrap(), tensor);
    }

    #[test]
    fn test_gradient() {
        let var_with_grad = Variable::new(6, Rc::new(NoOp), true);
        let var_without_grad = Variable::new(7, Rc::new(NoOp), false);
        
        assert!(var_with_grad.gradient().is_none());
        assert!(var_without_grad.gradient().is_none());

        // Set a gradient for var_with_grad
        let grad_tensor = Tensor::scalar(2.0);
        var_with_grad.node.borrow_mut().set_gradient(grad_tensor.clone());

        assert_eq!(var_with_grad.gradient().unwrap(), grad_tensor);
        assert!(var_without_grad.gradient().is_none());
    }

    #[test]
    fn test_backward() {
        let var = Variable::new(8, Rc::new(Add), true);
        let input1 = Variable::from_tensor(9, Tensor::scalar(2.0), true);
        let input2 = Variable::from_tensor(10, Tensor::scalar(3.0), true);
        
        var.node.borrow_mut().add_input(input1.node.clone());
        var.node.borrow_mut().add_input(input2.node.clone());
        var.node.borrow_mut().forward().unwrap();

        let result = var.backward();
        assert!(result.is_ok());
        let gradients = result.unwrap();
        assert_eq!(gradients.len(), 2);
        assert_eq!(gradients[0], Tensor::scalar(1.0));
        assert_eq!(gradients[1], Tensor::scalar(1.0));
    }

    #[test]
    fn test_update() {
        let var = Variable::from_tensor(11, Tensor::scalar(10.0), true);
        let grad = Variable::from_tensor(12, Tensor::scalar(2.0), false);
        
        let result = var.update(&grad, 0.1);
        assert!(result.is_ok());
        assert_eq!(var.value().unwrap(), Tensor::scalar(9.8));

        // Test update on a variable that doesn't require gradients
        let var_no_grad = Variable::from_tensor(13, Tensor::scalar(5.0), false);
        let result = var_no_grad.update(&grad, 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_id() {
        let var = Variable::new(14, Rc::new(NoOp), true);
        assert_eq!(var.id(), 14);
    }

    #[test]
    fn test_as_any() {
        let var = Variable::new(15, Rc::new(NoOp), true);
        let any_var = var.as_any();
        assert!(any_var.is::<Variable>());
    }

    #[test]
    fn test_clone() {
        let var = Variable::from_tensor(16, Tensor::scalar(4.0), true);
        let cloned_var = var.clone();
        
        assert_eq!(var.id(), cloned_var.id());
        assert_eq!(var.requires_grad, cloned_var.requires_grad);
        assert_eq!(var.value().unwrap(), cloned_var.value().unwrap());
    }

    #[test]
    fn test_debug() {
        let var = Variable::from_tensor(17, Tensor::scalar(6.0), true);
        let debug_output = format!("{:?}", var);
        assert!(debug_output.contains("Variable"));
        assert!(debug_output.contains("node"));
        assert!(debug_output.contains("requires_grad: true"));
    }
}
