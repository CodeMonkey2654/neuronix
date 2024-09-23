use std::rc::Rc;
use std::cell::RefCell;
use crate::tensor::Tensor;
use std::rc::Weak;
use crate::op::Op;
use crate::tensor::TensorError;
use crate::variable::Variable;

#[derive(Debug, Clone)]
pub struct Node {
    id: usize,
    op: Rc<dyn Op>,
    pub inputs: Vec<Rc<RefCell<Node>>>,
    outputs: Vec<Weak<RefCell<Node>>>,
    value: Option<Tensor>,
    gradient: Option<Tensor>,
}

impl Node {
    pub fn new(id: usize, op: Rc<dyn Op>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            id,
            op,
            inputs: Vec::new(),
            outputs: Vec::new(),
            value: None,
            gradient: None,
        }))
    }

    pub fn add_input(&mut self, input: Rc<RefCell<Node>>) {
        self.inputs.push(input);
    }

    pub fn add_output(&mut self, output: Weak<RefCell<Node>>) {
        self.outputs.push(output);
    }

    pub fn set_value(&mut self, value: Tensor) {
        self.value = Some(value);
    }

    pub fn set_gradient(&mut self, gradient: Tensor) {
        self.gradient = Some(gradient);
    }

    pub fn forward(&mut self) -> Result<Tensor, TensorError> {
        let input_values: Vec<Tensor> = self.inputs
            .iter()
            .map(|input| input.borrow().value.as_ref().unwrap().clone())
            .collect();
        let input_refs: Vec<&Tensor> = input_values.iter().collect();
        self.op.forward(&input_refs)
    }
    
    pub fn backward(&mut self) -> Result<Vec<Tensor>, TensorError> {
        let input_values: Vec<Tensor> = self.inputs
            .iter()
            .map(|input| input.borrow().value.as_ref().unwrap().clone())
            .collect();
        let input_refs: Vec<&Tensor> = input_values.iter().collect();
        let output_gradient = self.gradient.as_ref().unwrap();
        self.op.backward(&input_refs, output_gradient)
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn value(&self) -> Option<&Tensor> {
        self.value.as_ref()
    }

    pub fn gradient(&self) -> Option<&Tensor> {
        self.gradient.as_ref()
    }

    pub fn op(&self) -> &dyn Op {
        self.op.as_ref()
    }

    pub fn as_variable(&self) -> Option<&Variable> {
        if let Some(var) = self.op.as_any().downcast_ref::<Variable>() {
            Some(var)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::Any;
    use crate::tensor::Tensor;

    // Mock Op implementation for testing
    #[derive(Debug, Clone)]
    struct MockOp;

    impl Op for MockOp {
        fn forward(&self, _inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
            Ok(Tensor::new(&[3], &[1.0, 2.0, 3.0], "f32").unwrap())
        }

        fn backward(&self, _inputs: &[&Tensor], _output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
            Ok(vec![Tensor::new(&[3], &[1.0, 1.0, 1.0], "f32").unwrap()])
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn name(&self) -> &str {
            "MockOp"
        }

        fn box_clone(&self) -> Box<dyn Op> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn test_node_creation() {
        let op = Rc::new(MockOp);
        let node = Node::new(1, op.clone());
        let node_ref = node.borrow();

        assert_eq!(node_ref.id(), 1);
        assert!(node_ref.inputs.is_empty());
        assert!(node_ref.outputs.is_empty());
        assert!(node_ref.value().is_none());
        assert!(node_ref.gradient().is_none());
    }

    #[test]
    fn test_add_input() {
        let op1 = Rc::new(MockOp);
        let op2 = Rc::new(MockOp);
        let node1 = Node::new(1, op1);
        let node2 = Node::new(2, op2);

        node2.borrow_mut().add_input(node1.clone());

        assert_eq!(node2.borrow().inputs.len(), 1);
        assert_eq!(node2.borrow().inputs[0].borrow().id(), 1);
    }

    #[test]
    fn test_add_output() {
        let op1 = Rc::new(MockOp);
        let op2 = Rc::new(MockOp);
        let node1 = Node::new(1, op1);
        let node2 = Node::new(2, op2);

        node1.borrow_mut().add_output(Rc::downgrade(&node2));

        assert_eq!(node1.borrow().outputs.len(), 1);
        assert_eq!(node1.borrow().outputs[0].upgrade().unwrap().borrow().id(), 2);
    }

    #[test]
    fn test_set_value() {
        let op = Rc::new(MockOp);
        let node = Node::new(1, op);
        let tensor = Tensor::new(&[3], &[4.0, 5.0, 6.0], "f32").unwrap();

        node.borrow_mut().set_value(tensor.clone());

        assert_eq!(node.borrow().value().unwrap(), &tensor);
    }

    #[test]
    fn test_set_gradient() {
        let op = Rc::new(MockOp);
        let node = Node::new(1, op);
        let tensor = Tensor::new(&[3], &[0.1, 0.2, 0.3], "f32").unwrap();

        node.borrow_mut().set_gradient(tensor.clone());

        assert_eq!(node.borrow().gradient().unwrap(), &tensor);
    }

    #[test]
    fn test_forward() {
        let op = Rc::new(MockOp);
        let node = Node::new(1, op);
        let input_node = Node::new(2, Rc::new(MockOp));
        input_node.borrow_mut().set_value(Tensor::new(&[3], &[1.0, 2.0, 3.0], "f32").unwrap());

        node.borrow_mut().add_input(input_node);

        let result = node.borrow_mut().forward().unwrap();

        assert_eq!(result, Tensor::new(&[3], &[1.0, 2.0, 3.0], "f32").unwrap());
    }

    #[test]
    fn test_backward() {
        let op = Rc::new(MockOp);
        let node = Node::new(1, op);
        let input_node = Node::new(2, Rc::new(MockOp));
        input_node.borrow_mut().set_value(Tensor::new(&[3], &[1.0, 2.0, 3.0], "f32").unwrap());

        node.borrow_mut().add_input(input_node);
        node.borrow_mut().set_gradient(Tensor::new(&[3], &[1.0, 1.0, 1.0], "f32").unwrap());

        let result = node.borrow_mut().backward().unwrap();

        assert_eq!(result, vec![Tensor::new(&[3], &[1.0, 1.0, 1.0], "f32").unwrap()]);
    }

    #[test]
    fn test_op() {
        let op = Rc::new(MockOp);
        let node = Node::new(1, op.clone());

        assert!(node.borrow().op().as_any().is::<MockOp>());
    }

    #[test]
    fn test_clone() {
        let op = Rc::new(MockOp);
        let node = Node::new(1, op);
        node.borrow_mut().set_value(Tensor::new(&[3], &[1.0, 2.0, 3.0], "f32").unwrap());
        node.borrow_mut().set_gradient(Tensor::new(&[3], &[0.1, 0.2, 0.3], "f32").unwrap());

        let cloned_node = node.borrow().clone();

        assert_eq!(cloned_node.id, node.borrow().id);
        assert_eq!(cloned_node.value, node.borrow().value);
        assert_eq!(cloned_node.gradient, node.borrow().gradient);
        assert!(cloned_node.inputs.is_empty());
        assert!(cloned_node.outputs.is_empty());
    }
}