use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::cell::RefCell;
use crate::node::Node;
use crate::tensor::Tensor;
use crate::variable::Variable;
use crate::op::Op;

#[derive(Debug)]
pub struct ComputationGraph {
    nodes: HashMap<usize, Rc<RefCell<Node>>>,
    gradient_tape: Vec<usize>,
    evaluated: HashSet<usize>,
}

impl ComputationGraph {
    pub fn new() -> Self {
        ComputationGraph {
            nodes: HashMap::new(),
            gradient_tape: Vec::new(),
            evaluated: HashSet::new(),
        }
    }

    pub fn add_node(&mut self, op: Rc<dyn Op>, inputs: Vec<&Variable>) -> Variable {
        let id = self.nodes.len();
        let node = Node::new(id, op);
        
        let requires_grad = inputs.iter().any(|v| v.requires_grad);
        
        for input in &inputs {
            node.borrow_mut().add_input(Rc::clone(&input.node));
            input.node.borrow_mut().add_output(Rc::downgrade(&node));
        }

        let variable = Variable {
            node: Rc::clone(&node),
            requires_grad,
        };

        self.nodes.insert(id, node);
        variable
    }

    pub fn add_node_with_variables(&mut self, op: Rc<dyn Op>, inputs: Vec<&Variable>, variables: Vec<Variable>) -> Variable {
        let id = self.nodes.len();
        let node = Node::new(id, op);
        
        let requires_grad = inputs.iter().any(|v| v.requires_grad) || variables.iter().any(|v| v.requires_grad);
        
        for input in &inputs {
            node.borrow_mut().add_input(Rc::clone(&input.node));
            input.node.borrow_mut().add_output(Rc::downgrade(&node));
        }

        for variable in &variables {
            node.borrow_mut().add_input(Rc::clone(&variable.node));
            variable.node.borrow_mut().add_output(Rc::downgrade(&node));
        }

        let variable = Variable {
            node: Rc::clone(&node),
            requires_grad,
        };

        self.nodes.insert(id, node);
        variable
    }

    pub fn add_node_only(&mut self, op: Rc<dyn Op>) -> Rc<RefCell<Node>> {
        let id = self.nodes.len();
        let node = Node::new(id, op);
        self.nodes.insert(id, Rc::clone(&node));
        node
    }

    pub fn forward(&mut self, node: &Rc<RefCell<Node>>) -> Result<Tensor, String> {
        let id = node.borrow().id();
        
        if self.evaluated.contains(&id) {
            return node.borrow().value().cloned().ok_or_else(|| "Node value not found".to_string());
        }

        // Recursively compute inputs
        for input in node.borrow().inputs.iter() {
            self.forward(input)?;
        }

        // Collect input values after they've all been computed
        let input_values: Vec<Tensor> = node.borrow().inputs.iter()
            .map(|input| input.borrow().value().cloned().unwrap())
            .collect();

        let _input_refs: Vec<&Tensor> = input_values.iter().collect();
        let result = node.borrow_mut().forward().map_err(|e| e.to_string())?;
        node.borrow_mut().set_value(result.clone());
        self.evaluated.insert(id);
        self.gradient_tape.push(id);

        Ok(result)
    }

    pub fn backward(&mut self, loss_node: &Rc<RefCell<Node>>) -> Result<(), String> {
        self.gradient_tape.clear();
        self.evaluated.clear();
        
        // Forward pass
        self.forward(loss_node)?;

        // Set initial gradient
        loss_node.borrow_mut().set_gradient(Tensor::ones(&[1], "f32").map_err(|e| e.to_string())?);

        // Backward pass
        while let Some(id) = self.gradient_tape.pop() {
            let node = self.nodes.get(&id).unwrap();
            let gradients = node.borrow_mut().backward().map_err(|e| e.to_string())?;

            for (input, grad) in node.borrow().inputs.iter().zip(gradients) {
                let mut input_ref = input.borrow_mut();
                let existing_grad = input_ref.gradient().cloned();
                match existing_grad {
                    Some(eg) => input_ref.set_gradient(&eg + &grad),
                    None => input_ref.set_gradient(grad),
                }
            }
        }

        Ok(())
    }

    pub fn get_gradient(&self, node: &Rc<RefCell<Node>>) -> Option<Tensor> {
        node.borrow().gradient().cloned()
    }

    pub fn get_tensor(&self, node: &Rc<RefCell<Node>>) -> Option<Tensor> {
        node.borrow().value().cloned()
    }

    pub fn create_variable(&mut self, tensor: Tensor, requires_grad: bool) -> Variable {
        let id = self.nodes.len();
        let variable = Variable::from_tensor(id, tensor, requires_grad);
        self.nodes.insert(id, Rc::clone(&variable.node));
        variable
    }

    pub fn trainable_variables(&self) -> Vec<Variable> {
        self.nodes.values()
            .filter_map(|node| {
                node.borrow().as_variable().cloned().filter(|var| var.requires_grad)
            })
            .collect()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{Add, Multiply};
    use crate::tensor::Tensor;

    #[test]
    fn test_create_variable() {
        let mut graph = ComputationGraph::new();
        let tensor = Tensor::new(&[2, 2], vec![1.0, 2.0, 3.0, 4.0].as_slice(), "f32").unwrap();
        let variable = graph.create_variable(tensor.clone(), true);

        assert!(variable.requires_grad);
        assert_eq!(graph.get_tensor(&variable.node), Some(tensor));
    }

    #[test]
    fn test_add_node() {
        let mut graph = ComputationGraph::new();
        let a = graph.create_variable(Tensor::new(&[2], vec![1.0, 2.0].as_slice(), "f32").unwrap(), true);
        let b = graph.create_variable(Tensor::new(&[2], vec![3.0, 4.0].as_slice(), "f32").unwrap(), true);

        let add_op = Rc::new(Add);
        let c = graph.add_node(add_op, vec![&a, &b]);

        assert!(c.requires_grad);
        assert_eq!(graph.nodes.len(), 3);
    }

    #[test]
    fn test_forward() {
        let mut graph = ComputationGraph::new();
        let a = graph.create_variable(Tensor::new(&[2], vec![1.0, 2.0].as_slice(), "f32").unwrap(), true);
        let b = graph.create_variable(Tensor::new(&[2], vec![3.0, 4.0].as_slice(), "f32").unwrap(), true);

        let add_op = Rc::new(Add);
        let c = graph.add_node(add_op, vec![&a, &b]);

        let result = graph.forward(&c.node).unwrap();
        assert_eq!(result, Tensor::new(&[2], vec![4.0, 6.0].as_slice(), "f32").unwrap());
    }

    #[test]
    fn test_backward() {
        let mut graph = ComputationGraph::new();
        let a = graph.create_variable(Tensor::new(&[1], vec![2.0].as_slice(), "f32").unwrap(), true);
        let b = graph.create_variable(Tensor::new(&[1], vec![3.0].as_slice(), "f32").unwrap(), true);

        let mul_op = Rc::new(Multiply);
        let c = graph.add_node(mul_op, vec![&a, &b]);

        graph.backward(&c.node).unwrap();

        assert_eq!(graph.get_gradient(&a.node), Some(Tensor::new(&[1], vec![3.0].as_slice(), "f32").unwrap()));
        assert_eq!(graph.get_gradient(&b.node), Some(Tensor::new(&[1], vec![2.0].as_slice(), "f32").unwrap()));
    }

    #[test]
    fn test_trainable_variables() {
        let mut graph = ComputationGraph::new();
        let a = graph.create_variable(Tensor::new(&[1], vec![1.0].as_slice(), "f32").unwrap(), true);
        let b = graph.create_variable(Tensor::new(&[1], vec![2.0].as_slice(), "f32").unwrap(), false);
        let c = graph.create_variable(Tensor::new(&[1], vec![3.0].as_slice(), "f32").unwrap(), true);

        let trainable = graph.trainable_variables();
        assert_eq!(trainable.len(), 2);
        assert!(trainable.iter().any(|v| v.node.borrow().id() == a.node.borrow().id()));
        assert!(trainable.iter().any(|v| v.node.borrow().id() == c.node.borrow().id()));
    }

    #[test]
    fn test_complex_graph() {
        let mut graph = ComputationGraph::new();
        let a = graph.create_variable(Tensor::new(&[1], vec![2.0].as_slice(), "f32").unwrap(), true);
        let b = graph.create_variable(Tensor::new(&[1], vec![3.0].as_slice(), "f32").unwrap(), true);
        let c = graph.create_variable(Tensor::new(&[1], vec![4.0].as_slice(), "f32").unwrap(), true);

        let mul_op = Rc::new(Multiply);
        let add_op = Rc::new(Add);

        let d = graph.add_node(mul_op.clone(), vec![&a, &b]);
        let e = graph.add_node(add_op, vec![&d, &c]);

        graph.forward(&e.node).unwrap();
        graph.backward(&e.node).unwrap();

        assert_eq!(graph.get_tensor(&e.node), Some(Tensor::new(&[1], vec![10.0].as_slice(), "f32").unwrap()));
        assert_eq!(graph.get_gradient(&a.node), Some(Tensor::new(&[1], vec![3.0].as_slice(), "f32").unwrap()));
        assert_eq!(graph.get_gradient(&b.node), Some(Tensor::new(&[1], vec![2.0].as_slice(), "f32").unwrap()));
        assert_eq!(graph.get_gradient(&c.node), Some(Tensor::new(&[1], vec![1.0].as_slice(), "f32").unwrap()));
    }
}