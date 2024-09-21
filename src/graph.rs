use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use crate::node::Node;
use crate::op::Op;
use crate::tensor::Tensor;
use crate::variable::Variable;
use crate::optimizer::Optimizer;
use crate::op::Identity;

pub struct ComputationGraph {
    nodes: HashMap<usize, Rc<RefCell<Node>>>,
    next_id: usize,
    gradient_tape: Vec<usize>,
}

impl ComputationGraph {
    pub fn new() -> Self {
        ComputationGraph {
            nodes: HashMap::new(),
            next_id: 0,
            gradient_tape: Vec::new(),
        }
    }

    pub fn add_node(&mut self, op: Rc<dyn Op>, inputs: Vec<Rc<RefCell<Node>>>) -> Rc<RefCell<Node>> {
        let id = self.next_id;
        self.next_id += 1;

        let node = Node::new(id, op);
        
        {
            let mut node_mut = node.borrow_mut();
            for input in inputs.iter() {
                node_mut.add_input(Rc::clone(input));
                input.borrow_mut().add_output(Rc::downgrade(&node));
            }
        }

        self.nodes.insert(id, Rc::clone(&node));
        node
    }

    pub fn add_variable(&mut self, tensor: Tensor, requires_grad: bool) -> Variable {
        let id = self.next_id;
        self.next_id += 1;

        let node = Node::new(id, Rc::new(Identity));
        node.borrow_mut().set_value(tensor);
        Variable {
            node,
            requires_grad,
        }
    }

    pub fn forward(&mut self, input_values: Tensor) -> Result<Tensor, String> {
        self.gradient_tape.clear();
        let mut computed_values: HashMap<usize, Tensor> = HashMap::new();
        computed_values.insert(0, input_values); // Assuming the input node has id 0

        let sorted_nodes = self.topological_sort();

        for &node_id in &sorted_nodes {
            let node = self.nodes.get(&node_id).unwrap();
            let mut node_mut = node.borrow_mut();

            let input_tensors: Vec<Tensor> = node_mut.inputs
                .iter()
                .map(|input| computed_values[&input.borrow().id()].clone())
                .collect();

            let output = node_mut.op().forward(&input_tensors).map_err(|e| e.to_string())?;
            node_mut.set_value(output.clone());
            computed_values.insert(node_id, output);

            self.gradient_tape.push(node_id);
        }

        Ok(computed_values[sorted_nodes.last().unwrap()].clone())
    }

    pub fn backward_and_optimize(&mut self, loss_node_id: usize, optimizer: &mut dyn Optimizer) -> Result<(), String> {
        let mut gradients: HashMap<usize, Tensor> = HashMap::new();
        
        let loss_node = self.nodes.get(&loss_node_id).unwrap();
        let loss_shape = loss_node.borrow().value().unwrap().shape().to_vec();
        gradients.insert(loss_node_id, Tensor::new_f32(ndarray::Array::ones(loss_shape).into_dyn()));

        for &node_id in self.gradient_tape.iter().rev() {
            let node = self.nodes.get(&node_id).unwrap();
            let mut node_mut = node.borrow_mut();

            if let Some(grad) = gradients.get(&node_id) {
                node_mut.set_gradient(grad.clone());

                let input_gradients = node_mut.backward().map_err(|e| e.to_string())?;
                
                for (input_node, input_grad) in node_mut.inputs.iter().zip(input_gradients.iter()) {
                    let input_id = input_node.borrow().id();
                    gradients.clone()
                        .entry(input_id)
                        .and_modify(|existing_grad| {
                            *existing_grad = existing_grad.clone() + input_grad.clone();
                        })
                        .or_insert_with(|| input_grad.clone());
                }

                let param = Variable::new(node_id, Rc::new(Identity), true);
                param.set_value(node_mut.value().unwrap().clone());
                let updated_param = optimizer.update(param, grad.clone());
                node_mut.set_value(updated_param);
            }
        }

        Ok(())
    }

    fn topological_sort(&self) -> Vec<usize> {
        let mut sorted = Vec::new();
        let mut visited = HashMap::new();

        for &node_id in self.nodes.keys() {
            self.dfs(node_id, &mut visited, &mut sorted);
        }

        sorted
    }

    fn dfs(&self, node_id: usize, visited: &mut HashMap<usize, bool>, sorted: &mut Vec<usize>) {
        if visited.get(&node_id) == Some(&true) {
            return;
        }

        visited.insert(node_id, true);

        let node = self.nodes.get(&node_id).unwrap();
        for input in node.borrow().inputs.iter() {
            self.dfs(input.borrow().id(), visited, sorted);
        }

        sorted.push(node_id);
    }

    fn add_loss_node(&mut self, loss_op: Rc<dyn Op>, inputs: Vec<Rc<RefCell<Node>>>) -> Rc<RefCell<Node>> {
        let id = self.next_id;
        self.next_id += 1;

        let node = Node::new(id, loss_op);
        for input in inputs {
            node.borrow_mut().inputs.push(Rc::clone(&input));
        }
        self.nodes.insert(id, node.clone());
        node.clone()
    }
}
