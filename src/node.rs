use std::rc::Rc;
use std::cell::RefCell;
use crate::tensor::Tensor;
use std::rc::Weak;
use crate::op::Op;
use crate::tensor::TensorError;

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
        self.op.forward(&input_values)
    }
    pub fn backward(&mut self) -> Result<Vec<Tensor>, TensorError> {
        let input_values: Vec<&Tensor> = self.inputs
            .iter()
            .map(|input| input.borrow().value.as_ref().unwrap())
            .collect();
        let output_gradients: Vec<&Tensor> = self.outputs
            .iter()
            .filter_map(|output| output.upgrade())
            .map(|output| output.borrow().gradient.as_ref().unwrap())
            .collect();
        self.op.backward(&input_values, output_gradients.first().unwrap())
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
}