use std::rc::Rc;
use crate::graph::Node;
use crate::optimizers::optimizer::Optimizer;
use ndarray::ArrayD;

pub struct SGD {
    pub learning_rate: f64,
    pub momentum: f64,
    velocities: Vec<ArrayD<f64>>,
}

impl SGD {
    pub fn new(learning_rate: f64, momentum: f64) -> Self {
        Self {
            learning_rate,
            momentum,
            velocities: Vec::new(),
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, parameters: &[Rc<Node>]) {
        if self.velocities.is_empty() {
            self.velocities = parameters.iter().map(|p| p.value.borrow().as_ref().unwrap().clone()).collect();
        }

        for (i, param) in parameters.iter().enumerate() {
            let grad = param.grad;
            self.velocities[i] = self.momentum * self.velocities[i] - self.learning_rate * grad.borrow().as_ref().unwrap().clone();
            param.set_value(param.value.borrow().as_ref().unwrap().clone() + &self.velocities[i]);
        }
    }

    fn zero_grad(&mut self, parameters: &[Rc<Node>]) {
        for param in parameters {
            param.zero_grad();
        }
    }
}
