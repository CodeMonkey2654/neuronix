use crate::tensor::Tensor;
use crate::variable::Variable;
use std::collections::HashMap;

pub trait Optimizer {
    fn step(&mut self, variables: &[Variable]) -> Result<(), String>;
    fn zero_grad(&mut self);
    fn update(&mut self, param: Variable, grad: Tensor) -> Tensor;
}

pub struct SGD {
    learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        SGD { learning_rate }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, variables: &[Variable]) -> Result<(), String> {
        for var in variables {
            if let Some(grad) = var.gradient() {
                let updated_value = self.update(var.clone(), grad);
                var.set_value(updated_value);
            }
        }
        Ok(())
    }

    fn zero_grad(&mut self) {
        // SGD doesn't need to store gradients, so this is a no-op
    }

    fn update(&mut self, param: Variable, grad: Tensor) -> Tensor {
        let value = param.value().unwrap();
        value - (grad * Tensor::scalar(self.learning_rate))
    }
}

pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: usize,
    m: HashMap<usize, Tensor>,
    v: HashMap<usize, Tensor>,
}

impl Adam {
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, variables: &[Variable]) -> Result<(), String> {
        self.t += 1;
        for var in variables {
            if let Some(grad) = var.gradient() {
                let updated_value = self.update(var.clone(), grad);
                var.set_value(updated_value);
            }
        }
        Ok(())
    }

    fn zero_grad(&mut self) {
        self.m.clear();
        self.v.clear();
        self.t = 0;
    }

    fn update(&mut self, param: Variable, grad: Tensor) -> Tensor {
        let var_id = param.id();
        let value = param.value().unwrap();
        let m = self.m.entry(var_id).or_insert_with(|| Tensor::zeros(grad.shape()).unwrap());
        let v = self.v.entry(var_id).or_insert_with(|| Tensor::zeros(grad.shape()).unwrap());

        *m = (grad.clone() * Tensor::scalar(1.0 - self.beta1)) + (m.clone() * Tensor::scalar(self.beta1));
        *v = (grad.clone() * grad.clone() * Tensor::scalar(1.0 - self.beta2)) + (v.clone() * Tensor::scalar(self.beta2));

        let m_hat = m.clone() / Tensor::scalar(1.0 - self.beta1.powi(self.t as i32));
        let v_hat = v.clone() / Tensor::scalar(1.0 - self.beta2.powi(self.t as i32));

        value - (m_hat / (v_hat.sqrt() + Tensor::scalar(self.epsilon))) * Tensor::scalar(self.learning_rate)
    }
}

pub struct RMSprop {
    learning_rate: f32,
    alpha: f32,
    epsilon: f32,
    v: HashMap<usize, Tensor>,
}

impl RMSprop {
    pub fn new(learning_rate: f32, alpha: f32, epsilon: f32) -> Self {
        RMSprop {
            learning_rate,
            alpha,
            epsilon,
            v: HashMap::new(),
        }
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self, variables: &[Variable]) -> Result<(), String> {
        for var in variables {
            if let Some(grad) = var.gradient() {
                let updated_value = self.update(var.clone(), grad);
                var.set_value(updated_value);
            }
        }
        Ok(())
    }

    fn zero_grad(&mut self) {
        self.v.clear();
    }

    fn update(&mut self, param: Variable, grad: Tensor) -> Tensor {
        let var_id = param.id();
        let value = param.value().unwrap();
        let v = self.v.entry(var_id).or_insert_with(|| Tensor::zeros(grad.shape()).unwrap());

        *v = ((grad.clone() * grad.clone()) * Tensor::scalar(1.0 - self.alpha)) + (v.clone() * Tensor::scalar(self.alpha));

        value - (grad.clone() / (v.clone().sqrt() + Tensor::scalar(self.epsilon))) * Tensor::scalar(self.learning_rate)
    }
}

pub struct Momentum {
    learning_rate: f32,
    momentum: f32,
    velocity: HashMap<usize, Tensor>,
}

impl Momentum {
    pub fn new(learning_rate: f32, momentum: f32) -> Self {
        Momentum {
            learning_rate,
            momentum,
            velocity: HashMap::new(),
        }
    }
}

impl Optimizer for Momentum {
    fn step(&mut self, variables: &[Variable]) -> Result<(), String> {
        for var in variables {
            if let Some(grad) = var.gradient() {
                let updated_value = self.update(var.clone(), grad);
                var.set_value(updated_value);
            }
        }
        Ok(())
    }

    fn zero_grad(&mut self) {
        self.velocity.clear();
    }

    fn update(&mut self, param: Variable, grad: Tensor) -> Tensor {
        let var_id = param.id();
        let value = param.value().unwrap();
        let v = self.velocity.entry(var_id).or_insert_with(|| Tensor::zeros(value.shape()).unwrap());

        *v = (grad * Tensor::scalar(self.learning_rate)) + (v.clone() * Tensor::scalar(self.momentum));

        value - v.clone()
    }
}
