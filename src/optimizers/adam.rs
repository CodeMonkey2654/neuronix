use crate::graph::Node;
use crate::optimizers::optimizer::Optimizer;
use ndarray::{ArrayD};
use std::rc::Rc;

/// Adam optimizer.
pub struct Adam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    m: Vec<ArrayD<f64>>,
    v: Vec<ArrayD<f64>>,
    t: usize,
}

impl Adam {
    /// Creates a new Adam optimizer with the given hyperparameters.
    pub fn new(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, parameters: &[Rc<Node>]) {
        if self.m.is_empty() {
            // Initialize first and second moment vectors
            for param in parameters {
                let shape = param.value.borrow().as_ref().unwrap().shape().to_vec();
                self.m.push(ArrayD::<f64>::zeros(shape.clone()));
                self.v.push(ArrayD::<f64>::zeros(shape));
            }
        }

        self.t += 1;

        for (i, param) in parameters.iter().enumerate() {
            let grad = param.grad.borrow().as_ref().unwrap();
            let mut m = &mut self.m[i];
            let mut v = &mut self.v[i];

            // Update biased first moment estimate
            *m = self.beta1 * &*m + (1.0 - self.beta1) * grad;

            // Update biased second raw moment estimate
            *v = self.beta2 * &*v + (1.0 - self.beta2) * grad.mapv(|g| g * g);

            // Compute bias-corrected first and second moment estimates
            let m_hat = &*m / (1.0 - self.beta1.powi(self.t as i32));
            let v_hat = &*v / (1.0 - self.beta2.powi(self.t as i32));

            // Update parameter
            let updated_value = param.value.borrow().as_ref().unwrap()
                - self.learning_rate * &m_hat / (v_hat.mapv(|v| v.sqrt()) + self.epsilon);
            param.set_value(updated_value);
        }
    }

    fn zero_grad(&mut self, parameters: &[Rc<Node>]) {
        for param in parameters {
            param.zero_grad();
        }
    }
}
