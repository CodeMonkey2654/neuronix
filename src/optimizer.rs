use crate::tensor::Tensor;

pub trait Optimizer {
    fn step(&mut self, params: &mut [Tensor<f32>], grads: &[Tensor<f32>]);
    fn zero_grad(&mut self);
}

pub struct SGDOptimizer {
    learning_rate: f32,
}

impl SGDOptimizer {
    pub fn new(learning_rate: f32) -> Self {
        SGDOptimizer { learning_rate }
    }
}

impl Optimizer for SGDOptimizer {
    fn step(&mut self, params: &mut [Tensor<f32>], grads: &[Tensor<f32>]) {
        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            *param = param.clone() - grad.clone() * self.learning_rate;
        }
    }

    fn zero_grad(&mut self) {
        // SGD doesn't maintain any state for gradients, so this method is empty
    }
}