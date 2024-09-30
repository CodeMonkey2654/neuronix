use crate::layer::Layer;
use crate::tensor::Tensor;
use crate::optimizer::Optimizer;

pub struct NeuralNetwork {
    layers: Vec<Box<dyn Layer>>,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        NeuralNetwork { layers }
    }

    pub fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output);
        }
        output
    }

    pub fn backward(&mut self, input: &Tensor<f32>, grad_output: &Tensor<f32>) -> Tensor<f32> {
        let mut grad = grad_output.clone();
        let mut layer_inputs = vec![input.clone()];

        // Forward pass to store intermediate inputs
        for layer in &self.layers {
            let output = layer.forward(layer_inputs.last().unwrap());
            layer_inputs.push(output);
        }

        // Backward pass
        for (layer, layer_input) in self.layers.iter_mut().rev().zip(layer_inputs.iter().rev().skip(1)) {
            grad = layer.backward(layer_input, &grad);
        }

        grad
    }

    pub fn get_params_and_grads(&mut self) -> (Vec<&mut Tensor<f32>>, Vec<&Tensor<f32>>) {
        let mut params = Vec::new();
        let mut grads = Vec::new();

        for layer in &mut self.layers {
            params.extend(layer.get_params());
        }
        let layer_grads: Vec<_> = self.layers.iter().flat_map(|layer| layer.get_grads()).collect();
        grads.extend(layer_grads);

        (params, grads)
    }

    pub fn train(&mut self, input: &Tensor<f32>, target: &Tensor<f32>, optimizer: &mut dyn Optimizer, loss_fn: &dyn Fn(&Tensor<f32>, &Tensor<f32>) -> Tensor<f32>) -> f32 {
        // Forward pass
        let output = self.forward(input);

        // Compute loss
        let loss = loss_fn(&output, target);

        // Backward pass
        let grad_output = output - target.clone(); // Assuming MSE loss for simplicity
        self.backward(input, &grad_output);

        // Update parameters
        let (params, grads) = self.get_params_and_grads();
        optimizer.step(params.as_mut_slice(), grads.as_slice());

        loss.data()[0]
    }
}
