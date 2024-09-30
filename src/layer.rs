use crate::tensor::Tensor;

pub trait Layer {
    fn forward(&self, input: &Tensor<f32>) -> Tensor<f32>;
    fn backward(&mut self, input: &Tensor<f32>, grad_output: &Tensor<f32>) -> Tensor<f32>;
    fn get_params(&mut self) -> Vec<&mut Tensor<f32>>;
    fn get_grads(&self) -> Vec<&Tensor<f32>>;
}


pub struct DenseLayer {
    weights: Tensor<f32>,
    biases: Tensor<f32>,
    grad_weights: Tensor<f32>,
    grad_biases: Tensor<f32>,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        // Initialize weights and biases with random values
        let weights = Tensor::rand(&[input_size, output_size], -0.1, 0.1);
        let biases = Tensor::zeros(&[output_size]);
        let grad_weights = Tensor::zeros(&[input_size, output_size]);
        let grad_biases = Tensor::zeros(&[output_size]);
        
        DenseLayer { weights, biases, grad_weights, grad_biases }
    }
}

impl Layer for DenseLayer {
    fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        // Perform the affine transformation: input * weights + biases
        let z = input.matmul(&self.weights) + self.biases.clone();
        z
    }

    fn backward(&mut self, input: &Tensor<f32>, grad_output: &Tensor<f32>) -> Tensor<f32> {
        // Compute gradients
        self.grad_weights = input.transpose().matmul(grad_output);
        self.grad_biases = grad_output.sum_axis(0);
        
        // Compute gradient with respect to input
        let grad_input = grad_output.matmul(&self.weights.transpose());
        grad_input
    }

    fn get_params(&mut self) -> Vec<&mut Tensor<f32>> {
        vec![&mut self.weights, &mut self.biases]
    }

    fn get_grads(&self) -> Vec<&Tensor<f32>> {
        vec![&self.grad_weights, &self.grad_biases]
    }
}


// ACTIVATION LAYERS

pub struct ReLU {
    mask: Tensor<bool>,
}

impl ReLU {
    pub fn new() -> Self {
        ReLU { mask: Tensor::zeros_like(&Tensor::zeros(&[])) }
    }
}

impl Layer for ReLU {
    fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        self.mask = input.gt(&Tensor::zeros_like(input));
        input * self.mask.cast::<f32>()
    }

    fn backward(&mut self, input: &Tensor<f32>, grad_output: &Tensor<f32>) -> Tensor<f32> {
        grad_output * self.mask.cast::<f32>()
    }

    fn get_params(&mut self) -> Vec<&mut Tensor<f32>> {
        vec![] // ReLU has no learnable parameters
    }

    fn get_grads(&self) -> Vec<&Tensor<f32>> {
        vec![] // ReLU has no gradients to store
    }
}


pub struct Sigmoid {
    output: Tensor<f32>,
}

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid { output: Tensor::zeros_like(&Tensor::zeros(&[])) }
    }
}

impl Layer for Sigmoid {
    fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        self.output = 1.0 / (1.0 + (-input).exp());
        self.output.clone()
    }

    fn backward(&mut self, input: &Tensor<f32>, grad_output: &Tensor<f32>) -> Tensor<f32> {
        let grad_input = grad_output * (1.0 - self.output.clone()) * self.output.clone();
        grad_input
    }

    fn get_params(&mut self) -> Vec<&mut Tensor<f32>> {
        vec![] // Sigmoid has no learnable parameters
    }

    fn get_grads(&self) -> Vec<&Tensor<f32>> {
        vec![] // Sigmoid has no gradients to store
    }
}


pub struct Tanh {
    output: Tensor<f32>,
}

impl Tanh {
    pub fn new() -> Self {
        Tanh { output: Tensor::zeros_like(&Tensor::zeros(&[])) }
    }
}

impl Layer for Tanh {
    fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        self.output = (input.exp() - (-input).exp()) / (input.exp() + (-input).exp());
        self.output.clone()
    }

    fn backward(&mut self, input: &Tensor<f32>, grad_output: &Tensor<f32>) -> Tensor<f32> {
        let grad_input = grad_output * (1.0 - self.output.clone() * self.output.clone());
        grad_input
    }

    fn get_params(&mut self) -> Vec<&mut Tensor<f32>> {
        vec![] // Tanh has no learnable parameters
    }

    fn get_grads(&self) -> Vec<&Tensor<f32>> {
        vec![] // Tanh has no gradients to store
    }
}


pub struct Softmax {
    output: Tensor<f32>,
    logits: Tensor<f32>,
}

impl Softmax {
    pub fn new() -> Self {
        Softmax { output: Tensor::zeros_like(&Tensor::zeros(&[])), logits: Tensor::zeros_like(&Tensor::zeros(&[])) }
    }
}

impl Layer for Softmax {
    fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        self.logits = input.clone();
        let exp = (-input).exp();
        self.output = exp / exp.sum_axis(1).expand_dims(1);
        self.output.clone()
    }

    fn backward(&mut self, input: &Tensor<f32>, grad_output: &Tensor<f32>) -> Tensor<f32> {
        let grad_input = self.output.clone() * (grad_output - (self.output.clone() * grad_output).sum_axis(1).expand_dims(1));
        grad_input
    }

    fn get_params(&mut self) -> Vec<&mut Tensor<f32>> {
        vec![] // Softmax has no learnable parameters
    }

    fn get_grads(&self) -> Vec<&Tensor<f32>> {
        vec![] // Softmax has no gradients to store
    }
}


// LOSS LAYERS

pub struct MSELoss {
    output: Tensor<f32>,
    target: Tensor<f32>,
}

impl MSELoss {
    pub fn new() -> Self {
        MSELoss { output: Tensor::zeros_like(&Tensor::zeros(&[])), target: Tensor::zeros_like(&Tensor::zeros(&[])) }
    }
}

impl Layer for MSELoss {
    fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        self.output = (input - self.target).powi(2).mean();
        self.output.clone()
    }

    fn backward(&mut self, input: &Tensor<f32>, grad_output: &Tensor<f32>) -> Tensor<f32> {
        let grad_input = 2.0 * (input - self.target);
        grad_input
    }

    fn get_params(&mut self) -> Vec<&mut Tensor<f32>> {
        vec![] // MSELoss has no learnable parameters
    }

    fn get_grads(&self) -> Vec<&Tensor<f32>> {
        vec![] // MSELoss has no gradients to store
    }
}

pub struct CrossEntropyLoss {
    output: Tensor<f32>,
    target: Tensor<f32>,
}

impl CrossEntropyLoss {
    pub fn new() -> Self {
        CrossEntropyLoss { output: Tensor::zeros_like(&Tensor::zeros(&[])), target: Tensor::zeros_like(&Tensor::zeros(&[])) }
    }
}

impl Layer for CrossEntropyLoss {
    fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        self.output = -self.target * (input - input.log_softmax()).sum_axis(1);
        self.output.mean()
    }

    fn backward(&mut self, input: &Tensor<f32>, grad_output: &Tensor<f32>) -> Tensor<f32> {
        let grad_input = input - self.target;
        grad_input
    }

    fn get_params(&mut self) -> Vec<&mut Tensor<f32>> {
        vec![] // CrossEntropyLoss has no learnable parameters
    }

    fn get_grads(&self) -> Vec<&Tensor<f32>> {
        vec![] // CrossEntropyLoss has no gradients to store
    }
}


