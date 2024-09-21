use crate::tensor::{Tensor, TensorError};
use crate::variable::Variable;
use crate::graph::ComputationGraph;
use crate::op::{Conv2D, Add, MatMul, Sum, Transpose, Identity};
use std::rc::Rc;

pub trait Layer {
    fn forward(&self, input: &Variable, graph: &mut ComputationGraph) -> Result<Variable, TensorError>;
    fn backward(&self, input: &Variable, output_grad: &Variable, graph: &mut ComputationGraph) -> Result<(Variable, Vec<Variable>), TensorError>;
    fn initialize(&mut self, graph: &mut ComputationGraph) -> Result<(), TensorError>;
    fn parameters(&self) -> Vec<Variable>;
    fn update_parameters(&mut self, grads: &[Variable], learning_rate: f32) -> Result<(), TensorError>;
}

pub struct Dense {
    weights: Variable,
    bias: Variable,
    input_size: usize,
    output_size: usize,
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize, graph: &mut ComputationGraph) -> Self {
        let weights = graph.add_variable(Tensor::zeros(&[input_size, output_size]).unwrap(), true);
        let bias = graph.add_variable(Tensor::zeros(&[output_size]).unwrap(), true);
        Self {
            weights,
            bias,
            input_size,
            output_size,
        }
    }
}

impl Layer for Dense {
    fn forward(&self, input: &Variable, graph: &mut ComputationGraph) -> Result<Variable, TensorError> {
        let matmul_op = Rc::new(MatMul);
        let add_op = Rc::new(Add);
        
        let matmul_node = graph.add_node(matmul_op, vec![input.node.clone(), self.weights.node.clone()]);
        let output_node = graph.add_node(add_op, vec![matmul_node, self.bias.node.clone()]);
        
        Ok(Variable { node: output_node, requires_grad: true })
    }

    fn backward(&self, input: &Variable, output_grad: &Variable, graph: &mut ComputationGraph) -> Result<(Variable, Vec<Variable>), TensorError> {
        let matmul_op = Rc::new(MatMul);
        let transpose_op = Rc::new(Transpose);
        let sum_op = Rc::new(Sum);
        
        let transposed_input = graph.add_node(transpose_op.clone(), vec![input.node.clone()]);
        let weight_grad_node = graph.add_node(matmul_op.clone(), vec![transposed_input, output_grad.node.clone()]);
        
        let transposed_weights = graph.add_node(transpose_op, vec![self.weights.node.clone()]);
        let input_grad_node = graph.add_node(matmul_op, vec![output_grad.node.clone(), transposed_weights]);
        
        let bias_grad_node = graph.add_node(sum_op, vec![output_grad.node.clone()]);

        let input_grad = Variable::new(input_grad_node.borrow().id(), Rc::new(Identity), true);
        let weight_grad = Variable::new(weight_grad_node.borrow().id(), Rc::new(Identity), true);
        let bias_grad = Variable::new(bias_grad_node.borrow().id(), Rc::new(Identity), true);

        Ok((input_grad, vec![weight_grad, bias_grad]))
    }

    fn initialize(&mut self, _graph: &mut ComputationGraph) -> Result<(), TensorError> {
        let weight_stddev = (2.0 / (self.input_size + self.output_size) as f32).sqrt();
        let weight_data = Tensor::randn(&[self.input_size, self.output_size], 0.0, weight_stddev);
        self.weights.set_value(weight_data?);

        let bias_data = Tensor::zeros(&[self.output_size])?;
        self.bias.set_value(bias_data);

        Ok(())
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![self.weights.clone(), self.bias.clone()]
    }

    fn update_parameters(&mut self, grads: &[Variable], learning_rate: f32) -> Result<(), TensorError> {
        if grads.len() != 2 {
            return Err(TensorError::InvalidInputSize("Expected gradients for weights and bias".to_string()));
        }

        self.weights.update(&grads[0], learning_rate)?;
        self.bias.update(&grads[1], learning_rate)?;

        Ok(())
    }
}

pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Sequential { layers }
    }
}

impl Layer for Sequential {
    fn forward(&self, input: &Variable, graph: &mut ComputationGraph) -> Result<Variable, TensorError> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output, graph)?;
        }
        Ok(output)
    }

    fn backward(&self, input: &Variable, output_grad: &Variable, graph: &mut ComputationGraph) -> Result<(Variable, Vec<Variable>), TensorError> {
        let mut grad = output_grad.clone();
        let mut all_grads = Vec::new();
        let mut layer_inputs = vec![input.clone()];

        for layer in &self.layers {
            let output = layer.forward(layer_inputs.last().unwrap(), graph)?;
            layer_inputs.push(output);
        }

        for (layer, layer_input) in self.layers.iter().zip(layer_inputs.iter()).rev() {
            let (input_grad, param_grads) = layer.backward(layer_input, &grad, graph)?;
            grad = input_grad;
            all_grads.extend(param_grads);
        }

        Ok((grad, all_grads))
    }

    fn initialize(&mut self, graph: &mut ComputationGraph) -> Result<(), TensorError> {
        for layer in &mut self.layers {
            layer.initialize(graph)?;
        }
        Ok(())
    }

    fn parameters(&self) -> Vec<Variable> {
        self.layers.iter().flat_map(|layer| layer.parameters()).collect()
    }

    fn update_parameters(&mut self, grads: &[Variable], learning_rate: f32) -> Result<(), TensorError> {
        let mut grad_index = 0;
        for layer in &mut self.layers {
            let param_count = layer.parameters().len();
            let layer_grads = &grads[grad_index..grad_index + param_count];
            layer.update_parameters(layer_grads, learning_rate)?;
            grad_index += param_count;
        }
        Ok(())
    }
}

// Similar refactoring should be applied to Conv2DLayer, MaxPool2DLayer, FlattenLayer, ReLULayer, and LossLayer
// Here's an example for Conv2DLayer:

pub struct Conv2DLayer {
    weights: Variable,
    bias: Variable,
    kernel: Tensor
}

impl Conv2DLayer {
    pub fn new(in_channels: usize, out_channels: usize, kernel: Tensor, graph: &mut ComputationGraph) -> Self {
        let kernel_shape = kernel.shape();
        if kernel_shape.len() != 2 {
            panic!("Kernel must be a 2D tensor");
        }
        let kernel_size = (kernel_shape[0], kernel_shape[1]);
        let weights = graph.add_variable(Tensor::randn(&[out_channels, in_channels, kernel_size.0, kernel_size.1], 0.0, 0.05).unwrap(), true);
        let bias = graph.add_variable(Tensor::zeros(&[out_channels]).unwrap(), true);
        
        Conv2DLayer {
            weights,
            bias,
            kernel
        }
    }
}

impl Layer for Conv2DLayer {
    fn forward(&self, input: &Variable, graph: &mut ComputationGraph) -> Result<Variable, TensorError> {
        let conv_op = Rc::new(Conv2D::new(self.kernel.clone()));
        let add_op = Rc::new(Add);
        
        let conv_output = graph.add_node(conv_op, vec![input.node.clone(), self.weights.node.clone()]);
        let output = graph.add_node(add_op, vec![conv_output, self.bias.node.clone()]);
        
        let output_value = output.borrow().value().unwrap().clone();
        let output_id = output.borrow().id();
        Ok(Variable::from_tensor(output_id, output_value, true))
    }

    fn backward(&self, input: &Variable, output_grad: &Variable, graph: &mut ComputationGraph) -> Result<(Variable, Vec<Variable>), TensorError> {
        let conv_op = Rc::new(Conv2D::new(self.kernel.clone()));
        let transpose_op = Rc::new(Transpose);
        let sum_op = Rc::new(Sum);
        
        let transposed_input = graph.add_node(transpose_op.clone(), vec![input.node.clone()]);
        let weight_grad = graph.add_node(conv_op.clone(), vec![transposed_input, output_grad.node.clone()]);

        let transposed_weights = graph.add_node(transpose_op, vec![self.weights.node.clone()]);
        let input_grad = graph.add_node(conv_op, vec![output_grad.node.clone(), transposed_weights]);
        
        let bias_grad = graph.add_node(sum_op, vec![output_grad.node.clone()]);

        let input_grad_value = input_grad.borrow().value().unwrap().clone();
        let weight_grad_value = weight_grad.borrow().value().unwrap().clone();
        let bias_grad_value = bias_grad.borrow().value().unwrap().clone();
        let input_grad_var = Variable::from_tensor(input_grad.borrow().id(), input_grad_value, true);
        let weight_grad_var = Variable::from_tensor(weight_grad.borrow().id(), weight_grad_value, true);
        let bias_grad_var = Variable::from_tensor(bias_grad.borrow().id(), bias_grad_value, true);

        Ok((
            input_grad_var,
            vec![weight_grad_var, bias_grad_var]
        ))
    }

    fn initialize(&mut self, _graph: &mut ComputationGraph) -> Result<(), TensorError> {
        let weight_shape = self.weights.value().unwrap().shape().to_vec();
        let fan_in = weight_shape[1] * weight_shape[2] * weight_shape[3];
        let fan_out = weight_shape[0] * weight_shape[2] * weight_shape[3];
        let weight_stddev = (2.0 / (fan_in + fan_out) as f32).sqrt();
        
        let weight_data = Tensor::randn(&weight_shape, 0.0, weight_stddev);
        self.weights.set_value(weight_data.unwrap());

        let bias_shape = self.bias.value().unwrap().shape().to_vec();
        let bias_data = Tensor::zeros(&bias_shape)?;
        self.bias.set_value(bias_data);

        Ok(())
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![self.weights.clone(), self.bias.clone()]
    }

    fn update_parameters(&mut self, grads: &[Variable], learning_rate: f32) -> Result<(), TensorError> {
        if grads.len() != 2 {
            return Err(TensorError::InvalidInputSize("Expected gradients for weights and bias".to_string()));
        }

        self.weights.update(&grads[0], learning_rate)?;
        self.bias.update(&grads[1], learning_rate)?;

        Ok(())
    }
}
