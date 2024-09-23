use crate::tensor::{Tensor, TensorError};
use crate::variable::Variable;
use crate::graph::ComputationGraph;
use crate::op::{Add, MatMul, ReLU};
use std::rc::Rc;

pub trait Layer {
    fn forward(&self, input: &Variable, graph: &mut ComputationGraph) -> Result<Variable, TensorError>;
    fn parameters(&self) -> Vec<Variable>;
    fn initialize(&mut self, graph: &mut ComputationGraph) -> Result<(), TensorError>;
}

pub struct Dense {
    pub weights: Variable,
    pub bias: Variable,
    pub activation: Option<Rc<dyn Fn(&Variable, &mut ComputationGraph) -> Result<Variable, TensorError>>>,
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize, activation: Option<Rc<dyn Fn(&Variable, &mut ComputationGraph) -> Result<Variable, TensorError>>>) -> Self {
        let weights = Variable::from_tensor(0, Tensor::randn(&[input_size, output_size], 0.0, 0.01).unwrap(), true);
        let bias = Variable::from_tensor(1, Tensor::zeros(&[output_size, 1], "f32").unwrap(), true);
        Dense { weights, bias, activation }
    }
}

impl Layer for Dense {
    fn forward(&self, input: &Variable, graph: &mut ComputationGraph) -> Result<Variable, TensorError> {
        let matmul_op = Rc::new(MatMul);
        let add_op = Rc::new(Add);
        
        let matmul_node = graph.add_node(matmul_op, vec![input, &self.weights]);
        let output_node = graph.add_node(add_op, vec![&matmul_node, &self.bias]);
        
        let mut output = output_node;
        
        if let Some(activation) = &self.activation {
            output = activation(&output, graph)?;
        }
        
        Ok(output)
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![self.weights.clone(), self.bias.clone()]
    }

    fn initialize(&mut self, _graph: &mut ComputationGraph) -> Result<(), TensorError> {
        let weight_shape = self.weights.node.borrow().value().unwrap().shape();
        let weight_stddev = (2.0 / (weight_shape[0] + weight_shape[1]) as f32).sqrt();
        let weight_tensor = Tensor::randn(&weight_shape, 0.0, weight_stddev)?;
        self.weights.set_value(weight_tensor);

        let bias_shape = self.bias.node.borrow().value().unwrap().shape();
        let bias_tensor = Tensor::zeros(&bias_shape, "f32")?;
        self.bias.set_value(bias_tensor);
        
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

    fn parameters(&self) -> Vec<Variable> {
        self.layers.iter().flat_map(|layer| layer.parameters()).collect()
    }

    fn initialize(&mut self, graph: &mut ComputationGraph) -> Result<(), TensorError> {
        for layer in &mut self.layers {
            layer.initialize(graph)?;
        }
        Ok(())
    }
}

// Activation functions
pub fn relu(x: &Variable, graph: &mut ComputationGraph) -> Result<Variable, TensorError> {
    let relu_op = Rc::new(ReLU);
    let output_node = graph.add_node(relu_op, vec![x]);
    Ok(output_node)
}

pub fn sigmoid(x: &Variable, graph: &mut ComputationGraph) -> Result<Variable, TensorError> {
    let sigmoid_tensor = x.to_tensor().sigmoid()?;
    Ok(graph.create_variable(sigmoid_tensor, x.requires_grad))
}

pub fn tanh(x: &Variable, graph: &mut ComputationGraph) -> Result<Variable, TensorError> {
    let tanh_tensor = x.to_tensor().tanh()?;
    Ok(graph.create_variable(tanh_tensor, x.requires_grad))
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::graph::ComputationGraph;
    use approx::assert_relative_eq;

    #[test]
    fn test_dense_layer_creation() {
        let dense = Dense::new(10, 5, None);
        assert_eq!(dense.weights.node.borrow().value().unwrap().shape(), vec![10, 5]);
        assert_eq!(dense.bias.node.borrow().value().unwrap().shape(), vec![5, 1]);
    }

    #[test]
    fn test_dense_layer_forward() -> Result<(), TensorError> {
        let mut graph = ComputationGraph::new();
        let dense = Dense::new(3, 2, None);
        let input = Variable::from_tensor(0, Tensor::new(&[3, 1], &[1.0, 2.0, 3.0], "f32")?, true);
        
        let output = dense.forward(&input, &mut graph)?;
        assert_eq!(output.node.borrow().value().unwrap().shape(), vec![2, 1]);
        
        Ok(())
    }

    #[test]
    fn test_dense_layer_with_activation() -> Result<(), TensorError> {
        let mut graph = ComputationGraph::new();
        let dense = Dense::new(3, 2, Some(Rc::new(relu)));
        let input = Variable::from_tensor(0, Tensor::new(&[3, 1], &[1.0, 2.0, 3.0], "f32")?, true);
        
        let output = dense.forward(&input, &mut graph)?;
        assert_eq!(output.node.borrow().value().unwrap().shape(), vec![2, 1]);
        
        // Ensure all values are non-negative (ReLU property)
        let output_tensor = output.node.borrow().value().unwrap().to_array().unwrap(); // Store in a variable
        for value in output_tensor.iter() {
            assert!(*value >= 0.0);
        }
        
        Ok(())
    }

    #[test]
    fn test_dense_layer_parameters() {
        let dense = Dense::new(3, 2, None);
        let params = dense.parameters();
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_dense_layer_initialize() -> Result<(), TensorError> {
        let mut graph = ComputationGraph::new();
        let mut dense = Dense::new(3, 2, None);
        dense.initialize(&mut graph)?;
        
        let weight_shape = dense.weights.node.borrow().value().unwrap().shape();
        assert_eq!(weight_shape, vec![3, 2]);
        
        let bias_shape = dense.bias.node.borrow().value().unwrap().shape();
        assert_eq!(bias_shape, vec![2, 1]);
        
        Ok(())
    }

    #[test]
    fn test_sequential_layer_creation() {
        let layers: Vec<Box<dyn Layer>> = vec![
            Box::new(Dense::new(10, 5, None)),
            Box::new(Dense::new(5, 2, None)),
        ];
        let sequential = Sequential::new(layers);
        assert_eq!(sequential.layers.len(), 2);
    }

    #[test]
    fn test_sequential_layer_forward() -> Result<(), TensorError> {
        let mut graph = ComputationGraph::new();
        let layers: Vec<Box<dyn Layer>> = vec![
            Box::new(Dense::new(3, 2, Some(Rc::new(relu)))),
            Box::new(Dense::new(2, 1, None)),
        ];
        let sequential = Sequential::new(layers);
        let input = Variable::from_tensor(0, Tensor::new(&[3, 1], &[1.0, 2.0, 3.0], "f32")?, true);
        
        let output = sequential.forward(&input, &mut graph)?;
        assert_eq!(output.node.borrow().value().unwrap().shape(), vec![1, 1]);
        
        Ok(())
    }

    #[test]
    fn test_sequential_layer_parameters() {
        let layers: Vec<Box<dyn Layer>> = vec![
            Box::new(Dense::new(3, 2, None)),
            Box::new(Dense::new(2, 1, None)),
        ];
        let sequential = Sequential::new(layers);
        let params = sequential.parameters();
        assert_eq!(params.len(), 4); // 2 weights + 2 biases
    }

    #[test]
    fn test_sequential_layer_initialize() -> Result<(), TensorError> {
        let mut graph = ComputationGraph::new();
        let layers: Vec<Box<dyn Layer>> = vec![
            Box::new(Dense::new(3, 2, None)),
            Box::new(Dense::new(2, 1, None)),
        ];
        let mut sequential = Sequential::new(layers);
        sequential.initialize(&mut graph)?;
        
        Ok(())
    }

    #[test]
    fn test_relu_activation() -> Result<(), TensorError> {
        let mut graph = ComputationGraph::new();
        let input = Variable::from_tensor(0, Tensor::new(&[3, 1], &[-1.0, 0.0, 1.0], "f32")?, true);
        
        let output = relu(&input, &mut graph)?;
        let output_tensor = output.node.borrow().value().unwrap().to_array().unwrap();
        
        assert_relative_eq!(output_tensor[0], 0.0);
        assert_relative_eq!(output_tensor[1], 0.0);
        assert_relative_eq!(output_tensor[2], 1.0);
        
        Ok(())
    }

    #[test]
    fn test_sigmoid_activation() -> Result<(), TensorError> {
        let mut graph = ComputationGraph::new();
        let input = Variable::from_tensor(0, Tensor::new(&[3, 1], &[-1.0, 0.0, 1.0], "f32")?, true);
        
        let output = sigmoid(&input, &mut graph)?;
        let output_tensor = output.node.borrow().value().unwrap().to_array().unwrap();
        
        assert_relative_eq!(output_tensor[0], 0.26894142, epsilon = 1e-6);
        assert_relative_eq!(output_tensor[1], 0.5, epsilon = 1e-6);
        assert_relative_eq!(output_tensor[2], 0.7310586, epsilon = 1e-6);
        
        Ok(())
    }

    #[test]
    fn test_tanh_activation() -> Result<(), TensorError> {
        let mut graph = ComputationGraph::new();
        let input = Variable::from_tensor(0, Tensor::new(&[3, 1], &[-1.0, 0.0, 1.0], "f32")?, true);
        
        let output = tanh(&input, &mut graph)?;
        let output_tensor = output.node.borrow().value().unwrap().to_array().unwrap();
        
        assert_relative_eq!(output_tensor[0], -0.7615942, epsilon = 1e-6);
        assert_relative_eq!(output_tensor[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(output_tensor[2], 0.7615942, epsilon = 1e-6);
        
        Ok(())
    }
}