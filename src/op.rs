use crate::tensor::{Tensor, TensorError};

pub trait Op: std::fmt::Debug + 'static {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError>;
    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError>;
    fn name(&self) -> &str;
    fn box_clone(&self) -> Box<dyn Op>;
}

impl Clone for Box<dyn Op> {
    fn clone(&self) -> Box<dyn Op> {
        self.box_clone()
    }
}

#[derive(Debug, Clone)]
pub struct Add;
impl Op for Add {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 2 {
            return Err(TensorError::InvalidInputSize("Add operation requires 2 inputs".to_string()));
        }
        Ok(inputs[0] + inputs[1])
    }

    fn backward(&self, _inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        Ok(vec![output_gradient.clone(), output_gradient.clone()])
    }

    fn name(&self) -> &str {
        "Add"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct Subtract;
impl Op for Subtract {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 2 {
            return Err(TensorError::InvalidInputSize("Subtract operation requires 2 inputs".to_string()));
        }
        Ok(inputs[0] - inputs[1])
    }

    fn backward(&self, _inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        Ok(vec![output_gradient.clone(), -output_gradient])
    }

    fn name(&self) -> &str {
        "Subtract"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct Multiply;
impl Op for Multiply {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 2 {
            return Err(TensorError::InvalidInputSize("Multiply operation requires 2 inputs".to_string()));
        }
        Ok(inputs[0] * inputs[1])
    }

    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        Ok(vec![
            output_gradient * inputs[1],
            output_gradient * inputs[0],
        ])
    }

    fn name(&self) -> &str {
        "Multiply"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct Divide;
impl Op for Divide {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 2 {
            return Err(TensorError::InvalidInputSize("Divide operation requires 2 inputs".to_string()));
        }
        Ok(inputs[0] / inputs[1])
    }

    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        let reciprocal = inputs[1].reciprocal()?;
        Ok(vec![
            output_gradient * &reciprocal,
            -&(&(output_gradient * inputs[0]).clone() * &(&reciprocal * &reciprocal).clone())
        ])
    }

    fn name(&self) -> &str {
        "Divide"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct MatMul;
impl Op for MatMul {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 2 {
            return Err(TensorError::InvalidInputSize("MatMul operation requires 2 inputs".to_string()));
        }
        inputs[0].matmul(inputs[1])
    }

    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        Ok(vec![
            output_gradient.matmul(&inputs[1].transpose()?)?,
            inputs[0].transpose()?.matmul(output_gradient)?,
        ])
    }

    fn name(&self) -> &str {
        "MatMul"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct Exp;
impl Op for Exp {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 1 {
            return Err(TensorError::InvalidInputSize("Exp operation requires 1 input".to_string()));
        }
        inputs[0].exp()
    }

    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        Ok(vec![output_gradient * &inputs[0].exp()?])
    }

    fn name(&self) -> &str {
        "Exp"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct Log;
impl Op for Log {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 1 {
            return Err(TensorError::InvalidInputSize("Log operation requires 1 input".to_string()));
        }
        inputs[0].log()
    }

    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        Ok(vec![output_gradient / inputs[0]])
    }

    fn name(&self) -> &str {
        "Log"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct Pow;
impl Op for Pow {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 2 {
            return Err(TensorError::InvalidInputSize("Pow operation requires 2 inputs".to_string()));
        }
        inputs[0].pow_tensor(inputs[1])
    }

    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        let base = inputs[0];
        let exponent = inputs[1];
        let grad_base = &base.pow_tensor(&(exponent - &Tensor::scalar(1.0)))? * exponent;
        let grad_exponent = &base.pow_tensor(exponent)? * &base.log()?;
        
        Ok(vec![
            output_gradient * &grad_base,
            output_gradient * &grad_exponent,
        ])
    }

    fn name(&self) -> &str {
        "Pow"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct Sum;
impl Op for Sum {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 1 {
            return Err(TensorError::InvalidInputSize("Sum operation requires 1 input".to_string()));
        }
        inputs[0].sum(None)
    }

    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        output_gradient.broadcast(&inputs[0].shape())
            .map(|broadcasted| vec![broadcasted])
    }

    fn name(&self) -> &str {
        "Sum"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct Mean;
impl Op for Mean {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 1 {
            return Err(TensorError::InvalidInputSize("Mean operation requires 1 input".to_string()));
        }
        inputs[0].mean(None)
    }

    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        let input_shape = inputs[0].shape();
        let num_elements = input_shape.iter().product::<usize>() as f32;
        output_gradient.broadcast(&input_shape)
            .map(|broadcasted| vec![&broadcasted / &Tensor::scalar(num_elements)])
    }

    fn name(&self) -> &str {
        "Mean"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct Tanh;
impl Op for Tanh {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 1 {
            return Err(TensorError::InvalidInputSize("Tanh operation requires 1 input".to_string()));
        }
        inputs[0].tanh()
    }

    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        let tanh_x = inputs[0].tanh()?;
        let one_minus_tanh_sq = &Tensor::ones(&tanh_x.shape(), "f32")? - &(&tanh_x * &tanh_x);
        Ok(vec![output_gradient * &one_minus_tanh_sq])
    }

    fn name(&self) -> &str {
        "Tanh"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct ReLU;
impl Op for ReLU {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 1 {
            return Err(TensorError::InvalidInputSize("ReLU operation requires 1 input".to_string()));
        }
        inputs[0].relu()
    }

    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        let relu_x = inputs[0].gt(&Tensor::zeros(inputs[0].shape())?)?;
        Ok(vec![output_gradient * relu_x])
    }

    fn name(&self) -> &str {
        "ReLU"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct Sigmoid;
impl Op for Sigmoid {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 1 {
            return Err(TensorError::InvalidInputSize("Sigmoid operation requires 1 input".to_string()));
        }
        inputs[0].sigmoid()
    }

    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        let sigmoid_x = inputs[0].sigmoid()?;
        let sigmoid_derivative = &sigmoid_x * (Tensor::ones(sigmoid_x.shape())? - &sigmoid_x);
        Ok(vec![output_gradient * sigmoid_derivative])
    }

    fn name(&self) -> &str {
        "Sigmoid"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}   

#[derive(Debug, Clone)]
pub struct Softmax;
impl Op for Softmax {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 1 {
            return Err(TensorError::InvalidInputSize("Softmax operation requires 1 input".to_string()));
        }
        inputs[0].softmax()
    }
    
    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        let softmax_output = self.forward(inputs)?;
        Ok(vec![softmax_output.dot(&output_gradient)?])
    }

    fn name(&self) -> &str {
        "Softmax"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct MeanSquaredError;
impl Op for MeanSquaredError {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 2 {
            return Err(TensorError::InvalidInputSize("Mean Squared Error operation requires 2 inputs".to_string()));
        }
        let diff = inputs[0] - inputs[1];
        (&diff * &diff).mean(Some(1))
    }

    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        let batch_size = inputs[0].shape()[0] as f32;
        let diff = inputs[0] - inputs[1];
        let grad = &(&diff * &Tensor::scalar(2.0 / batch_size)) * &output_gradient;
        Ok(vec![grad.clone(), -&grad])
    }

    fn name(&self) -> &str {
        "MeanSquaredError"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct CrossEntropyLoss;
impl Op for CrossEntropyLoss {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 2 {
            return Err(TensorError::InvalidInputSize("Cross Entropy Loss operation requires 2 inputs".to_string()));
        }
        let epsilon = 1e-7;
        let predictions = inputs[0].clip(epsilon, 1.0 - epsilon)?;
        let targets = inputs[1];
        -&(&(*targets * *predictions.log()?).sum_axis(Some(1))?.mean(None))
    }

    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        let epsilon = 1e-7;
        let predictions = inputs[0].clip(epsilon, 1.0 - epsilon)?;
        let targets = inputs[1];
        let batch_size = predictions.shape()[0] as f32;
        let grad = -((targets / &predictions) * output_gradient) / batch_size;
        Ok(vec![grad, Tensor::zeros(targets.shape())?])
    }

    fn name(&self) -> &str {
        "CrossEntropyLoss"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct HingeLoss;
impl Op for HingeLoss {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 2 {
            return Err(TensorError::InvalidInputSize("Hinge Loss operation requires 2 inputs".to_string()));
        }
        let predictions = inputs[0];
        let targets = inputs[1];
        (Tensor::ones(predictions.shape())? - predictions * targets).max(Some(0))?.mean(None)
    }

    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        let predictions = inputs[0];
        let targets = inputs[1];
        let batch_size = predictions.shape()[0] as f32;
        let condition = (predictions * targets).lt(&Tensor::scalar(1.0))?;
        let true_case = (-targets * output_gradient) / batch_size;
        let false_case = Tensor::zeros(predictions.shape())?;
        let grad = Tensor::where_cond(&condition, &true_case, &false_case)?;
        Ok(vec![grad, Tensor::zeros(targets.shape())?])
    }

    fn name(&self) -> &str {
        "HingeLoss"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct BinaryCrossEntropyLoss;
impl Op for BinaryCrossEntropyLoss {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 2 {
            return Err(TensorError::InvalidInputSize("Binary Cross Entropy Loss operation requires 2 inputs".to_string()));
        }
        let epsilon = 1e-7;
        let predictions = inputs[0].clip(epsilon, 1.0 - epsilon)?;
        let targets = inputs[1];
        let loss = targets * predictions.log()? + (Tensor::ones(targets.shape())? - targets) * (Tensor::ones(predictions.shape())? - &predictions).log()?;
        -loss.mean(Some(0))?
    }

    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        let epsilon = 1e-7;
        let predictions = inputs[0].clip(epsilon, 1.0 - epsilon)?;
        let targets = inputs[1];
        let batch_size = predictions.shape()[0] as f32;
        let grad = ((predictions - targets) / (predictions * (Tensor::ones(predictions.shape())? - &predictions))) * output_gradient / batch_size;
        Ok(vec![grad, Tensor::zeros(targets.shape())?])
    }

    fn name(&self) -> &str {
        "BinaryCrossEntropyLoss"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct HuberLoss {
    delta: f32,
}

impl HuberLoss {
    pub fn new(delta: f32) -> Self {
        HuberLoss { delta }
    }
}

impl Op for HuberLoss {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 2 {
            return Err(TensorError::InvalidInputSize("Huber Loss operation requires 2 inputs".to_string()));
        }
        let diff = inputs[0] - inputs[1];
        let abs_diff = diff.abs()?;
        let quadratic = &diff * &diff * 0.5;
        let linear = abs_diff.sub_scalar(0.5 * self.delta)? * self.delta;
        Tensor::where_cond(&abs_diff.lt(&Tensor::scalar(self.delta))?, &quadratic, &linear)?.mean(None)
    }

    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        let diff = inputs[0] - inputs[1];
        let abs_diff = diff.abs()?;
        let batch_size = inputs[0].shape()[0] as f32;
        let grad = Tensor::where_cond(
            &abs_diff.lt(&Tensor::scalar(self.delta))?,
            &diff,
            &(diff.sign()? * self.delta)
        )? * output_gradient / batch_size;

        Ok(vec![grad.clone(), -grad])
    }

    fn name(&self) -> &str {
        "HuberLoss"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct Conv2D {
    kernel: Tensor
}

impl Conv2D {
    pub fn new(kernel: Tensor) -> Self {
        Conv2D { kernel }
    }
}

impl Op for Conv2D {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 1 {
            return Err(TensorError::InvalidInputSize("Conv2D operation requires 1 input".to_string()));
        }
        inputs[0].conv2d(&self.kernel)
    }

    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        let input_grad = output_gradient.conv2d_backward_input(
            &self.kernel,
            inputs[0].shape(),
        )?;
        let kernel_grad = output_gradient.conv2d_backward_kernel(
            inputs[0],
            self.kernel.shape(),
        )?;
        Ok(vec![input_grad, kernel_grad])
    }

    fn name(&self) -> &str {
        "Conv2D"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct MaxPool2D {
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl MaxPool2D {
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        MaxPool2D { kernel_size, stride, padding }
    }
}

impl Op for MaxPool2D {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 1 {
            return Err(TensorError::InvalidInputSize("MaxPool2D operation requires 1 input".to_string()));
        }
        inputs[0].max_pool2d(self.kernel_size, self.stride, self.padding)
    }

    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        output_gradient.max_pool2d_backward(
            inputs[0],
            self.kernel_size,
            self.stride,
            self.padding
        ).map(|input_grad| vec![input_grad])
    }

    fn name(&self) -> &str {
        "MaxPool2D"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct Flatten;

impl Op for Flatten {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 1 {
            return Err(TensorError::InvalidInputSize("Flatten operation requires 1 input".to_string()));
        }
        let shape = inputs[0].shape();
        let batch_size = shape[0];
        let flattened_size: usize = shape.iter().skip(1).product();
        inputs[0].reshape(&[batch_size, flattened_size])
    }

    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        output_gradient.reshape(inputs[0].shape()).map(|reshaped| vec![reshaped])
    }

    fn name(&self) -> &str {
        "Flatten"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct Identity;

impl Op for Identity {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        Ok(inputs[0].clone())
    }

    fn backward(&self, _inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        Ok(vec![output_gradient.clone()])
    }

    fn name(&self) -> &str {
        "Identity"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct Transpose;

impl Op for Transpose {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        Ok(inputs[0].transpose())
    }

    fn backward(&self, _inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        Ok(vec![output_gradient.transpose()])
    }

    fn name(&self) -> &str {
        "Transpose"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }
}