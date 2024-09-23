use crate::tensor::{Tensor, TensorError};
use std::any::Any;

pub trait Op: std::fmt::Debug + 'static {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError>;
    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError>;
    fn name(&self) -> &str;
    fn box_clone(&self) -> Box<dyn Op>;
    fn as_any(&self) -> &dyn Any;
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

    fn as_any(&self) -> &dyn Any {
        self
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

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug, Clone)]
pub struct Linear;
impl Op for Linear {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 2 {
            return Err(TensorError::InvalidInputSize("Linear operation requires 2 inputs".to_string()));
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
        "Linear"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
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

    fn as_any(&self) -> &dyn Any {
        self
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

    fn as_any(&self) -> &dyn Any {
        self
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

    fn as_any(&self) -> &dyn Any {
        self
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

    fn as_any(&self) -> &dyn Any {
        self
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

    fn as_any(&self) -> &dyn Any {
        self
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

    fn as_any(&self) -> &dyn Any {
        self
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

    fn as_any(&self) -> &dyn Any {
        self
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

    fn as_any(&self) -> &dyn Any {
        self
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

    fn as_any(&self) -> &dyn Any {
        self
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
        let relu_x = inputs[0].gt(&Tensor::zeros(&inputs[0].shape(), "f32")?)?;
        Ok(vec![output_gradient * &relu_x])
    }

    fn name(&self) -> &str {
        "ReLU"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
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
        let sigmoid_derivative = &sigmoid_x * &(&Tensor::ones(&sigmoid_x.shape(), "f32")? - &sigmoid_x);
        Ok(vec![output_gradient * &sigmoid_derivative])
    }

    fn name(&self) -> &str {
        "Sigmoid"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
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

    fn as_any(&self) -> &dyn Any {
        self
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

    fn as_any(&self) -> &dyn Any {
        self
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
        let log_predictions = predictions.log()?;
        let product = targets * &log_predictions;
        let sum = product.sum_axis(1)?;
        let loss = sum.mean(None)?;
        let neg_loss = -&loss;
        Ok(neg_loss)
    }

    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        let epsilon = 1e-7;
        let predictions = inputs[0].clip(epsilon, 1.0 - epsilon)?;
        let targets = inputs[1];
        let batch_size = predictions.shape()[0] as f32;
        let grad = &-&(&(targets / &predictions) * &output_gradient) / &Tensor::scalar(batch_size);
        Ok(vec![grad, Tensor::zeros(&targets.shape(), "f32")?])
    }

    fn name(&self) -> &str {
        "CrossEntropyLoss"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
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
        let ones = Tensor::ones(&predictions.shape(), "f32")?;
        (&ones - &(predictions * targets)).max(Some(0))?.mean(None)
    }

    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        let predictions = inputs[0];
        let targets = inputs[1];
        let batch_size = Tensor::scalar(predictions.shape()[0] as f32);
        let condition = &(predictions * targets).lt(&Tensor::scalar(1.0))?;
        let true_case = &(&(-targets) * output_gradient) / &batch_size;
        let false_case = Tensor::zeros(&predictions.shape(), "f32")?;
        let grad = Tensor::where_cond(&condition, &true_case, &false_case, &predictions)?;
        Ok(vec![grad, Tensor::zeros(&targets.shape(), "f32")?])
    }

    fn name(&self) -> &str {
        "HingeLoss"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
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
        let one = Tensor::scalar(1.0);
        let loss = -(&(&(targets * &predictions.log()?) + &(&(&one - targets) * &(&(&one - &predictions).log()?))));
        loss.mean(None)
    }

    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        let epsilon = 1e-7;
        let predictions = inputs[0].clip(epsilon, 1.0 - epsilon)?;
        let targets = inputs[1];
        let batch_size = Tensor::scalar(predictions.shape()[0] as f32);
        let one = Tensor::scalar(1.0);
        let grad = &(&(&predictions - targets) / &(&predictions * &(&one - &predictions))) * output_gradient;
        let grad = &grad / &batch_size;
        Ok(vec![grad, Tensor::zeros(&targets.shape(), "f32")?])
    }

    fn name(&self) -> &str {
        "BinaryCrossEntropyLoss"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
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
        let quadratic = &(&diff * &diff) * &Tensor::scalar(0.5);
        let linear = &(&abs_diff - &Tensor::scalar(0.5 * self.delta)) * &Tensor::scalar(self.delta);
        let condition = abs_diff.lt(&Tensor::scalar(self.delta))?;
        let loss = condition.select(&quadratic, &linear)?;
        loss.mean(None)
    }

    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        let diff = inputs[0] - inputs[1];
        let abs_diff = diff.abs()?;
        let batch_size = Tensor::scalar(inputs[0].shape()[0] as f32);
        let condition = abs_diff.lt(&Tensor::scalar(self.delta))?;
        let grad = condition.select(&diff, &(&diff.sign()? * &Tensor::scalar(self.delta)))?;
        let scaled_grad = &(&grad * &output_gradient) / &batch_size;
        Ok(vec![scaled_grad.clone(), -&scaled_grad])
    }

    fn name(&self) -> &str {
        "HuberLoss"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
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
            &inputs[0].shape(),
        )?;
        let kernel_grad = output_gradient.conv2d_backward_kernel(
            &inputs[0],
            &self.kernel.shape(),
        )?;
        Ok(vec![input_grad, kernel_grad])
    }

    fn name(&self) -> &str {
        "Conv2D"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
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
            &inputs[0],
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

    fn as_any(&self) -> &dyn Any {
        self
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
        output_gradient.reshape(&inputs[0].shape()).map(|reshaped| vec![reshaped])
    }

    fn name(&self) -> &str {
        "Flatten"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
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

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug, Clone)]
pub struct Transpose;

impl Op for Transpose {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 1 {
            return Err(TensorError::InvalidInputSize("Transpose operation requires 1 input".to_string()));
        }
        inputs[0].transpose()
    }

    fn backward(&self, _inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        output_gradient.transpose().map(|transposed| vec![transposed])
    }

    fn name(&self) -> &str {
        "Transpose"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ... (existing code remains unchanged)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use approx::assert_relative_eq;
    use crate::tensor::TensorData;

    fn assert_tensor_eq(a: &Tensor, b: &Tensor, epsilon: f32) {
        assert_eq!(a.shape(), b.shape(), "Tensor shapes do not match");
        let a_data = &a.data;
        let b_data = &b.data;
        match (a_data, b_data) {
            (TensorData::F32(a_array), TensorData::F32(b_array)) => {
                for (a_val, b_val) in a_array.borrow().iter().zip(b_array.borrow().iter()) {
                    assert_relative_eq!(a_val, b_val, epsilon = epsilon);
                }
            },
            (TensorData::F64(a_array), TensorData::F64(b_array)) => {
                for (a_val, b_val) in a_array.borrow().iter().zip(b_array.borrow().iter()) {
                    assert_relative_eq!(a_val, b_val, epsilon = epsilon as f64);
                }
            },
            _ => panic!("Mismatched tensor data types"),
        }
    }

    #[test]
    fn test_add_forward() {
        let a = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0], "f32").unwrap();
        let b = Tensor::new(&[2, 2], &[5.0, 6.0, 7.0, 8.0], "f32").unwrap();
        let add_op = Add;
        let result = add_op.forward(&[&a, &b]).unwrap();
        let expected = Tensor::new(&[2, 2], &[6.0, 8.0, 10.0, 12.0], "f32").unwrap();
        assert_tensor_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn test_add_backward() {
        let a = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0], "f32").unwrap();
        let b = Tensor::new(&[2, 2], &[5.0, 6.0, 7.0, 8.0], "f32").unwrap();
        let add_op = Add;
        let output_gradient = Tensor::new(&[2, 2], &[1.0, 1.0, 1.0, 1.0], "f32").unwrap();
        let gradients = add_op.backward(&[&a, &b], &output_gradient).unwrap();
        assert_eq!(gradients.len(), 2);
        assert_tensor_eq(&gradients[0], &output_gradient, 1e-6);
        assert_tensor_eq(&gradients[1], &output_gradient, 1e-6);
    }

    #[test]
    fn test_subtract_forward() {
        let a = Tensor::new(&[2, 2], &[5.0, 6.0, 7.0, 8.0], "f32").unwrap();
        let b = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0], "f32").unwrap();
        let subtract_op = Subtract;
        let result = subtract_op.forward(&[&a, &b]).unwrap();
        let expected = Tensor::new(&[2, 2], &[4.0, 4.0, 4.0, 4.0], "f32").unwrap();
        assert_tensor_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn test_subtract_backward() {
        let a = Tensor::new(&[2, 2], &[5.0, 6.0, 7.0, 8.0], "f32").unwrap();
        let b = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0], "f32").unwrap();
        let subtract_op = Subtract;
        let output_gradient = Tensor::new(&[2, 2], &[1.0, 1.0, 1.0, 1.0], "f32").unwrap();
        let gradients = subtract_op.backward(&[&a, &b], &output_gradient).unwrap();
        assert_eq!(gradients.len(), 2);
        assert_tensor_eq(&gradients[0], &output_gradient, 1e-6);
        assert_tensor_eq(&gradients[1], &(-&output_gradient), 1e-6);
    }

    #[test]
    fn test_multiply_forward() {
        let a = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0], "f32").unwrap();
        let b = Tensor::new(&[2, 2], &[5.0, 6.0, 7.0, 8.0], "f32").unwrap();
        let multiply_op = Multiply;
        let result = multiply_op.forward(&[&a, &b]).unwrap();
        let expected = Tensor::new(&[2, 2], &[5.0, 12.0, 21.0, 32.0], "f32").unwrap();
        assert_tensor_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn test_multiply_backward() {
        let a = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0], "f32").unwrap();
        let b = Tensor::new(&[2, 2], &[5.0, 6.0, 7.0, 8.0], "f32").unwrap();
        let multiply_op = Multiply;
        let output_gradient = Tensor::new(&[2, 2], &[1.0, 1.0, 1.0, 1.0], "f32").unwrap();
        let gradients = multiply_op.backward(&[&a, &b], &output_gradient).unwrap();
        assert_eq!(gradients.len(), 2);
        assert_tensor_eq(&gradients[0], &b, 1e-6);
        assert_tensor_eq(&gradients[1], &a, 1e-6);
    }

    #[test]
    fn test_divide_forward() {
        let a = Tensor::new(&[2, 2], &[10.0, 12.0, 14.0, 16.0], "f32").unwrap();
        let b = Tensor::new(&[2, 2], &[2.0, 3.0, 2.0, 4.0], "f32").unwrap();
        let divide_op = Divide;
        let result = divide_op.forward(&[&a, &b]).unwrap();
        let expected = Tensor::new(&[2, 2], &[5.0, 4.0, 7.0, 4.0], "f32").unwrap();
        assert_tensor_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn test_divide_backward() {
        let a = Tensor::new(&[2, 2], &[10.0, 12.0, 14.0, 16.0], "f32").unwrap();
        let b = Tensor::new(&[2, 2], &[2.0, 3.0, 2.0, 4.0], "f32").unwrap();
        let divide_op = Divide;
        let output_gradient = Tensor::new(&[2, 2], &[1.0, 1.0, 1.0, 1.0], "f32").unwrap();
        let gradients = divide_op.backward(&[&a, &b], &output_gradient).unwrap();
        assert_eq!(gradients.len(), 2);
        let expected_grad_a = Tensor::new(&[2, 2], &[0.5, 1.0/3.0, 0.5, 0.25], "f32").unwrap();
        let expected_grad_b = Tensor::new(&[2, 2], &[-2.5, -4.0/3.0, -3.5, -1.0], "f32").unwrap();
        assert_tensor_eq(&gradients[0], &expected_grad_a, 1e-6);
        assert_tensor_eq(&gradients[1], &expected_grad_b, 1e-6);
    }

    #[test]
    fn test_matmul_forward() {
        let a = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "f32").unwrap();
        let b = Tensor::new(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], "f32").unwrap();
        let matmul_op = MatMul;
        let result = matmul_op.forward(&[&a, &b]).unwrap();
        let expected = Tensor::new(&[2, 2], &[58.0, 64.0, 139.0, 154.0], "f32").unwrap();
        assert_tensor_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn test_matmul_backward() {
        let a = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "f32").unwrap();
        let b = Tensor::new(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], "f32").unwrap();
        let matmul_op = MatMul;
        let output_gradient = Tensor::new(&[2, 2], &[1.0, 1.0, 1.0, 1.0], "f32").unwrap();
        let gradients = matmul_op.backward(&[&a, &b], &output_gradient).unwrap();
        assert_eq!(gradients.len(), 2);
        let expected_grad_a = Tensor::new(&[2, 3], &[15.0, 19.0, 23.0, 15.0, 19.0, 23.0], "f32").unwrap();
        let expected_grad_b = Tensor::new(&[3, 2], &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0], "f32").unwrap();
        assert_tensor_eq(&gradients[0], &expected_grad_a, 1e-6);
        assert_tensor_eq(&gradients[1], &expected_grad_b, 1e-6);
    }

    #[test]
    fn test_exp_forward() {
        let a = Tensor::new(&[2, 2], &[0.0, 1.0, 2.0, 3.0], "f32").unwrap();
        let exp_op = Exp;
        let result = exp_op.forward(&[&a]).unwrap();
        let expected = Tensor::new(&[2, 2], &[1.0, 2.71828, 7.38906, 20.08554], "f32").unwrap();
        assert_tensor_eq(&result, &expected, 1e-5);
    }

    #[test]
    fn test_exp_backward() {
        let a = Tensor::new(&[2, 2], &[0.0, 1.0, 2.0, 3.0], "f32").unwrap();
        let exp_op = Exp;
        let output_gradient = Tensor::new(&[2, 2], &[1.0, 1.0, 1.0, 1.0], "f32").unwrap();
        let gradients = exp_op.backward(&[&a], &output_gradient).unwrap();
        assert_eq!(gradients.len(), 1);
        let expected = Tensor::new(&[2, 2], &[1.0, 2.71828, 7.38906, 20.08554], "f32").unwrap();
        assert_tensor_eq(&gradients[0], &expected, 1e-5);
    }

    #[test]
    fn test_log_forward() {
        let a = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0], "f32").unwrap();
        let log_op = Log;
        let result = log_op.forward(&[&a]).unwrap();
        let expected = Tensor::new(&[2, 2], &[0.0, 0.69315, 1.09861, 1.38629], "f32").unwrap();
        assert_tensor_eq(&result, &expected, 1e-5);
    }

    #[test]
    fn test_log_backward() {
        let a = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0], "f32").unwrap();
        let log_op = Log;
        let output_gradient = Tensor::new(&[2, 2], &[1.0, 1.0, 1.0, 1.0], "f32").unwrap();
        let gradients = log_op.backward(&[&a], &output_gradient).unwrap();
        assert_eq!(gradients.len(), 1);
        let expected = Tensor::new(&[2, 2], &[1.0, 0.5, 1.0/3.0, 0.25], "f32").unwrap();
        assert_tensor_eq(&gradients[0], &expected, 1e-6);
    }

    #[test]
    fn test_pow_forward() {
        let base = Tensor::new(&[2, 2], &[2.0, 3.0, 4.0, 5.0], "f32").unwrap();
        let exponent = Tensor::new(&[2, 2], &[2.0, 2.0, 3.0, 3.0], "f32").unwrap();
        let pow_op = Pow;
        let result = pow_op.forward(&[&base, &exponent]).unwrap();
        let expected = Tensor::new(&[2, 2], &[4.0, 9.0, 64.0, 125.0], "f32").unwrap();
        assert_tensor_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn test_pow_backward() {
        let base = Tensor::new(&[2, 2], &[2.0, 3.0, 4.0, 5.0], "f32").unwrap();
        let exponent = Tensor::new(&[2, 2], &[2.0, 2.0, 3.0, 3.0], "f32").unwrap();
        let pow_op = Pow;
        let output_gradient = Tensor::new(&[2, 2], &[1.0, 1.0, 1.0, 1.0], "f32").unwrap();
        let gradients = pow_op.backward(&[&base, &exponent], &output_gradient).unwrap();
        assert_eq!(gradients.len(), 2);
        let expected_grad_base = Tensor::new(&[2, 2], &[4.0, 6.0, 48.0, 75.0], "f32").unwrap();
        let expected_grad_exponent = Tensor::new(&[2, 2], &[2.77259, 9.88751, 88.72284, 201.71315], "f32").unwrap();
        assert_tensor_eq(&gradients[0], &expected_grad_base, 1e-5);
        assert_tensor_eq(&gradients[1], &expected_grad_exponent, 1e-5);
    }

    #[test]
    fn test_sum_forward() {
        let a = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0], "f32").unwrap();
        let sum_op = Sum;
        let result = sum_op.forward(&[&a]).unwrap();
        let expected = Tensor::new(&[1], &[10.0], "f32").unwrap();
        assert_tensor_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn test_sum_backward() {
        let a = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0], "f32").unwrap();
        let sum_op = Sum;
        let output_gradient = Tensor::new(&[1], &[1.0], "f32").unwrap();
        let gradients = sum_op.backward(&[&a], &output_gradient).unwrap();
        assert_eq!(gradients.len(), 1);
        let expected = Tensor::new(&[2, 2], &[1.0, 1.0, 1.0, 1.0], "f32").unwrap();
        assert_tensor_eq(&gradients[0], &expected, 1e-6);
    }

    #[test]
    fn test_mean_forward() {
        let a = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0], "f32").unwrap();
        let mean_op = Mean;
        let result = mean_op.forward(&[&a]).unwrap();
        let expected = Tensor::new(&[1], &[2.5], "f32").unwrap();
        assert_tensor_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn test_mean_backward() {
        let a = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0], "f32").unwrap();
        let mean_op = Mean;
        let output_gradient = Tensor::new(&[1], &[1.0], "f32").unwrap();
        let gradients = mean_op.backward(&[&a], &output_gradient).unwrap();
        assert_eq!(gradients.len(), 1);
        let expected = Tensor::new(&[2, 2], &[0.25, 0.25, 0.25, 0.25], "f32").unwrap();
        assert_tensor_eq(&gradients[0], &expected, 1e-6);
    }

    #[test]
    fn test_tanh_forward() {
        let a = Tensor::new(&[2, 2], &[-1.0, 0.0, 1.0, 2.0], "f32").unwrap();
        let tanh_op = Tanh;
        let result = tanh_op.forward(&[&a]).unwrap();
        let expected = Tensor::new(&[2, 2], &[-0.76159, 0.0, 0.76159, 0.96403], "f32").unwrap();
        assert_tensor_eq(&result, &expected, 1e-5);
    }

    #[test]
    fn test_tanh_backward() {
        let a = Tensor::new(&[2, 2], &[-1.0, 0.0, 1.0, 2.0], "f32").unwrap();
        let tanh_op = Tanh;
        let output_gradient = Tensor::new(&[2, 2], &[1.0, 1.0, 1.0, 1.0], "f32").unwrap();
        let gradients = tanh_op.backward(&[&a], &output_gradient).unwrap();
        assert_eq!(gradients.len(), 1);
        let expected = Tensor::new(&[2, 2], &[0.41997, 1.0, 0.41997, 0.07065], "f32").unwrap();
        assert_tensor_eq(&gradients[0], &expected, 1e-5);
    }

    #[test]
    fn test_relu_forward() {
        let a = Tensor::new(&[2, 2], &[-1.0, 0.0, 1.0, 2.0], "f32").unwrap();
        let relu_op = ReLU;
        let result = relu_op.forward(&[&a]).unwrap();
        let expected = Tensor::new(&[2, 2], &[0.0, 0.0, 1.0, 2.0], "f32").unwrap();
        assert_tensor_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn test_relu_backward() {
        let a = Tensor::new(&[2, 2], &[-1.0, 0.0, 1.0, 2.0], "f32").unwrap();
        let relu_op = ReLU;
        let output_gradient = Tensor::new(&[2, 2], &[1.0, 1.0, 1.0, 1.0], "f32").unwrap();
        let gradients = relu_op.backward(&[&a], &output_gradient).unwrap();
        assert_eq!(gradients.len(), 1);
        let expected = Tensor::new(&[2, 2], &[0.0, 0.0, 1.0, 1.0], "f32").unwrap();
        assert_tensor_eq(&gradients[0], &expected, 1e-6);
    }

    #[test]
    fn test_sigmoid_forward() {
        let a = Tensor::new(&[2, 2], &[-1.0, 0.0, 1.0, 2.0], "f32").unwrap();
        let sigmoid_op = Sigmoid;
        let result = sigmoid_op.forward(&[&a]).unwrap();
        let expected = Tensor::new(&[2, 2], &[0.26894, 0.5, 0.73106, 0.88080], "f32").unwrap();
        assert_tensor_eq(&result, &expected, 1e-5);
    }

    #[test]
    fn test_sigmoid_backward() {
        let a = Tensor::new(&[2, 2], &[-1.0, 0.0, 1.0, 2.0], "f32").unwrap();
        let sigmoid_op = Sigmoid;
        let output_gradient = Tensor::new(&[2, 2], &[1.0, 1.0, 1.0, 1.0], "f32").unwrap();
        let gradients = sigmoid_op.backward(&[&a], &output_gradient).unwrap();
        assert_eq!(gradients.len(), 1);
        let expected = Tensor::new(&[2, 2], &[0.19661, 0.25, 0.19661, 0.10499], "f32").unwrap();
        assert_tensor_eq(&gradients[0], &expected, 1e-5);
    }

    #[test]
    fn test_softmax_forward() {
        let a = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0], "f32").unwrap();
        let softmax_op = Softmax;
        let result = softmax_op.forward(&[&a]).unwrap();
        let expected = Tensor::new(&[2, 2], &[0.26894, 0.73106, 0.26894, 0.73106], "f32").unwrap();
        assert_tensor_eq(&result, &expected, 1e-5);
    }

    #[test]
    fn test_softmax_backward() {
        let a = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0], "f32").unwrap();
        let softmax_op = Softmax;
        let output_gradient = Tensor::new(&[2, 2], &[1.0, 1.0, 1.0, 1.0], "f32").unwrap();
        let gradients = softmax_op.backward(&[&a], &output_gradient).unwrap();
        assert_eq!(gradients.len(), 1);
        let expected = Tensor::new(&[2, 2], &[0.19661, 0.19661, 0.19661, 0.19661], "f32").unwrap();
        assert_tensor_eq(&gradients[0], &expected, 1e-5);
    }

    #[test]
    fn test_mean_squared_error_forward() {
        let predictions = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0], "f32").unwrap();
        let targets = Tensor::new(&[2, 2], &[0.9, 2.1, 3.2, 3.8], "f32").unwrap();
        let mse_op = MeanSquaredError;
        let result = mse_op.forward(&[&predictions, &targets]).unwrap();
        let expected = Tensor::new(&[1], &[0.0125], "f32").unwrap();
        assert_tensor_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn test_mean_squared_error_backward() {
        let predictions = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0], "f32").unwrap();
        let targets = Tensor::new(&[2, 2], &[0.9, 2.1, 3.2, 3.8], "f32").unwrap();
        let mse_op = MeanSquaredError;
        let output_gradient = Tensor::new(&[1], &[1.0], "f32").unwrap();
        let gradients = mse_op.backward(&[&predictions, &targets], &output_gradient).unwrap();
        assert_eq!(gradients.len(), 2);
        let expected_grad_pred = Tensor::new(&[2, 2], &[0.05, -0.05, -0.1, 0.1], "f32").unwrap();
        let expected_grad_targets = Tensor::new(&[2, 2], &[-0.05, 0.05, 0.1, -0.1], "f32").unwrap();
        assert_tensor_eq(&gradients[0], &expected_grad_pred, 1e-6);
        assert_tensor_eq(&gradients[1], &expected_grad_targets, 1e-6);
    }

    #[test]
    fn test_cross_entropy_loss_forward() {
        let predictions = Tensor::new(&[2, 3], &[0.1, 0.2, 0.7, 0.3, 0.3, 0.4], "f32").unwrap();
        let targets = Tensor::new(&[2, 3], &[0.0, 0.0, 1.0, 0.0, 1.0, 0.0], "f32").unwrap();
        let ce_op = CrossEntropyLoss;
        let result = ce_op.forward(&[&predictions, &targets]).unwrap();
        let expected = Tensor::new(&[1], &[0.7133], "f32").unwrap();
        assert_tensor_eq(&result, &expected, 1e-4);
    }

    #[test]
    fn test_cross_entropy_loss_backward() {
        let predictions = Tensor::new(&[2, 3], &[0.1, 0.2, 0.7, 0.3, 0.3, 0.4], "f32").unwrap();
        let targets = Tensor::new(&[2, 3], &[0.0, 0.0, 1.0, 0.0, 1.0, 0.0], "f32").unwrap();
        let ce_op = CrossEntropyLoss;
        let output_gradient = Tensor::new(&[1], &[1.0], "f32").unwrap();
        let gradients = ce_op.backward(&[&predictions, &targets], &output_gradient).unwrap();
        assert_eq!(gradients.len(), 2);
        let expected_grad_pred = Tensor::new(&[2, 3], &[0.0, 0.0, -0.7143, 0.0, -1.6667, 0.0], "f32").unwrap();
        let expected_grad_targets = Tensor::zeros(&[2, 3], "f32").unwrap();
        assert_tensor_eq(&gradients[0], &expected_grad_pred, 1e-4);
        assert_tensor_eq(&gradients[1], &expected_grad_targets, 1e-6);
    }

    #[test]
    fn test_hinge_loss_forward() {
        let predictions = Tensor::new(&[2, 2], &[0.5, -0.5, 0.1, 0.9], "f32").unwrap();
        let targets = Tensor::new(&[2, 2], &[1.0, -1.0, 1.0, 1.0], "f32").unwrap();
        let hinge_op = HingeLoss;
        let result = hinge_op.forward(&[&predictions, &targets]).unwrap();
        let expected = Tensor::new(&[1], &[0.475], "f32").unwrap();
        assert_tensor_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn test_hinge_loss_backward() {
        let predictions = Tensor::new(&[2, 2], &[0.5, -0.5, 0.1, 0.9], "f32").unwrap();
        let targets = Tensor::new(&[2, 2], &[1.0, -1.0, 1.0, 1.0], "f32").unwrap();
        let hinge_op = HingeLoss;
        let output_gradient = Tensor::new(&[1], &[1.0], "f32").unwrap();
        let gradients = hinge_op.backward(&[&predictions, &targets], &output_gradient).unwrap();
        assert_eq!(gradients.len(), 2);
        let expected_grad_pred = Tensor::new(&[2, 2], &[-0.125, 0.0, -0.25, 0.0], "f32").unwrap();
        let expected_grad_targets = Tensor::zeros(&[2, 2], "f32").unwrap();
        assert_tensor_eq(&gradients[0], &expected_grad_pred, 1e-6);
        assert_tensor_eq(&gradients[1], &expected_grad_targets, 1e-6);
    }

    #[test]
    fn test_binary_cross_entropy_loss_forward() {
        let predictions = Tensor::new(&[2, 2], &[0.6, 0.4, 0.7, 0.3], "f32").unwrap();
        let targets = Tensor::new(&[2, 2], &[1.0, 0.0, 1.0, 0.0], "f32").unwrap();
        let bce_op = BinaryCrossEntropyLoss;
        let result = bce_op.forward(&[&predictions, &targets]).unwrap();
        let expected = Tensor::new(&[1], &[0.4581], "f32").unwrap();
        assert_tensor_eq(&result, &expected, 1e-4);
    }

    #[test]
    fn test_binary_cross_entropy_loss_backward() {
        let predictions = Tensor::new(&[2, 2], &[0.6, 0.4, 0.7, 0.3], "f32").unwrap();
        let targets = Tensor::new(&[2, 2], &[1.0, 0.0, 1.0, 0.0], "f32").unwrap();
        let bce_op = BinaryCrossEntropyLoss;
        let output_gradient = Tensor::new(&[1], &[1.0], "f32").unwrap();
        let gradients = bce_op.backward(&[&predictions, &targets], &output_gradient).unwrap();
        assert_eq!(gradients.len(), 2);
        let expected_grad_pred = Tensor::new(&[2, 2], &[-0.4167, 0.4167, -0.3571, 0.3571], "f32").unwrap();
        let expected_grad_targets = Tensor::zeros(&[2, 2], "f32").unwrap();
        assert_tensor_eq(&gradients[0], &expected_grad_pred, 1e-4);
        assert_tensor_eq(&gradients[1], &expected_grad_targets, 1e-6);
    }

    #[test]
    fn test_huber_loss_forward() {
        let predictions = Tensor::new(&[2, 2], &[0.9, 1.5, 2.1, 3.0], "f32").unwrap();
        let targets = Tensor::new(&[2, 2], &[1.0, 2.0, 2.0, 3.5], "f32").unwrap();
        let huber_op = HuberLoss::new(1.0);
        let result = huber_op.forward(&[&predictions, &targets]).unwrap();
        let expected = Tensor::new(&[1], &[0.0775], "f32").unwrap();
        assert_tensor_eq(&result, &expected, 1e-4);
    }

    #[test]
    fn test_huber_loss_backward() {
        let predictions = Tensor::new(&[2, 2], &[0.9, 1.5, 2.1, 3.0], "f32").unwrap();
        let targets = Tensor::new(&[2, 2], &[1.0, 2.0, 2.0, 3.5], "f32").unwrap();
        let huber_op = HuberLoss::new(1.0);
        let output_gradient = Tensor::new(&[1], &[1.0], "f32").unwrap();
        let gradients = huber_op.backward(&[&predictions, &targets], &output_gradient).unwrap();
        assert_eq!(gradients.len(), 2);
        let expected_grad_pred = Tensor::new(&[2, 2], &[-0.025, -0.125, 0.025, -0.125], "f32").unwrap();
        let expected_grad_targets = Tensor::new(&[2, 2], &[0.025, 0.125, -0.025, 0.125], "f32").unwrap();
        assert_tensor_eq(&gradients[0], &expected_grad_pred, 1e-4);
        assert_tensor_eq(&gradients[1], &expected_grad_targets, 1e-4);
    }

    #[test]
    fn test_conv2d_forward() {
        let input = Tensor::new(&[1, 1, 4, 4], &[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0
        ], "f32").unwrap();
        let kernel = Tensor::new(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0], "f32").unwrap();
        let conv2d_op = Conv2D::new(kernel);
        let result = conv2d_op.forward(&[&input]).unwrap();
        let expected = Tensor::new(&[1, 1, 3, 3], &[
            37.0, 47.0, 57.0,
            67.0, 77.0, 87.0,
            97.0, 107.0, 117.0
        ], "f32").unwrap();
        assert_tensor_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn test_conv2d_backward() {
        let input = Tensor::new(&[1, 1, 4, 4], &[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0
        ], "f32").unwrap();
        let kernel = Tensor::new(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0], "f32").unwrap();
        let conv2d_op = Conv2D::new(kernel);
        let output_gradient = Tensor::new(
            &[1, 1, 3, 3],
            &[
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0
            ],
            "f32"
        ).unwrap();
        let gradients = conv2d_op.backward(&[&input], &output_gradient).unwrap();
        assert_eq!(gradients.len(), 2);

        let expected_grad_input = Tensor::new(
            &[1, 1, 4, 4],
            &[
                1.0, 3.0, 3.0, 2.0,
                4.0, 10.0, 10.0, 6.0,
                4.0, 10.0, 10.0, 6.0,
                3.0, 7.0, 7.0, 4.0
            ],
            "f32"
        ).unwrap();
        let expected_grad_kernel = Tensor::new(
            &[1, 1, 2, 2],
            &[
                66.0, 72.0,
                84.0, 90.0
            ],
            "f32"
        ).unwrap();

        assert_tensor_eq(&gradients[0], &expected_grad_input, 1e-6);
        assert_tensor_eq(&gradients[1], &expected_grad_kernel, 1e-6);
    }

    #[test]
    fn test_max_pool2d_forward() {
        let input = Tensor::new(
            &[1, 1, 4, 4],
            &[
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0
            ],
            "f32"
        ).unwrap();
        let max_pool2d_op = MaxPool2D::new(2, 2, 0);
        let result = max_pool2d_op.forward(&[&input]).unwrap();
        let expected = Tensor::new(
            &[1, 1, 2, 2],
            &[
                6.0, 8.0,
                14.0, 16.0
            ],
            "f32"
        ).unwrap();
        assert_tensor_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn test_max_pool2d_backward() {
        let input = Tensor::new(
            &[1, 1, 4, 4],
            &[
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0
            ],
            "f32"
        ).unwrap();
        let max_pool2d_op = MaxPool2D::new(2, 2, 0);
        let output_gradient = Tensor::new(
            &[1, 1, 2, 2],
            &[
                1.0, 1.0,
                1.0, 1.0
            ],
            "f32"
        ).unwrap();
        let gradients = max_pool2d_op.backward(&[&input], &output_gradient).unwrap();
        assert_eq!(gradients.len(), 1);
        let expected_grad_input = Tensor::new(
            &[1, 1, 4, 4],
            &[
                0.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 1.0
            ],
            "f32"
        ).unwrap();
        assert_tensor_eq(&gradients[0], &expected_grad_input, 1e-6);
    }

    #[test]
    fn test_flatten_forward() {
        let input = Tensor::new(
            &[2, 2, 2],
            &[
                1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0,
                7.0, 8.0
            ],
            "f32"
        ).unwrap();
        let flatten_op = Flatten;
        let result = flatten_op.forward(&[&input]).unwrap();
        let expected = Tensor::new(
            &[2, 4],
            &[
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0
            ],
            "f32"
        ).unwrap();
        assert_tensor_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn test_flatten_backward() {
        let input = Tensor::new(
            &[2, 2, 2],
            &[
                1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0,
                7.0, 8.0
            ],
            "f32"
        ).unwrap();
        let flatten_op = Flatten;
        let output_gradient = Tensor::new(
            &[2, 4],
            &[
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0
            ],
            "f32"
        ).unwrap();
        let gradients = flatten_op.backward(&[&input], &output_gradient).unwrap();
        assert_eq!(gradients.len(), 1);
        let expected_grad_input = Tensor::new(
            &[2, 2, 2],
            &[
                1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0,
                7.0, 8.0
            ],
            "f32"
        ).unwrap();
        assert_tensor_eq(&gradients[0], &expected_grad_input, 1e-6);
    }

    #[test]
    fn test_identity_forward() {
        let input = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0], "f32").unwrap();
        let identity_op = Identity;
        let result = identity_op.forward(&[&input]).unwrap();
        assert_tensor_eq(&result, &input, 1e-6);
    }

    #[test]
    fn test_identity_backward() {
        let input = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0], "f32").unwrap();
        let identity_op = Identity;
        let output_gradient = Tensor::new(&[2, 2], &[1.0, 1.0, 1.0, 1.0], "f32").unwrap();
        let gradients = identity_op.backward(&[&input], &output_gradient).unwrap();
        assert_eq!(gradients.len(), 1);
        assert_tensor_eq(&gradients[0], &output_gradient, 1e-6);
    }

    #[test]
    fn test_transpose_forward() {
        let input = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "f32").unwrap();
        let transpose_op = Transpose;
        let result = transpose_op.forward(&[&input]).unwrap();
        let expected = Tensor::new(&[3, 2], &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], "f32").unwrap();
        assert_tensor_eq(&result, &expected, 1e-6);
    }

    #[test]
    fn test_transpose_backward() {
        let input = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "f32").unwrap();
        let transpose_op = Transpose;
        let output_gradient = Tensor::new(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "f32").unwrap();
        let gradients = transpose_op.backward(&[&input], &output_gradient).unwrap();
        assert_eq!(gradients.len(), 1);
        let expected_grad_input = Tensor::new(&[2, 3], &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0], "f32").unwrap();
        assert_tensor_eq(&gradients[0], &expected_grad_input, 1e-6);
    }
}