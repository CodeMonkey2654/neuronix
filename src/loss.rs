use crate::tensor::Tensor;
use crate::op::Op;
use crate::tensor::TensorError;
use std::any::Any;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::tensor::TensorData;
    use approx::assert_relative_eq;

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
}