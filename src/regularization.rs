use crate::tensor::{Tensor, TensorError};
use crate::op::Op;
use std::any::Any;

#[derive(Debug, Clone)]
pub struct L2Regularization {
    pub lambda: f32,
}

impl Op for L2Regularization {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 1 {
            return Err(TensorError::InvalidInputSize("L2 Regularization operation requires 1 input".to_string()));
        }
        let weights = inputs[0];
        let l2_loss = &(weights * weights).sum(None)? * &Tensor::scalar(self.lambda / 2.0);
        Ok(l2_loss)
    }

    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        let weights = inputs[0];
        let grad = weights * &Tensor::scalar(self.lambda);
        Ok(vec![&grad * &output_gradient])
    }

    fn name(&self) -> &str {
        "L2Regularization"
    }

    fn box_clone(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug, Clone)]
pub struct L1Regularization {
    pub lambda: f32,
}

impl Op for L1Regularization {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
        if inputs.len() != 1 {
            return Err(TensorError::InvalidInputSize("L1 Regularization operation requires 1 input".to_string()));
        }
        let weights = inputs[0];
        let l1_loss = &weights.abs()?.sum(None)? * &Tensor::scalar(self.lambda);
        Ok(l1_loss)
    }

    fn backward(&self, inputs: &[&Tensor], output_gradient: &Tensor) -> Result<Vec<Tensor>, TensorError> {
        let weights = inputs[0];
        let grad = &weights.sign()? * &Tensor::scalar(self.lambda);
        Ok(vec![&grad * output_gradient])
    }

    fn name(&self) -> &str {
        "L1Regularization"
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
    fn test_l2_regularization() {
        let lambda = 0.1;
        let regularization = L2Regularization { lambda };
        let weights = Tensor::new(&[2, 2], &[1.0, -2.0, 3.0, -4.0], "f32").unwrap();
        let result = regularization.forward(&[&weights]).unwrap();
        let expected = Tensor::scalar(0.1 * (1.0 + 4.0 + 9.0 + 16.0));
        assert_tensor_eq(&result, &expected, 1e-6);

        let output_gradient = Tensor::scalar(1.0);
        let gradients = regularization.backward(&[&weights], &output_gradient).unwrap();
        let expected_grad = Tensor::new(&[2, 2], &[0.2, -0.4, 0.6, -0.8], "f32").unwrap();
        assert_tensor_eq(&gradients[0], &expected_grad, 1e-6);
    }

    #[test]
    fn test_l1_regularization() {
        let lambda = 0.1;
        let regularization = L1Regularization { lambda };

        let weights = Tensor::new(&[2, 2], &[1.0, -2.0, 3.0, -4.0], "f32").unwrap();
        let result = regularization.forward(&[&weights]).unwrap();
        let expected = Tensor::scalar(0.1 * (1.0 + 2.0 + 3.0 + 4.0));
        assert_tensor_eq(&result, &expected, 1e-6);

        let output_gradient = Tensor::scalar(1.0);
        let gradients = regularization.backward(&[&weights], &output_gradient).unwrap();
        let expected_grad = Tensor::new(&[2, 2], &[0.1, -0.1, 0.1, -0.1], "f32").unwrap();
        assert_tensor_eq(&gradients[0], &expected_grad, 1e-6);
    }

    #[test]
    fn test_l2_regularization_zero_lambda() {
        let lambda = 0.0;
        let regularization = L2Regularization { lambda };
        let weights = Tensor::new(&[2, 2], &[1.0, -2.0, 3.0, -4.0], "f32").unwrap();
        let result = regularization.forward(&[&weights]).unwrap();
        let expected = Tensor::scalar(0.0);
        assert_tensor_eq(&result, &expected, 1e-6);

        let output_gradient = Tensor::scalar(1.0);
        let gradients = regularization.backward(&[&weights], &output_gradient).unwrap();
        let expected_grad = Tensor::new(&[2, 2], &[0.0, 0.0, 0.0, 0.0], "f32").unwrap();
        assert_tensor_eq(&gradients[0], &expected_grad, 1e-6);
    }

    #[test]
    fn test_l1_regularization_zero_lambda() {
        let lambda = 0.0;
        let regularization = L1Regularization { lambda };
        let weights = Tensor::new(&[2, 2], &[1.0, -2.0, 3.0, -4.0], "f32").unwrap();
        let result = regularization.forward(&[&weights]).unwrap();
        let expected = Tensor::scalar(0.0);
        assert_tensor_eq(&result, &expected, 1e-6);

        let output_gradient = Tensor::scalar(1.0);
        let gradients = regularization.backward(&[&weights], &output_gradient).unwrap();
        let expected_grad = Tensor::new(&[2, 2], &[0.0, 0.0, 0.0, 0.0], "f32").unwrap();
        assert_tensor_eq(&gradients[0], &expected_grad, 1e-6);
    }
}
