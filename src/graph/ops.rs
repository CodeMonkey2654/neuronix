use crate::graph::Op;
use ndarray::prelude::*;
use ndarray::{Axis};

/// Element-wise Addition Operation
pub struct AddOp;

impl Op for AddOp {
    fn forward(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        inputs[0] + inputs[1]
    }

    fn backward(&self, _inputs: &[&ArrayD<f64>], grad_output: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        vec![grad_output.clone(), grad_output.clone()]
    }
}

// Scalar Addition Operation
pub struct ScalarAddOp {
    scalar: f64,
}

impl ScalarAddOp {
    pub fn new(scalar: f64) -> Self {
        Self { scalar }
    }
}

impl Op for ScalarAddOp {
    fn forward(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        inputs[0] + self.scalar
    }

    fn backward(&self, _inputs: &[&ArrayD<f64>], grad_output: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        vec![grad_output.clone()]
    }
}

/// Matrix Multiplication Operation
pub struct MatMulOp {
    transpose_a: bool,
    transpose_b: bool,
}

impl MatMulOp {
    pub fn new() -> Self {
        Self {
            transpose_a: false,
            transpose_b: false,
        }
    }

    pub fn new_transpose_second() -> Self {
        Self {
            transpose_a: false,
            transpose_b: true,
        }
    }
}

impl Op for MatMulOp {
    fn forward(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        let a = inputs[0].clone();
        let b = inputs[1].clone();

        let a_transposed = if self.transpose_a { a.t().clone() } else { a.clone() };
        let b_transposed = if self.transpose_b { b.t().clone() } else { b.clone() };

        a_transposed.dot(&b_transposed)
    }

    fn backward(&self, inputs: &[&ArrayD<f64>], grad_output: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let a = inputs[0];
        let b = inputs[1];

        let grad_a = grad_output.dot(&b.t());
        let grad_b = a.t().dot(grad_output);

        vec![grad_a, grad_b]
    }
}

/// Scalar Multiplication Operation
pub struct ScalarMulOp {
    scalar: f64,
}

impl ScalarMulOp {
    pub fn new(scalar: f64) -> Self {
        Self { scalar }
    }
}

impl Op for ScalarMulOp {
    fn forward(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        inputs[0] * self.scalar
    }

    fn backward(&self, _inputs: &[&ArrayD<f64>], grad_output: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let grad_input = grad_output * self.scalar;
        vec![grad_input]
    }
}

/// ReLU Activation Operation
pub struct ReLUOp;

impl Op for ReLUOp {
    fn forward(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        inputs[0].mapv(|x| x.max(0.0))
    }

    fn backward(&self, inputs: &[&ArrayD<f64>], grad_output: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let input = inputs[0];
        let grad_input = input.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }) * grad_output;
        vec![grad_input]
    }
}

/// Sigmoid Activation Operation
pub struct SigmoidOp;

impl Op for SigmoidOp {
    fn forward(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        inputs[0].mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }

    fn backward(&self, inputs: &[&ArrayD<f64>], grad_output: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let sigmoid = inputs[0].mapv(|x| 1.0 / (1.0 + (-x).exp()));
        let grad_input = sigmoid * (1.0 - sigmoid) * grad_output;
        vec![grad_input]
    }
}

/// Tanh Activation Operation
pub struct TanhOp;

impl Op for TanhOp {
    fn forward(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        inputs[0].mapv(|x| x.tanh())
    }

    fn backward(&self, inputs: &[&ArrayD<f64>], grad_output: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let tanh = inputs[0].mapv(|x| x.tanh());
        let grad_input = (1.0 - tanh.mapv(|x| x * x)) * grad_output;
        vec![grad_input]
    }
}

/// Softmax Operation
pub struct SoftmaxOp {
    axis: isize,
}

impl SoftmaxOp {
    pub fn new(axis: isize) -> Self {
        Self { axis }
    }
}

impl Op for SoftmaxOp {
    fn forward(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        let input = inputs[0];
        let axis = if self.axis < 0 {
            input.ndim() as isize + self.axis
        } else {
            self.axis
        } as usize;

        let max = input.map_axis(Axis(axis), |x| x.fold(std::f64::NEG_INFINITY, |a, &b| a.max(b)));
        let max = max.insert_axis(Axis(axis));
        let exp = (input - &max).mapv(|x| x.exp());
        let sum_exp = exp.sum_axis(Axis(axis)).insert_axis(Axis(axis));
        &exp / &sum_exp
    }

    fn backward(&self, inputs: &[&ArrayD<f64>], grad_output: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let softmax = self.forward(inputs);
        let axis = if self.axis < 0 {
            inputs[0].ndim() as isize + self.axis
        } else {
            self.axis
        } as usize;

        let grad_input = softmax * (grad_output - (&softmax * grad_output).sum_axis(Axis(axis)).insert_axis(Axis(axis)));
        vec![grad_input]
    }
}

/// Reshape Operation
pub struct ReshapeOp {
    new_shape: Vec<usize>,
}

impl ReshapeOp {
    pub fn new(new_shape: Vec<usize>) -> Self {
        Self { new_shape }
    }
}

impl Op for ReshapeOp {
    fn forward(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        inputs[0].clone().into_shape(self.new_shape.clone()).unwrap()
    }

    fn backward(&self, _inputs: &[&ArrayD<f64>], grad_output: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let original_shape = grad_output.shape().to_vec();
        let grad_input = grad_output.clone().into_shape(original_shape).unwrap();
        vec![grad_input]
    }
}

/// Transpose Operation
pub struct TransposeOp {
    axes: Vec<usize>,
}

impl TransposeOp {
    pub fn new(axes: Vec<usize>) -> Self {
        Self { axes }
    }
}

impl Op for TransposeOp {
    fn forward(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        inputs[0].clone().permuted_axes(self.axes.clone())
    }

    fn backward(&self, _inputs: &[&ArrayD<f64>], grad_output: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        // Inverse permutation
        let mut inverse_axes = vec![0; self.axes.len()];
        for (i, &axis) in self.axes.iter().enumerate() {
            inverse_axes[axis] = i;
        }
        let grad_input = grad_output.clone().permuted_axes(inverse_axes);
        vec![grad_input]
    }
}

/// Concatenation Operation
pub struct ConcatOp {
    axis: isize,
}

impl ConcatOp {
    pub fn new(axis: isize) -> Self {
        Self { axis }
    }
}

impl Op for ConcatOp {
    fn forward(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        let axis = if self.axis < 0 {
            inputs[0].ndim() as isize + self.axis
        } else {
            self.axis
        } as usize;

        let inputs: Vec<ArrayD<f64>> = inputs.iter().map(|&x| x.clone().to_owned()).collect();
        ndarray::stack(Axis(axis), &inputs).unwrap()
    }

    fn backward(&self, inputs: &[&ArrayD<f64>], grad_output: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let axis = if self.axis < 0 {
            grad_output.ndim() as isize + self.axis
        } else {
            self.axis
        } as usize;

        let mut grads = Vec::new();
        let mut start = 0;
        for input in inputs {
            let len = input.shape()[axis];
            let end = start + len;
            let grad_slice = grad_output.slice_axis(Axis(axis), (start..end).into());
            grads.push(grad_slice.to_owned());
            start = end;
        }
        grads
    }
}

/// Slicing Operation
pub struct SliceOp {
    slices: Vec<(usize, Option<usize>, Option<isize>)>, // (start, end, step) per axis
}

impl SliceOp {
    pub fn new(slices: Vec<(usize, Option<usize>, Option<isize>)>) -> Self {
        Self { slices }
    }
}

impl Op for SliceOp {
    fn forward(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        let mut indices = ndarray::SliceInfo::<_, ndarray::IxDyn, >::new_empty(inputs[0].ndim());
        for (i, &(start, end, step)) in self.slices.iter().enumerate() {
            let end = end.unwrap_or(inputs[0].shape()[i]);
            let step = step.unwrap_or(1);
            indices.push(ndarray::Slice {
                start: start as isize,
                end: Some(end as isize),
                step,
            });
        }
        inputs[0].slice(&indices).to_owned()
    }

    fn backward(&self, inputs: &[&ArrayD<f64>], grad_output: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let mut grad_input = ArrayD::<f64>::zeros(inputs[0].shape());
        let mut indices = ndarray::SliceInfo::<_, ndarray::IxDyn, inputs[0].ndim()>::new_empty(inputs[0].ndim());
        for (i, &(start, end, step)) in self.slices.iter().enumerate() {
            let end = end.unwrap_or(inputs[0].shape()[i]);
            let step = step.unwrap_or(1);
            indices.push(ndarray::Slice {
                start: start as isize,
                end: Some(end as isize),
                step,
            });
        }
        grad_input.slice_mut(&indices).assign(&grad_output);
        vec![grad_input]
    }
}

/// Element-wise Subtraction Operation
pub struct SubtractOp;

impl Op for SubtractOp {
    fn forward(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        &inputs[0] - &inputs[1]
    }

    fn backward(&self, _inputs: &[&ArrayD<f64>], grad_output: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        vec![grad_output.clone(), -grad_output]
    }
}

/// Element-wise Squaring Operation
pub struct SquareOp;

impl Op for SquareOp {
    fn forward(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        inputs[0].mapv(|x| x * x)
    }

    fn backward(&self, inputs: &[&ArrayD<f64>], grad_output: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let input = inputs[0];
        let grad_input = 2.0 * input * grad_output;
        vec![grad_input]
    }
}

/// Mean Operation (Reduction over specified axes)
pub struct MeanOp {
    axes: Option<Vec<usize>>,
}

impl MeanOp {
    pub fn new(axes: Option<Vec<usize>>) -> Self {
        Self { axes }
    }
}

impl Op for MeanOp {
    fn forward(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        let input = inputs[0];
        if let Some(axes) = &self.axes {
            let mut result = input.clone();
            for &axis in axes {
                result = result.mean_axis(Axis(axis)).unwrap();
            }
            result
        } else {
            ArrayD::from_elem(ndarray::IxDyn(&[]), input.mean().unwrap())
        }
    }

    fn backward(&self, inputs: &[&ArrayD<f64>], grad_output: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let input = inputs[0];
        let count = input.len() as f64;
        let count_array = ArrayD::from_elem(input.shape(), 1.0 / count);
        let grad_input = grad_output.broadcast(input.shape()).unwrap() * &count_array;
        vec![grad_input]
    }
}

/// Element-wise Natural Logarithm Operation
pub struct LogOp;

impl Op for LogOp {
    fn forward(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        inputs[0].mapv(|x| x.ln())
    }

    fn backward(&self, inputs: &[&ArrayD<f64>], grad_output: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let input = inputs[0];
        let grad_input = grad_output / input;
        vec![grad_input]
    }
}

/// Element-wise Multiplication Operation
pub struct MultiplyOp;

impl Op for MultiplyOp {
    fn forward(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        inputs[0] * inputs[1]
    }

    fn backward(&self, inputs: &[&ArrayD<f64>], grad_output: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let grad_input_0 = grad_output * inputs[1];
        let grad_input_1 = grad_output * inputs[0];
        vec![grad_input_0, grad_input_1]
    }
}

/// Sum Operation (Reduction over specified axes)
pub struct SumOp {
    axes: Option<Vec<usize>>,
}

impl SumOp {
    pub fn new(axes: Option<Vec<usize>>) -> Self {
        Self { axes }
    }
}

impl Op for SumOp {
    fn forward(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        let input = inputs[0];
        if let Some(axes) = &self.axes {
            let mut result = input.clone();
            for &axis in axes {
                result = result.sum_axis(Axis(axis));
            }
            result
        } else {
            ArrayD::from_elem(ndarray::IxDyn(&[]), input.sum())
        }
    }

    fn backward(&self, inputs: &[&ArrayD<f64>], grad_output: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let input_shape = inputs[0].shape();
        let grad_input = grad_output.broadcast(input_shape).unwrap().to_owned();
        vec![grad_input]
    }
}

/// Element-wise Negation Operation
pub struct NegateOp;

impl Op for NegateOp {
    fn forward(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        -inputs[0]
    }

    fn backward(&self, _inputs: &[&ArrayD<f64>], grad_output: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        vec![-grad_output]
    }
}

// src/graph/op.rs

use rand::thread_rng;
use rand::Rng;

pub struct DropoutMaskOp {
    rate: f64,
}

impl DropoutMaskOp {
    pub fn new(rate: f64) -> Self {
        Self { rate }
    }
}

impl Op for DropoutMaskOp {
    fn forward(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        let input_shape = inputs[0].shape();
        let mut rng = thread_rng();
        let mask = inputs[0].mapv(|_| {
            if rng.gen_bool(1.0 - self.rate) {
                1.0
            } else {
                0.0
            }
        });
        mask
    }

    fn backward(&self, _inputs: &[&ArrayD<f64>], _grad_output: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        // The mask is not learnable, so gradients do not flow through it
        vec![ArrayD::zeros(_inputs[0].shape())]
    }
}

pub struct AbsOp;

impl Op for AbsOp {
    fn forward(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        inputs[0].mapv(|x| x.abs())
    }

    fn backward(&self, inputs: &[&ArrayD<f64>], grad_output: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let grad_input = inputs[0].mapv(|x| if x > 0.0 { 1.0 } else { -1.0 }) * grad_output;
        vec![grad_input]
    }
}

pub struct ReciprocalSqrtOp;

impl Op for ReciprocalSqrtOp {
    fn forward(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        inputs[0].mapv(|x| 1.0 / x.sqrt())
    }

    fn backward(&self, inputs: &[&ArrayD<f64>], grad_output: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let input = inputs[0];
        let grad_input = -0.5 * grad_output / input.mapv(|x| x * x * x);
        vec![grad_input]
    }
}

pub struct LinearOp;

impl Op for LinearOp {
    fn forward(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        let input = inputs[0];
        let weight = inputs[1];
        let bias = inputs[2];
        let dot_product = input.dot(weight);
        dot_product + bias
    }

    fn backward(&self, inputs: &[&ArrayD<f64>], grad_output: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let input = inputs[0];
        let weight = inputs[1];
        let grad_input = grad_output.dot(&weight.t());
        let grad_weight = input.t().dot(grad_output);
        let grad_bias = grad_output.sum_axis(Axis(0));
        vec![grad_input, grad_weight, grad_bias]
    }
}

pub struct IdentityOp;

impl Op for IdentityOp {
    fn forward(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64> {
        inputs[0].clone()
    }

    fn backward(&self, _inputs: &[&ArrayD<f64>], grad_output: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        vec![grad_output.clone()]
    }
}