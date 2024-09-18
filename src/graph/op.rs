use ndarray::ArrayD;

/// Trait representing an operation in the computational graph.
pub trait Op {
    /// Computes the output of the operation given input values.
    fn forward(&self, inputs: &[&ArrayD<f64>]) -> ArrayD<f64>;

    /// Computes the gradients w.r.t. the inputs given the gradient of the output.
    fn backward(&self, inputs: &[&ArrayD<f64>], grad_output: &ArrayD<f64>) -> Vec<ArrayD<f64>>;
}
