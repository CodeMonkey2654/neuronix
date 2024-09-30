use ndarray::{ArrayD, IxDyn, SliceInfo, SliceInfoElem};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use std::fmt;
use std::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};
use num_traits::Zero;

/// A Tensor struct that wraps around ndarray's ArrayD.
#[derive(Clone)]
pub struct Tensor<T> {
    data: ArrayD<T>,
}

impl<T> Tensor<T>
where
    T: Clone + Default,
{
    /// Creates a new Tensor from a vector of data and a shape.
    pub fn new(data: Vec<T>, shape: &[usize]) -> Self {
        let arr = ArrayD::from_shape_vec(IxDyn(shape), data)
            .expect("Shape and data length do not match.");
        Tensor { data: arr }
    }

    /// Creates a Tensor of zeros with the given shape.
    pub fn zeros(shape: &[usize]) -> Self
    where
        T: Zero,
    {
        let arr = ArrayD::<T>::zeros(IxDyn(shape));
        Tensor { data: arr }
    }

    /// Creates a Tensor of ones with the given shape.
    pub fn ones(shape: &[usize]) -> Self
    where
        T: Default + Clone + From<u8>,
    {
        let arr = ArrayD::<T>::from_elem(IxDyn(shape), T::from(1u8));
        Tensor { data: arr }
    }

    /// Creates a Tensor with random values uniformly distributed between low and high.
    pub fn rand(shape: &[usize], low: T, high: T) -> Self
    where
        T: rand::distributions::uniform::SampleUniform,
    {
        let arr = ArrayD::<T>::random(IxDyn(shape), Uniform::new(low, high));
        Tensor { data: arr }
    }

    /// Returns the shape of the Tensor.
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Reshapes the Tensor to a new shape.
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        let arr = self
            .data
            .clone()
            .into_shape_with_order(IxDyn(new_shape))
            .expect("Invalid shape for reshape.");
        Tensor { data: arr }
    }

    /// Transposes the Tensor by reversing its axes.
    pub fn transpose(&self) -> Self {
        Tensor {
            data: self.data.t().to_owned(),
        }
    }

    /// Permutes the dimensions of the Tensor according to the specified axes.
    pub fn permute(&self, axes: &[usize]) -> Self {
        Tensor {
            data: self.data.clone().permuted_axes(axes),
        }
    }

    /// Expands the dimensions of the Tensor at the specified axis.
    pub fn expand_dims(&self, axis: usize) -> Self {
        let mut shape = self.shape().to_vec();
        shape.insert(axis, 1);
        self.reshape(&shape)
    }

    /// Squeezes the dimensions of size 1 from the Tensor.
    pub fn squeeze(&self) -> Self {
        let shape: Vec<usize> = self.shape().iter().cloned().filter(|&x| x != 1).collect();
        self.reshape(&shape)
    }

    /// Returns a reference to the underlying data.
    pub fn data(&self) -> &ArrayD<T> {
        &self.data
    }

    /// Returns a mutable reference to the underlying data.
    pub fn data_mut(&mut self) -> &mut ArrayD<T> {
        &mut self.data
    }
}

impl<T> fmt::Debug for Tensor<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.data.fmt(f)
    }
}

/// Implementing arithmetic operations

impl<T> Add for Tensor<T>
where
    T: ndarray::ScalarOperand + Add<Output = T> + Clone,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Tensor {
            data: &self.data + &rhs.data,
        }
    }
}

impl<T> Sub for Tensor<T>
where
    T: ndarray::ScalarOperand + Sub<Output = T> + Clone,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Tensor {
            data: &self.data - &rhs.data,
        }
    }
}

impl<T> Mul for Tensor<T>
where
    T: ndarray::ScalarOperand + Mul<Output = T> + Clone,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Tensor {
            data: &self.data * &rhs.data,
        }
    }
}

impl<T> Div for Tensor<T>
where
    T: ndarray::ScalarOperand + Div<Output = T> + Clone,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Tensor {
            data: &self.data / &rhs.data,
        }
    }
}

impl<T> Neg for Tensor<T>
where
    T: ndarray::ScalarOperand + Clone,
    for<'a> &'a T: Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Tensor {
            data: self.data.map(|x| -x),
        }
    }
}

/// Implementing scalar operations

impl<T> Add<T> for Tensor<T>
where
    T: ndarray::ScalarOperand + Add<Output = T> + Clone,
{
    type Output = Self;

    fn add(self, scalar: T) -> Self::Output {
        Tensor {
            data: self.data + scalar,
        }
    }
}

impl<T> Sub<T> for Tensor<T>
where
    T: ndarray::ScalarOperand + Sub<Output = T> + Clone,
{
    type Output = Self;

    fn sub(self, scalar: T) -> Self::Output {
        Tensor {
            data: self.data - scalar,
        }
    }
}

impl<T> Mul<T> for Tensor<T>
where
    T: ndarray::ScalarOperand + Mul<Output = T> + Clone,
{
    type Output = Self;

    fn mul(self, scalar: T) -> Self::Output {
        Tensor {
            data: self.data * scalar,
        }
    }
}

impl<T> Div<T> for Tensor<T>
where
    T: ndarray::ScalarOperand + Div<Output = T> + Clone,
{
    type Output = Self;

    fn div(self, scalar: T) -> Self::Output {
        Tensor {
            data: self.data / scalar,
        }
    }
}

/// Implementing matrix multiplication

impl<T> Tensor<T>
where
    T: ndarray::ScalarOperand + ndarray::LinalgScalar + Clone,
{
    /// Performs matrix multiplication between two Tensors.
    pub fn matmul(&self, rhs: &Self) -> Self {
        let a = self.data.clone().into_dimensionality::<ndarray::Ix2>();
        let b = rhs.data.clone().into_dimensionality::<ndarray::Ix2>();
        match (a, b) {
            (Ok(a), Ok(b)) => {
                let result = a.dot(&b);
                Tensor {
                    data: result.into_dyn(),
                }
            }
            _ => panic!("Both tensors must be 2-dimensional for matmul."),
        }
    }
}

/// Implementing reduction operations

impl<T> Tensor<T>
where
    T: ndarray::LinalgScalar + std::iter::Sum + Copy + PartialOrd,
{
    /// Sums all elements in the Tensor.
    pub fn sum(&self) -> T {
        self.data.iter().cloned().sum()
    }

    /// Computes the mean of all elements in the Tensor.
    pub fn mean(&self) -> T
    where
        T: Div<Output = T> + From<usize>,
    {
        let total: T = self.sum();
        let count = self.data.len();
        total / T::from(count)
    }

    /// Finds the maximum element in the Tensor.
    pub fn max(&self) -> T {
        self.data
            .iter()
            .cloned()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .expect("Cannot compute max of empty tensor.")
    }

    /// Finds the minimum element in the Tensor.
    pub fn min(&self) -> T {
        self.data
            .iter()
            .cloned()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .expect("Cannot compute min of empty tensor.")
    }
}

/// Implementing indexing and slicing

impl<T> Index<usize> for Tensor<T>
where
    T: Clone,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> IndexMut<usize> for Tensor<T>
where
    T: Clone,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T> Tensor<T>
where
    T: Clone,
{
    /// Slices the Tensor using ndarray's s![] macro.
    pub fn slice(&self, info: &[SliceInfoElem]) -> Self {
        let slice_info = unsafe { SliceInfo::<_, ndarray::IxDyn, ndarray::IxDyn>::new(info).unwrap() };
        let view = self.data.slice(&slice_info);
        Tensor {
            data: view.to_owned(),
        }
    }
}

/// Implementing element-wise operations

impl<T> Tensor<T>
where
    T: Clone + ndarray::ScalarOperand + PartialEq,
{
    /// Compares elements and returns a Tensor of booleans where the condition is true.
    pub fn equal(&self, rhs: &Self) -> Tensor<bool> {
        Tensor {
            data: ndarray::Array::from_elem(self.data.shape(), self.data.eq(&rhs.data)),
        }
    }
}

impl Tensor<f32> {
    /// Applies the exponential function element-wise.
    pub fn exp(&self) -> Self {
        Tensor {
            data: self.data.mapv(|a| a.exp()),
        }
    }

    /// Applies the natural logarithm function element-wise.
    pub fn log(&self) -> Self {
        Tensor {
            data: self.data.mapv(|a| a.ln()),
        }
    }

    /// Applies the ReLU function element-wise.
    pub fn relu(&self) -> Self {
        Tensor {
            data: self.data.mapv(|a| if a > 0.0 { a } else { 0.0 }),
        }
    }

    pub fn pow(&self, scalar: f32) -> Self {
        Tensor {
            data: self.data.mapv(|a| a.powf(scalar)),
        }
    }
}

/// Example usage

fn main() {
    // Create two tensors
    let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let tensor2 = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);

    // Perform element-wise addition
    let result = tensor1.clone() + tensor2.clone();

    // Perform matrix multiplication
    let matmul_result = tensor1.matmul(&tensor2);

    // Compute sum
    let sum = tensor1.sum();

    // Print the results
    println!("Element-wise addition result:\n{:?}", result.data());
    println!("Matrix multiplication result:\n{:?}", matmul_result.data());
    println!("Sum of tensor1 elements: {:?}", sum);

    // Apply ReLU function
    let relu_result = tensor1.relu();
    println!("ReLU applied to tensor1:\n{:?}", relu_result.data());
}
