use ndarray::Array2;
use ndarray_rand::RandomExt;
use rand_distr::{Normal, Uniform};
use rand::thread_rng;

/// Initializes parameters using Xavier/Glorot uniform initialization.
pub fn xavier_uniform(shape: (usize, usize)) -> Array2<f64> {
    let (fan_in, fan_out) = shape;
    let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
    let mut rng = thread_rng();
    let uniform = Uniform::new(-limit, limit);
    Array2::random_using(shape, uniform, &mut rng)
}

/// Initializes parameters using Xavier/Glorot normal initialization.
pub fn xavier_normal(shape: (usize, usize)) -> Array2<f64> {
    let (fan_in, fan_out) = shape;
    let std_dev = (2.0 / (fan_in + fan_out) as f64).sqrt();
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, std_dev).unwrap();
    Array2::random_using(shape, normal, &mut rng)
}

/// Initializes parameters using He normal initialization.
pub fn he_normal(shape: (usize, usize)) -> Array2<f64> {
    let fan_in = shape.0;
    let std_dev = (2.0 / fan_in as f64).sqrt();
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, std_dev).unwrap();
    Array2::random_using(shape, normal, &mut rng)
}

/// Initializes embedding matrices.
pub fn initialize_embedding(shape: (usize, usize)) -> Array2<f64> {
    // Use uniform distribution in [-0.5, 0.5]
    let mut rng = thread_rng();
    let uniform = Uniform::new(-0.5, 0.5);
    Array2::random_using(shape, uniform, &mut rng)
}

/// General parameter initialization function with Xavier uniform initialization.
pub fn initialize_parameters(shape: (usize, usize)) -> Array2<f64> {
    xavier_uniform(shape)
}
