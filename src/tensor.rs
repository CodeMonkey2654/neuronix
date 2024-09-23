

use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign};
use ndarray::{Axis, ArrayD, IxDyn, SliceInfo, SliceInfoElem};
use std::cmp::PartialEq;
use std::rc::Rc;
use std::cell::RefCell;
use rand_distr::Distribution;

#[derive(Debug, Clone)]
pub enum TensorData {
    F32(Rc<RefCell<ArrayD<f32>>>),
    F64(Rc<RefCell<ArrayD<f64>>>),
}

#[derive(Debug, Clone)]
pub enum TensorError {
    ShapeMismatch(String),
    InvalidSlice(String),
    UnsupportedDtype(String),
    BroadcastError(String),
    InvalidInputSize(String),
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::ShapeMismatch(msg) => write!(f, "Shape mismatch: {}", msg),
            TensorError::InvalidSlice(msg) => write!(f, "Invalid slice: {}", msg),
            TensorError::UnsupportedDtype(msg) => write!(f, "Unsupported dtype: {}", msg),
            TensorError::BroadcastError(msg) => write!(f, "Broadcast error: {}", msg),
            TensorError::InvalidInputSize(msg) => write!(f, "Invalid input size: {}", msg),
        }
    }
}


#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: TensorData,
}

impl Tensor {
    pub fn new<T: Into<f64> + Copy>(shape: &[usize], data: &[T], dtype: &str) -> Result<Self, TensorError> {
        match dtype {
            "f32" => {
                let f32_data: Vec<f32> = data.iter().map(|&x| x.into() as f32).collect();
                let array = ArrayD::from_shape_vec(shape, f32_data).map_err(|e| TensorError::ShapeMismatch(e.to_string()))?;
                Ok(Tensor::new_f32(array))
            },
            "f64" => {
                let f64_data: Vec<f64> = data.iter().map(|&x| x.into()).collect();
                let array = ArrayD::from_shape_vec(shape, f64_data).map_err(|e| TensorError::ShapeMismatch(e.to_string()))?;
                Ok(Tensor::new_f64(array))
            },
            _ => Err(TensorError::UnsupportedDtype("Unsupported dtype".to_string())),
        }
    }
    pub fn new_f32(data: ArrayD<f32>) -> Self {
        Tensor {
            data: TensorData::F32(Rc::new(RefCell::new(data))),
        }
    }

    pub fn new_f64(data: ArrayD<f64>) -> Self {
        Tensor {
            data: TensorData::F64(Rc::new(RefCell::new(data))),
        }
    }

    pub fn scalar(value: f32) -> Self {
        Tensor {
            data: TensorData::F32(Rc::new(RefCell::new(ArrayD::from_elem(vec![], value)))),
        }
    }

    pub fn slice(&self, slices: &[SliceInfoElem]) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let slice_info: SliceInfo<_, IxDyn, _> = unsafe { SliceInfo::new_unchecked(slices, std::marker::PhantomData, std::marker::PhantomData) };
                let sliced = array.borrow().slice(slice_info).to_owned();
                Ok(Tensor::new_f32(sliced))
            },
            TensorData::F64(array) => {
                let slice_info: SliceInfo<_, IxDyn, _> = unsafe { SliceInfo::new_unchecked(slices, std::marker::PhantomData, std::marker::PhantomData) };
                let sliced = array.borrow().slice(slice_info).to_owned();
                Ok(Tensor::new_f64(sliced))
            },
        }
    }

    pub fn clip(&self, min: f32, max: f32) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let clipped = array.borrow().mapv(|x| x.clamp(min, max));
                Ok(Tensor::new_f32(clipped))
            },
            TensorData::F64(array) => {
                let min = min as f64;
                let max = max as f64;
                let clipped = array.borrow().mapv(|x| x.clamp(min, max));
                Ok(Tensor::new_f64(clipped))
            },
        }
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let reshaped = array.borrow().clone().into_shape(new_shape)
                    .map_err(|e| TensorError::ShapeMismatch(e.to_string()))?;
                Ok(Tensor::new_f32(reshaped))
            },
            TensorData::F64(array) => {
                let reshaped = array.borrow().clone().into_shape(new_shape)
                    .map_err(|e| TensorError::ShapeMismatch(e.to_string()))?;
                Ok(Tensor::new_f64(reshaped))
            },
        }
    }

    pub fn broadcast(&self, new_shape: &[usize]) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let broadcasted = array.borrow().broadcast(new_shape)
                    .ok_or_else(|| TensorError::BroadcastError("Broadcasting failed".into()))?
                    .to_owned();
                Ok(Tensor::new_f32(broadcasted))
            },
            TensorData::F64(array) => {
                let broadcasted = array.borrow().broadcast(new_shape)
                    .ok_or_else(|| TensorError::BroadcastError("Broadcasting failed".into()))?
                    .to_owned();
                Ok(Tensor::new_f64(broadcasted))
            },
        }
    }

    pub fn sum_axis(&self, axis: usize) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let result = array.borrow().sum_axis(Axis(axis));
                Ok(Tensor::new_f32(result))
            },
            TensorData::F64(array) => {
                let result = array.borrow().sum_axis(Axis(axis));
                Ok(Tensor::new_f64(result))
            },
        }
    }

    pub fn get_f32(&self) -> Result<Rc<RefCell<ArrayD<f32>>>, TensorError> {
        match &self.data {
            TensorData::F32(array) => Ok(Rc::clone(array)),
            _ => Err(TensorError::UnsupportedDtype("Tensor is not of type f32".into())),
        }
    }

    pub fn get_f64(&self) -> Result<Rc<RefCell<ArrayD<f64>>>, TensorError> {
        match &self.data {
            TensorData::F64(array) => Ok(Rc::clone(array)),
            _ => Err(TensorError::UnsupportedDtype("Tensor is not of type f64".into())),
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        match &self.data {
            TensorData::F32(array) => array.borrow().shape().to_vec(),
            TensorData::F64(array) => array.borrow().shape().to_vec(),
        }
    }

    pub fn ndim(&self) -> usize {
        match &self.data {
            TensorData::F32(array) => array.borrow().ndim(),
            TensorData::F64(array) => array.borrow().ndim(),
        }
    }

    pub fn display(&self) {
        match &self.data {
            TensorData::F32(array) => println!("{:?}", array.borrow()),
            TensorData::F64(array) => println!("{:?}", array.borrow()),
        }
    }

    pub fn reciprocal(&self) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let reciprocal = array.borrow().mapv(|x| 1.0 / x);
                Ok(Tensor::new_f32(reciprocal))
            },
            TensorData::F64(array) => {
                let reciprocal = array.borrow().mapv(|x| 1.0 / x);
                Ok(Tensor::new_f64(reciprocal))
            },
        }
    }

    pub fn sign(&self) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let sign = array.borrow().mapv(|x| x.signum());
                Ok(Tensor::new_f32(sign))
            },
            TensorData::F64(array) => {
                let sign = array.borrow().mapv(|x| x.signum());
                Ok(Tensor::new_f64(sign))
            },
        }
    }

    pub fn abs(&self) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let abs = array.borrow().mapv(|x| x.abs());
                Ok(Tensor::new_f32(abs))
            },
            TensorData::F64(array) => {
                let abs = array.borrow().mapv(|x| x.abs());
                Ok(Tensor::new_f64(abs))
            },
        }
    }

    pub fn pow(&self, power: f32) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let mut result = array.borrow().clone();
                for elem in result.iter_mut() {
                    *elem = elem.powf(power);
                }
                Ok(Tensor::new_f32(result))
            },
            TensorData::F64(array) => {
                let mut result = array.borrow().clone();
                for elem in result.iter_mut() {
                    *elem = elem.powf(power as f64);
                }
                Ok(Tensor::new_f64(result))
            },
        }
    }

    pub fn pow_scalar(&self, power: f32) -> Result<Self, TensorError> {
        self.pow(power)
    }

    pub fn pow_tensor(&self, power: &Tensor) -> Result<Self, TensorError> {
        match (&self.data, &power.data) {
            (TensorData::F32(self_array), TensorData::F32(power_array)) => {
                let result = self_array.borrow().mapv(|x| x.powf(*power_array.borrow().first().unwrap()));
                Ok(Tensor::new_f32(result))
            },
            (TensorData::F64(self_array), TensorData::F32(power_array)) => {
                let result = self_array.borrow().mapv(|x| x.powf(*power_array.borrow().first().unwrap() as f64));
                Ok(Tensor::new_f64(result))
            },
            _ => Err(TensorError::UnsupportedDtype("Unsupported dtype combination for pow_tensor".to_string())),
        }
    }

    pub fn sqrt(&self) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let sqrt = array.borrow().mapv(|x| x.sqrt());
                Ok(Tensor::new_f32(sqrt))
            },
            TensorData::F64(array) => {
                let sqrt = array.borrow().mapv(|x| x.sqrt());
                Ok(Tensor::new_f64(sqrt))
            },
        }
    }

    pub fn exp(&self) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let exp = array.borrow().mapv(|x| x.exp());
                Ok(Tensor::new_f32(exp))
            },
            TensorData::F64(array) => {
                let exp = array.borrow().mapv(|x| x.exp());
                Ok(Tensor::new_f64(exp))
            },
        }
    }

    pub fn log(&self) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let log = array.borrow().mapv(|x| x.ln());
                Ok(Tensor::new_f32(log))
            },
            TensorData::F64(array) => {
                let log = array.borrow().mapv(|x| x.ln());
                Ok(Tensor::new_f64(log))
            },
        }
    }

    pub fn log2(&self) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let log2 = array.borrow().mapv(|x| x.log2());
                Ok(Tensor::new_f32(log2))
            },
            TensorData::F64(array) => {
                let log2 = array.borrow().mapv(|x| x.log2());
                Ok(Tensor::new_f64(log2))
            },
        }
    }

    pub fn log10(&self) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let log10 = array.borrow().mapv(|x| x.log10());
                Ok(Tensor::new_f32(log10))
            },
            TensorData::F64(array) => {
                let log10 = array.borrow().mapv(|x| x.log10());
                Ok(Tensor::new_f64(log10))
            },
        }
    }

    pub fn sin(&self) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let sin = array.borrow().mapv(|x| x.sin());
                Ok(Tensor::new_f32(sin))
            },
            TensorData::F64(array) => {
                let sin = array.borrow().mapv(|x| x.sin());
                Ok(Tensor::new_f64(sin))
            },
        }
    }

    pub fn cos(&self) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let cos = array.borrow().mapv(|x| x.cos());
                Ok(Tensor::new_f32(cos))
            },
            TensorData::F64(array) => {
                let cos = array.borrow().mapv(|x| x.cos());
                Ok(Tensor::new_f64(cos))
            },
        }
    }

    pub fn tan(&self) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let tan = array.borrow().mapv(|x| x.tan());
                Ok(Tensor::new_f32(tan))
            },
            TensorData::F64(array) => {
                let tan = array.borrow().mapv(|x| x.tan());
                Ok(Tensor::new_f64(tan))
            },
        }
    }

    pub fn tanh(&self) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let tanh = array.borrow().mapv(|x| x.tanh());
                Ok(Tensor::new_f32(tanh))
            },
            TensorData::F64(array) => {
                let tanh = array.borrow().mapv(|x| x.tanh());
                Ok(Tensor::new_f64(tanh))
            },
        }
    }

    pub fn asin(&self) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let asin = array.borrow().mapv(|x| x.asin());
                Ok(Tensor::new_f32(asin))
            },
            TensorData::F64(array) => {
                let asin = array.borrow().mapv(|x| x.asin());
                Ok(Tensor::new_f64(asin))
            },
        }
    }

    pub fn acos(&self) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let acos = array.borrow().mapv(|x| x.acos());
                Ok(Tensor::new_f32(acos))
            },
            TensorData::F64(array) => {
                let acos = array.borrow().mapv(|x| x.acos());
                Ok(Tensor::new_f64(acos))
            },
        }
    }

    pub fn atan(&self) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let atan = array.borrow().mapv(|x| x.atan());
                Ok(Tensor::new_f32(atan))
            },
            TensorData::F64(array) => {
                let atan = array.borrow().mapv(|x| x.atan());
                Ok(Tensor::new_f64(atan))
            },
        }
    }   
    
    pub fn mean(&self, axis: Option<usize>) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let mean = match axis {
                    Some(ax) => array.borrow().mean_axis(Axis(ax))
                        .ok_or_else(|| TensorError::InvalidInputSize("Invalid axis".into()))?,
                    None => ArrayD::from_elem(vec![], array.borrow().mean().unwrap()),
                };
                Ok(Tensor::new_f32(mean))
            },
            TensorData::F64(array) => {
                let mean = match axis {
                    Some(ax) => array.borrow().mean_axis(Axis(ax))
                        .ok_or_else(|| TensorError::InvalidInputSize("Invalid axis".into()))?,
                    None => ArrayD::from_elem(vec![], array.borrow().mean().unwrap()),
                };
                Ok(Tensor::new_f64(mean))
            },
        }
    }

    pub fn std(&self, axis: Option<usize>) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let array_ref = array.borrow();
                let mean = match axis {
                    Some(ax) => {
                        let sum = array_ref.sum_axis(Axis(ax));
                        let count = array_ref.len_of(Axis(ax)) as f32;
                        sum / count
                    },
                    None => {
                        let sum: f32 = array_ref.iter().sum();
                        let count = array_ref.len() as f32;
                        ArrayD::from_elem(vec![], sum / count)
                    },
                };
                let squared_diff = array_ref.mapv(|x| {
                    let diff = x - mean.iter().next().unwrap();
                    diff * diff
                });
                let variance = match axis {
                    Some(ax) => {
                        let sum = squared_diff.sum_axis(Axis(ax));
                        let count = squared_diff.len_of(Axis(ax)) as f32;
                        sum / count
                    },
                    None => {
                        let sum: f32 = squared_diff.iter().sum();
                        let count = squared_diff.len() as f32;
                        ArrayD::from_elem(vec![], sum / count)
                    },
                };
                Ok(Tensor::new_f32(variance.mapv(|x| x.sqrt())))
            },
            TensorData::F64(array) => {
                let array_ref = array.borrow();
                let mean = match axis {
                    Some(ax) => {
                        let sum = array_ref.sum_axis(Axis(ax));
                        let count = array_ref.len_of(Axis(ax)) as f64;
                        sum / count
                    },
                    None => {
                        let sum: f64 = array_ref.iter().sum();
                        let count = array_ref.len() as f64;
                        ArrayD::from_elem(vec![], sum / count)
                    },
                };
                let squared_diff = array_ref.mapv(|x| {
                    let diff = x - mean.iter().next().unwrap();
                    diff * diff
                });
                let variance = match axis {
                    Some(ax) => {
                        let sum = squared_diff.sum_axis(Axis(ax));
                        let count = squared_diff.len_of(Axis(ax)) as f64;
                        sum / count
                    },
                    None => {
                        let sum: f64 = squared_diff.iter().sum();
                        let count = squared_diff.len() as f64;
                        ArrayD::from_elem(vec![], sum / count)
                    },
                };
                Ok(Tensor::new_f64(variance.mapv(|x| x.sqrt())))
            },
        }
    }

    pub fn var(&self, axis: Option<usize>) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let array_ref = array.borrow();
                let mean = match axis {
                    Some(ax) => {
                        let sum = array_ref.sum_axis(Axis(ax));
                        let count = array_ref.len_of(Axis(ax)) as f32;
                        sum / count
                    },
                    None => {
                        let sum: f32 = array_ref.iter().sum();
                        let count = array_ref.len() as f32;
                        ArrayD::from_elem(vec![], sum / count)
                    },
                };
                let squared_diff = array_ref.mapv(|x| {
                    let diff = x - mean.iter().next().unwrap();
                    diff * diff
                });
                let variance = match axis {
                    Some(ax) => {
                        let sum = squared_diff.sum_axis(Axis(ax));
                        let count = squared_diff.len_of(Axis(ax)) as f32;
                        sum / count
                    },
                    None => {
                        let sum: f32 = squared_diff.iter().sum();
                        let count = squared_diff.len() as f32;
                        ArrayD::from_elem(vec![], sum / count)
                    },
                };
                Ok(Tensor::new_f32(variance))
            },
            TensorData::F64(array) => {
                let array_ref = array.borrow();
                let mean = match axis {
                    Some(ax) => {
                        let sum = array_ref.sum_axis(Axis(ax));
                        let count = array_ref.len_of(Axis(ax)) as f64;
                        sum / count
                    },
                    None => {
                        let sum: f64 = array_ref.iter().sum();
                        let count = array_ref.len() as f64;
                        ArrayD::from_elem(vec![], sum / count)
                    },
                };
                let squared_diff = array_ref.mapv(|x| {
                    let diff = x - mean.iter().next().unwrap();
                    diff * diff
                });
                let variance = match axis {
                    Some(ax) => {
                        let sum = squared_diff.sum_axis(Axis(ax));
                        let count = squared_diff.len_of(Axis(ax)) as f64;
                        sum / count
                    },
                    None => {
                        let sum: f64 = squared_diff.iter().sum();
                        let count = squared_diff.len() as f64;
                        ArrayD::from_elem(vec![], sum / count)
                    },
                };
                Ok(Tensor::new_f64(variance))
            },
        }
    }

    pub fn max(&self, axis: Option<usize>) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let max = match axis {
                    Some(ax) => array.borrow().map_axis(Axis(ax), |view| *view.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()),
                    None => ArrayD::from_elem(vec![], array.borrow().iter().cloned().fold(f32::NEG_INFINITY, f32::max)),
                };
                Ok(Tensor::new_f32(max))
            },
            TensorData::F64(array) => {
                let max = match axis {
                    Some(ax) => array.borrow().map_axis(Axis(ax), |view| *view.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()),
                    None => ArrayD::from_elem(vec![], array.borrow().iter().cloned().fold(f64::NEG_INFINITY, f64::max)),
                };
                Ok(Tensor::new_f64(max))
            },
        }
    }
    
    pub fn min(&self, axis: Option<usize>) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let min = match axis {
                    Some(ax) => array.borrow().map_axis(Axis(ax), |view| *view.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()),
                    None => ArrayD::from_elem(vec![], array.borrow().iter().cloned().fold(f32::INFINITY, f32::min)),
                };
                Ok(Tensor::new_f32(min))
            },
            TensorData::F64(array) => {
                let min = match axis {
                    Some(ax) => array.borrow().map_axis(Axis(ax), |view| *view.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()),
                    None => ArrayD::from_elem(vec![], array.borrow().iter().cloned().fold(f64::INFINITY, f64::min)),
                };
                Ok(Tensor::new_f64(min))
            },
        }
    }

    pub fn argmax(&self, axis: Option<usize>) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let argmax = match axis {
                    Some(ax) => array.borrow().map_axis(Axis(ax), |view| view.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0 as f32),
                    None => ArrayD::from_elem(vec![], array.borrow().iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0 as f32),
                };
                Ok(Tensor::new_f32(argmax))
            },
            TensorData::F64(array) => {
                let argmax = match axis {
                    Some(ax) => array.borrow().map_axis(Axis(ax), |view| view.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0 as f64),
                    None => ArrayD::from_elem(vec![], array.borrow().iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0 as f64),
                };
                Ok(Tensor::new_f64(argmax))
            },
        }
    }

    pub fn argmin(&self, axis: Option<usize>) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let argmin = match axis {
                    Some(ax) => array.borrow().map_axis(Axis(ax), |view| view.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0 as f32),
                    None => ArrayD::from_elem(vec![], array.borrow().iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0 as f32),
                };
                Ok(Tensor::new_f32(argmin))
            },
            TensorData::F64(array) => {
                let argmin = match axis {
                    Some(ax) => array.borrow().map_axis(Axis(ax), |view| view.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0 as f64),
                    None => ArrayD::from_elem(vec![], array.borrow().iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0 as f64),
                };
                Ok(Tensor::new_f64(argmin))
            },
        }
    }

    pub fn sum(&self, axis: Option<usize>) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let sum = match axis {
                    Some(ax) => {
                        let summed = array.borrow().sum_axis(Axis(ax));
                        ArrayD::from_shape_vec(summed.shape().to_vec(), summed.into_raw_vec())
                            .map_err(|e| TensorError::ShapeMismatch(e.to_string()))?
                    },
                    None => {
                        let total_sum = array.borrow().iter().fold(0.0, |acc, &x| acc + x);
                        ArrayD::from_elem(IxDyn(&[]), total_sum)
                    },
                };
                Ok(Tensor::new_f32(sum))
            },
            TensorData::F64(array) => {
                let sum = match axis {
                    Some(ax) => {
                        let summed = array.borrow().sum_axis(Axis(ax));
                        ArrayD::from_shape_vec(summed.shape().to_vec(), summed.into_raw_vec())
                            .map_err(|e| TensorError::ShapeMismatch(e.to_string()))?
                    },
                    None => {
                        let total_sum = array.borrow().iter().fold(0.0, |acc, &x| acc + x);
                        ArrayD::from_elem(IxDyn(&[]), total_sum)
                    },
                };
                Ok(Tensor::new_f64(sum))
            },
        }
    }

    pub fn dot(&self, other: &Tensor) -> Result<Self, TensorError> {
        if self.shape().len() != 2 || other.shape().len() != 2 {
            return Err(TensorError::ShapeMismatch("Dot product requires 2D tensors".to_string()));
        }
        
        let (m, n) = (self.shape()[0], self.shape()[1]);
        let (p, q) = (other.shape()[0], other.shape()[1]);
        
        if n != p {
            return Err(TensorError::ShapeMismatch("Inner dimensions must match for dot product".to_string()));
        }
        
        match (&self.data, &other.data) {
            (TensorData::F32(a), TensorData::F32(b)) => {
                let a = a.borrow();
                let b = b.borrow();
                let mut result = ArrayD::<f32>::zeros(IxDyn(&[m, q]));
                
                for i in 0..m {
                    for j in 0..q {
                        let mut sum = 0.0;
                        for k in 0..n {
                            sum += a[[i, k]] * b[[k, j]];
                        }
                        result[[i, j]] = sum;
                    }
                }
                
                Ok(Tensor::new_f32(result))
            },
            (TensorData::F64(a), TensorData::F64(b)) => {
                let a = a.borrow();
                let b = b.borrow();
                let mut result = ArrayD::<f64>::zeros(IxDyn(&[m, q]));
                
                for i in 0..m {
                    for j in 0..q {
                        let mut sum = 0.0;
                        for k in 0..n {
                            sum += a[[i, k]] * b[[k, j]];
                        }
                        result[[i, j]] = sum;
                    }
                }
                
                Ok(Tensor::new_f64(result))
            },
            _ => Err(TensorError::UnsupportedDtype("Dot product requires both tensors to have the same data type (F32 or F64)".to_string())),
        }
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Self, TensorError> {
        if self.shape().len() != 2 || other.shape().len() != 2 {
            return Err(TensorError::ShapeMismatch("Matmul requires 2D tensors".to_string()));
        }
        
        let (m, n) = (self.shape()[0], self.shape()[1]);
        let (p, q) = (other.shape()[0], other.shape()[1]);
        
        if n != p {
            return Err(TensorError::ShapeMismatch("Inner dimensions must match for matrix multiplication".to_string()));
        }
        
        match (&self.data, &other.data) {
            (TensorData::F32(a), TensorData::F32(b)) => {
                let a = a.borrow();
                let b = b.borrow();
                let mut result = ArrayD::<f32>::zeros(IxDyn(&[m, q]));
                
                for i in 0..m {
                    for j in 0..q {
                        let mut sum = 0.0;
                        for k in 0..n {
                            sum += a[[i, k]] * b[[k, j]];
                        }
                        result[[i, j]] = sum;
                    }
                }
                
                Ok(Tensor::new_f32(result))
            },
            (TensorData::F64(a), TensorData::F64(b)) => {
                let a = a.borrow();
                let b = b.borrow();
                let mut result = ArrayD::<f64>::zeros(IxDyn(&[m, q]));
                
                for i in 0..m {
                    for j in 0..q {
                        let mut sum = 0.0;
                        for k in 0..n {
                            sum += a[[i, k]] * b[[k, j]];
                        }
                        result[[i, j]] = sum;
                    }
                }
                
                Ok(Tensor::new_f64(result))
            },
            _ => Err(TensorError::UnsupportedDtype("Matmul requires both tensors to have the same data type (F32 or F64)".to_string())),
        }
    }

    pub fn transpose(&self) -> Result<Self, TensorError> {
        if self.shape().len() != 2 {
            return Err(TensorError::ShapeMismatch("Transpose requires a 2D tensor".to_string()));
        }
        
        match &self.data {
            TensorData::F32(array) => {
                let array_ref = array.borrow();
                let transposed = array_ref.t();
                Ok(Tensor::new_f32(transposed.to_owned()))
            },
            TensorData::F64(array) => {
                let array_ref = array.borrow();
                let transposed = array_ref.t();
                Ok(Tensor::new_f64(transposed.to_owned()))
            }
        }
    }

    pub fn relu(&self) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let array_ref = array.borrow();
                let relu = array_ref.mapv(|x| x.max(0.0));
                Ok(Tensor::new_f32(relu))
            },
            TensorData::F64(array) => {
                let array_ref = array.borrow();
                let relu = array_ref.mapv(|x| x.max(0.0));
                Ok(Tensor::new_f64(relu))
            }
        }
    }

    pub fn sigmoid(&self) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let array_ref = array.borrow();
                let sigmoid = array_ref.mapv(|x| 1.0 / (1.0 + (-x).exp()));
                Ok(Tensor::new_f32(sigmoid))
            },
            TensorData::F64(array) => {
                let array_ref = array.borrow();
                let sigmoid = array_ref.mapv(|x| 1.0 / (1.0 + (-x).exp()));
                Ok(Tensor::new_f64(sigmoid))
            },
        }
    }

    pub fn softmax(&self) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let array_ref = array.borrow();
                let softmax = array_ref.mapv(|x| x.exp() / array_ref.mapv(|y| y.exp()).sum());
                Ok(Tensor::new_f32(softmax))
            },
            TensorData::F64(array) => {
                let array_ref = array.borrow();
                let softmax = array_ref.mapv(|x| x.exp() / array_ref.mapv(|y| y.exp()).sum());
                Ok(Tensor::new_f64(softmax))
            },
        }
    }

    pub fn gt(&self, other: &Tensor) -> Result<Self, TensorError> {
        match (&self.data, &other.data) {
            (TensorData::F32(a), TensorData::F32(b)) => {
                let a = a.borrow();
                let b = b.borrow();
                let mut result = a.mapv(|_| 0.0f32);
                result.zip_mut_with(&a, |res, &a_val| {
                    *res = if a_val > *b.iter().next().unwrap_or(&0.0) { 1.0 } else { 0.0 };
                });
                Ok(Tensor::new_f32(result))
            },
            (TensorData::F64(a), TensorData::F64(b)) => {
                let a = a.borrow();
                let b = b.borrow();
                let mut result = a.mapv(|_| 0.0f64);
                result.zip_mut_with(&a, |res, &a_val| {
                    *res = if a_val > *b.iter().next().unwrap_or(&0.0) { 1.0 } else { 0.0 };
                });
                Ok(Tensor::new_f64(result))
            },
            _ => Err(TensorError::UnsupportedDtype("GT requires both tensors to have the same data type (F32 or F64)".to_string())),
        }
    }

    pub fn lt(&self, other: &Tensor) -> Result<Self, TensorError> {
        match (&self.data, &other.data) {
            (TensorData::F32(a), TensorData::F32(b)) => {
                let a = a.borrow();
                let b = b.borrow();
                let mut result = a.mapv(|_| 0.0f32);
                result.zip_mut_with(&a, |res, &a_val| {
                    *res = if a_val < *b.iter().next().unwrap_or(&0.0) { 1.0 } else { 0.0 };
                });
                Ok(Tensor::new_f32(result))
            },
            (TensorData::F64(a), TensorData::F64(b)) => {
                let a = a.borrow();
                let b = b.borrow();
                let mut result = a.mapv(|_| 0.0f64);
                result.zip_mut_with(&a, |res, &a_val| {
                    *res = if a_val < *b.iter().next().unwrap_or(&0.0) { 1.0 } else { 0.0 };
                });
                Ok(Tensor::new_f64(result))
            },
            _ => Err(TensorError::UnsupportedDtype("LT requires both tensors to have the same data type (F32 or F64)".to_string())),
        }
    }

    pub fn conv2d(&self, kernel: &Tensor) -> Result<Self, TensorError> {
        match (&self.data, &kernel.data) {
            (TensorData::F32(input), TensorData::F32(kernel)) => {
                let input = input.borrow();
                let kernel = kernel.borrow();
                let (batch_size, in_channels, in_height, in_width) = (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);
                let (out_channels, _, kernel_height, kernel_width) = (kernel.shape()[0], kernel.shape()[1], kernel.shape()[2], kernel.shape()[3]);

                let out_height = in_height - kernel_height + 1;
                let out_width = in_width - kernel_width + 1;

                let mut output = ArrayD::zeros(IxDyn(&[batch_size, out_channels, out_height, out_width]));

                for b in 0..batch_size {
                    for oc in 0..out_channels {
                        for oh in 0..out_height {
                            for ow in 0..out_width {
                                let mut sum = 0.0;
                                for ic in 0..in_channels {
                                    for kh in 0..kernel_height {
                                        for kw in 0..kernel_width {
                                            sum += input[[b, ic, oh + kh, ow + kw]] * kernel[[oc, ic, kh, kw]];
                                        }
                                    }
                                }
                                output[[b, oc, oh, ow]] = sum;
                            }
                        }
                    }
                }

                Ok(Tensor::new_f32(output))
            },
            (TensorData::F64(input), TensorData::F64(kernel)) => {
                let input = input.borrow();
                let kernel = kernel.borrow();
                let (batch_size, in_channels, in_height, in_width) = (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);
                let (out_channels, _, kernel_height, kernel_width) = (kernel.shape()[0], kernel.shape()[1], kernel.shape()[2], kernel.shape()[3]);

                let out_height = in_height - kernel_height + 1;
                let out_width = in_width - kernel_width + 1;

                let mut output = ArrayD::zeros(IxDyn(&[batch_size, out_channels, out_height, out_width]));

                for b in 0..batch_size {
                    for oc in 0..out_channels {
                        for oh in 0..out_height {
                            for ow in 0..out_width {
                                let mut sum = 0.0;
                                for ic in 0..in_channels {
                                    for kh in 0..kernel_height {
                                        for kw in 0..kernel_width {
                                            sum += input[[b, ic, oh + kh, ow + kw]] * kernel[[oc, ic, kh, kw]];
                                        }
                                    }
                                }
                                output[[b, oc, oh, ow]] = sum;
                            }
                        }
                    }
                }

                Ok(Tensor::new_f64(output))
            },
            _ => Err(TensorError::UnsupportedDtype("Conv2D requires both tensors to have the same data type (F32 or F64)".to_string())),
        }
    }

    pub fn conv2d_backward_input(&self, kernel: &Tensor, input_shape: &[usize]) -> Result<Self, TensorError> {
        match (&self.data, &kernel.data) {
            (TensorData::F32(output_grad), TensorData::F32(kernel)) => {
                let output_grad = output_grad.borrow();
                let kernel = kernel.borrow();
                let (batch_size, out_channels, out_height, out_width) = (output_grad.shape()[0], output_grad.shape()[1], output_grad.shape()[2], output_grad.shape()[3]);
                let (_, in_channels, kernel_height, kernel_width) = (kernel.shape()[0], kernel.shape()[1], kernel.shape()[2], kernel.shape()[3]);

                let mut input_grad = ArrayD::zeros(IxDyn(input_shape));

                for b in 0..batch_size {
                    for ic in 0..in_channels {
                        for ih in 0..input_shape[2] {
                            for iw in 0..input_shape[3] {
                                let mut sum = 0.0;
                                for oc in 0..out_channels {
                                    for kh in 0..kernel_height {
                                        for kw in 0..kernel_width {
                                            if ih >= kh && iw >= kw {
                                                let oh = ih - kh;
                                                let ow = iw - kw;
                                                if oh < out_height && ow < out_width {
                                                    sum += output_grad[[b, oc, oh, ow]] * kernel[[oc, ic, kh, kw]];
                                                }
                                            }
                                        }
                                    }
                                }
                                input_grad[[b, ic, ih, iw]] = sum;
                            }
                        }
                    }
                }

                Ok(Tensor::new_f32(input_grad))
            },
            (TensorData::F64(output_grad), TensorData::F64(kernel)) => {
                let output_grad = output_grad.borrow();
                let kernel = kernel.borrow();
                let (batch_size, out_channels, out_height, out_width) = (output_grad.shape()[0], output_grad.shape()[1], output_grad.shape()[2], output_grad.shape()[3]);
                let (_, in_channels, kernel_height, kernel_width) = (kernel.shape()[0], kernel.shape()[1], kernel.shape()[2], kernel.shape()[3]);

                let mut input_grad = ArrayD::zeros(IxDyn(input_shape));

                for b in 0..batch_size {
                    for ic in 0..in_channels {
                        for ih in 0..input_shape[2] {
                            for iw in 0..input_shape[3] {
                                let mut sum = 0.0;
                                for oc in 0..out_channels {
                                    for kh in 0..kernel_height {
                                        for kw in 0..kernel_width {
                                            if ih >= kh && iw >= kw {
                                                let oh = ih - kh;
                                                let ow = iw - kw;
                                                if oh < out_height && ow < out_width {
                                                    sum += output_grad[[b, oc, oh, ow]] * kernel[[oc, ic, kh, kw]];
                                                }
                                            }
                                        }
                                    }
                                }
                                input_grad[[b, ic, ih, iw]] = sum;
                            }
                        }
                    }
                }

                Ok(Tensor::new_f64(input_grad))
            },
            _ => Err(TensorError::UnsupportedDtype("Conv2D backward requires both tensors to have the same data type (F32 or F64)".to_string())),
        }
    }

    pub fn conv2d_backward_kernel(&self, input: &Tensor, kernel_shape: &[usize]) -> Result<Self, TensorError> {
        match (&self.data, &input.data) {
            (TensorData::F32(output_grad), TensorData::F32(input)) => {
                let output_grad = output_grad.borrow();
                let input = input.borrow();
                let (batch_size, out_channels, out_height, out_width) = (output_grad.shape()[0], output_grad.shape()[1], output_grad.shape()[2], output_grad.shape()[3]);
                let (_, in_channels, _, _) = (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);
                let (_, _, kernel_height, kernel_width) = (kernel_shape[0], kernel_shape[1], kernel_shape[2], kernel_shape[3]);

                let mut kernel_grad = ArrayD::zeros(IxDyn(kernel_shape));

                for oc in 0..out_channels {
                    for ic in 0..in_channels {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let mut sum = 0.0;
                                for b in 0..batch_size {
                                    for oh in 0..out_height {
                                        for ow in 0..out_width {
                                            sum += output_grad[[b, oc, oh, ow]] * input[[b, ic, oh + kh, ow + kw]];
                                        }
                                    }
                                }
                                kernel_grad[[oc, ic, kh, kw]] = sum;
                            }
                        }
                    }
                }

                Ok(Tensor::new_f32(kernel_grad))
            },
            (TensorData::F64(output_grad), TensorData::F64(input)) => {
                let output_grad = output_grad.borrow();
                let input = input.borrow();
                let (batch_size, out_channels, out_height, out_width) = (output_grad.shape()[0], output_grad.shape()[1], output_grad.shape()[2], output_grad.shape()[3]);
                let (_, in_channels, _, _) = (input.shape()[0], input.shape()[1], input.shape()[2], input.shape()[3]);
                let (_, _, kernel_height, kernel_width) = (kernel_shape[0], kernel_shape[1], kernel_shape[2], kernel_shape[3]);

                let mut kernel_grad = ArrayD::zeros(IxDyn(kernel_shape));

                for oc in 0..out_channels {
                    for ic in 0..in_channels {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let mut sum = 0.0;
                                for b in 0..batch_size {
                                    for oh in 0..out_height {
                                        for ow in 0..out_width {
                                            sum += output_grad[[b, oc, oh, ow]] * input[[b, ic, oh + kh, ow + kw]];
                                        }
                                    }
                                }
                                kernel_grad[[oc, ic, kh, kw]] = sum;
                            }
                        }
                    }
                }

                Ok(Tensor::new_f64(kernel_grad))
            },
            _ => Err(TensorError::UnsupportedDtype("Conv2D backward requires both tensors to have the same data type (F32 or F64)".to_string())),
        }
    }
    pub fn max_pool2d(&self, kernel_size: usize, stride: usize, padding: usize) -> Result<Self, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let array_ref = array.borrow();
                let shape = array_ref.shape();
                if shape.len() != 4 {
                    return Err(TensorError::ShapeMismatch("Input tensor must have 4 dimensions (batch_size, channels, height, width)".to_string()));
                }
                let (batch_size, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
                
                let output_height = (height + 2 * padding - kernel_size) / stride + 1;
                let output_width = (width + 2 * padding - kernel_size) / stride + 1;
                
                let mut output = ArrayD::zeros(IxDyn(&[batch_size, channels, output_height, output_width]));
                
                for b in 0..batch_size {
                    for c in 0..channels {
                        for h in 0..output_height {
                            for w in 0..output_width {
                                let h_start = h * stride - padding;
                                let w_start = w * stride - padding;
                                let h_end = (h_start + kernel_size).min(height);
                                let w_end = (w_start + kernel_size).min(width);
                                
                                let mut max_val = f32::NEG_INFINITY;
                                for i in h_start.max(0)..h_end {
                                    for j in w_start.max(0)..w_end {
                                        max_val = max_val.max(array_ref[[b, c, i, j]]);
                                    }
                                }
                                
                                output[[b, c, h, w]] = max_val;
                            }
                        }
                    }
                }
                
                Ok(Tensor::new_f32(output))
            },
            TensorData::F64(array) => {
                let array_ref = array.borrow();
                let shape = array_ref.shape();
                if shape.len() != 4 {
                    return Err(TensorError::ShapeMismatch("Input tensor must have 4 dimensions (batch_size, channels, height, width)".to_string()));
                }
                let (batch_size, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
                
                let output_height = (height + 2 * padding - kernel_size) / stride + 1;
                let output_width = (width + 2 * padding - kernel_size) / stride + 1;
                
                let mut output = ArrayD::zeros(IxDyn(&[batch_size, channels, output_height, output_width]));
                
                for b in 0..batch_size {
                    for c in 0..channels {
                        for h in 0..output_height {
                            for w in 0..output_width {
                                let h_start = h * stride - padding;
                                let w_start = w * stride - padding;
                                let h_end = (h_start + kernel_size).min(height);
                                let w_end = (w_start + kernel_size).min(width);
                                
                                let mut max_val = f64::NEG_INFINITY;
                                for i in h_start.max(0)..h_end {
                                    for j in w_start.max(0)..w_end {
                                        max_val = max_val.max(array_ref[[b, c, i, j]]);
                                    }
                                }
                                
                                output[[b, c, h, w]] = max_val;
                            }
                        }
                    }
                }
                
                Ok(Tensor::new_f64(output))
            },
        }
    }

    pub fn max_pool2d_backward(&self, input: &Tensor, kernel_size: usize, stride: usize, padding: usize) -> Result<Self, TensorError> {
        match (&self.data, &input.data) {
            (TensorData::F32(grad_output), TensorData::F32(input_array)) => {
                let grad_output = grad_output.borrow();
                let input_array = input_array.borrow();
                let [batch_size, channels, input_height, input_width] = input_array.shape().iter().take(4).cloned().collect::<Vec<_>>().try_into().map_err(|_| TensorError::ShapeMismatch("Input tensor must have 4 dimensions (batch_size, channels, height, width)".to_string()))?;
                
                let [_, _, output_height, output_width] = grad_output.shape().iter().take(4).cloned().collect::<Vec<_>>().try_into().map_err(|_| TensorError::ShapeMismatch("Gradient output tensor must have 4 dimensions (batch_size, channels, height, width)".to_string()))?;
                
                let mut grad_input = ArrayD::zeros(input_array.shape());
                
                for b in 0..batch_size {
                    for c in 0..channels {
                        for h in 0..output_height {
                            for w in 0..output_width {
                                let h_start = h * stride - padding;
                                let w_start = w * stride - padding;
                                let h_end = (h_start + kernel_size).min(input_height);
                                let w_end = (w_start + kernel_size).min(input_width);
                                
                                let mut max_val = f32::NEG_INFINITY;
                                let mut max_idx = (0, 0);
                                
                                for i in h_start.max(0)..h_end {
                                    for j in w_start.max(0)..w_end {
                                        let val = input_array[[b, c, i, j]];
                                        if val > max_val {
                                            max_val = val;
                                            max_idx = (i, j);
                                        }
                                    }
                                }
                                
                                grad_input[[b, c, max_idx.0, max_idx.1]] += grad_output[[b, c, h, w]];
                            }
                        }
                    }
                }
                
                Ok(Tensor::new_f32(grad_input))
            },
            (TensorData::F64(grad_output), TensorData::F64(input_array)) => {
                let grad_output = grad_output.borrow();
                let input_array = input_array.borrow();
                let [batch_size, channels, input_height, input_width] = input_array.shape().iter().take(4).cloned().collect::<Vec<_>>().try_into().map_err(|_| TensorError::ShapeMismatch("Input tensor must have 4 dimensions (batch_size, channels, height, width)".to_string()))?;
                
                let [_, _, output_height, output_width] = grad_output.shape().iter().take(4).cloned().collect::<Vec<_>>().try_into().map_err(|_| TensorError::ShapeMismatch("Gradient output tensor must have 4 dimensions (batch_size, channels, height, width)".to_string()))?;
                
                let mut grad_input = ArrayD::zeros(input_array.shape());
                
                for b in 0..batch_size {
                    for c in 0..channels {
                        for h in 0..output_height {
                            for w in 0..output_width {
                                let h_start = h * stride - padding;
                                let w_start = w * stride - padding;
                                let h_end = (h_start + kernel_size).min(input_height);
                                let w_end = (w_start + kernel_size).min(input_width);
                                
                                let mut max_val = f64::NEG_INFINITY;
                                let mut max_idx = (0, 0);
                                
                                for i in h_start.max(0)..h_end {
                                    for j in w_start.max(0)..w_end {
                                        let val = input_array[[b, c, i, j]];
                                        if val > max_val {
                                            max_val = val;
                                            max_idx = (i, j);
                                        }
                                    }
                                }
                                
                                grad_input[[b, c, max_idx.0, max_idx.1]] += grad_output[[b, c, h, w]];
                            }
                        }
                    }
                }
                
                Ok(Tensor::new_f64(grad_input))
            },
            _ => Err(TensorError::UnsupportedDtype("Max pool 2D backward requires both tensors to have the same data type (F32 or F64)".to_string())),
        }
    }
}

impl Tensor {
    pub fn ones(shape: &[usize], dtype: &str) -> Result<Self, TensorError> {
        match dtype {
            "f32" => {
                let ones = ArrayD::from_elem(IxDyn(shape), 1.0f32);
                Ok(Tensor::new_f32(ones))
            },
            "f64" => {
                let ones = ArrayD::from_elem(IxDyn(shape), 1.0f64);
                Ok(Tensor::new_f64(ones))
            },
            _ => Err(TensorError::UnsupportedDtype("Ones requires dtype to be 'f32' or 'f64'".to_string())),
        }
    }

    pub fn zeros(shape: &[usize], dtype: &str) -> Result<Self, TensorError> {
        match dtype {
            "f32" => {
                let zeros = ArrayD::from_elem(IxDyn(shape), 0.0f32);
                Ok(Tensor::new_f32(zeros))
            },
            "f64" => {
                let zeros = ArrayD::from_elem(IxDyn(shape), 0.0f64);
                Ok(Tensor::new_f64(zeros))
            },
            _ => Err(TensorError::UnsupportedDtype("Zeros requires dtype to be 'f32' or 'f64'".to_string())),
        }
    }

    pub fn randn(shape: &[usize], mean: f32, stddev: f32) -> Result<Self, TensorError> {
        let mut rng = rand::thread_rng();
        let dist = rand_distr::Normal::new(mean, stddev).unwrap();
        let tensor = ArrayD::from_shape_fn(IxDyn(shape), |_| dist.sample(&mut rng));
        Ok(Tensor::new_f32(tensor))
    }

    pub fn to_scalar(&self) -> Result<f32, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let array = array.borrow();
                Ok(array[0])
            },
            TensorData::F64(array) => {
                let array = array.borrow();
                Ok(array[0] as f32)
            }
        }
    }
    

    pub fn where_cond(&self, condition: &Tensor, true_case: &Tensor, false_case: &Tensor) -> Result<Self, TensorError> {
        match (&condition.data, &true_case.data, &false_case.data) {
            (TensorData::F32(cond), TensorData::F32(t), TensorData::F32(f)) => {
                let cond = cond.borrow();
                let t = t.borrow();
                let f = f.borrow();
                let mut result = cond.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
                result.zip_mut_with(&t, |c, t_val| *c = if *c > 0.0 { *t_val } else { 0.0 });
                result.zip_mut_with(&f, |c, f_val| *c = if *c > 0.0 { *c } else { *f_val });
                Ok(Tensor::new_f32(result))
            },
            (TensorData::F64(cond), TensorData::F64(t), TensorData::F64(f)) => {
                let cond = cond.borrow();
                let t = t.borrow();
                let f = f.borrow();
                let mut result = cond.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
                result.zip_mut_with(&t, |c, t_val| *c = if *c > 0.0 { *t_val } else { 0.0 });
                result.zip_mut_with(&f, |c, f_val| *c = if *c > 0.0 { *c } else { *f_val });
                Ok(Tensor::new_f64(result))
            },
            _ => Err(TensorError::UnsupportedDtype("Where requires all tensors to have the same data type (F32 or F64)".to_string())),
        }
    }

    pub fn select(&self, condition: &Tensor, other: &Tensor) -> Result<Self, TensorError> {
        match (&self.data, &condition.data, &other.data) {
            (TensorData::F32(a), TensorData::F32(cond), TensorData::F32(b)) => {
                let mut a = a.borrow_mut();
                let cond = cond.borrow();
                let b = b.borrow();
                ndarray::Zip::from(&mut *a)
                    .and(&*cond)
                    .and(&*b)
                    .for_each(|a_val, &cond_val, &b_val| {
                        *a_val = if cond_val > 0.0 { *a_val } else { b_val };
                    });
                Ok(Tensor::new_f32(a.to_owned()))
            },
            (TensorData::F64(a), TensorData::F64(cond), TensorData::F64(b)) => {
                let mut a = a.borrow_mut();
                let cond = cond.borrow();
                let b = b.borrow();
                ndarray::Zip::from(&mut *a)
                    .and(&*cond)
                    .and(&*b)
                    .for_each(|a_val, &cond_val, &b_val| {
                        *a_val = if cond_val > 0.0 { *a_val } else { b_val };
                    });
                Ok(Tensor::new_f64(a.to_owned()))
            },
            _ => Err(TensorError::UnsupportedDtype("Select requires all tensors to have the same data type (F32 or F64)".to_string())),
        }
    }

    pub fn to_array(&self) -> Result<ArrayD<f32>, TensorError> {
        match &self.data {
            TensorData::F32(array) => {
                let array = array.borrow();
                Ok(array.to_owned())
            },
            _ => Err(TensorError::UnsupportedDtype("Tensor to array conversion requires F32 data type".to_string())),
        }
    }

    pub fn is_close(&self, other: &Tensor, epsilon: f32) -> bool {
        match (&self.data, &other.data) {
            (TensorData::F32(a), TensorData::F32(b)) => {
                let a = a.borrow();
                let b = b.borrow();
                a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < epsilon)
            },
            _ => false,
        }
    }
}

impl Add for &Tensor {
    type Output = Tensor;
    
    fn add(self, other: &Tensor) -> Tensor {
        match (&self.data, &other.data) {
            (TensorData::F32(a), TensorData::F32(b)) => Tensor {
                data: TensorData::F32(Rc::new(RefCell::new(&*a.borrow() + &*b.borrow()))),
            },
            (TensorData::F64(a), TensorData::F64(b)) => Tensor {
                data: TensorData::F64(Rc::new(RefCell::new(&*a.borrow() + &*b.borrow()))),
            },
            _ => panic!("Addition requires both tensors to have the same data type."),
        }
    }
}

impl Sub for &Tensor {
    type Output = Tensor;
    
    fn sub(self, other: &Tensor) -> Tensor {
        match (&self.data, &other.data) {
            (TensorData::F32(a), TensorData::F32(b)) => Tensor {
                data: TensorData::F32(Rc::new(RefCell::new(&*a.borrow() - &*b.borrow()))),
            },
            (TensorData::F64(a), TensorData::F64(b)) => Tensor {
                data: TensorData::F64(Rc::new(RefCell::new(&*a.borrow() - &*b.borrow()))),
            },
            _ => panic!("Subtraction requires both tensors to have the same data type."),
        }
    }
}

impl Mul for &Tensor {
    type Output = Tensor;
    
    fn mul(self, other: &Tensor) -> Tensor {
        match (&self.data, &other.data) {
            (TensorData::F32(a), TensorData::F32(b)) => Tensor {
                data: TensorData::F32(Rc::new(RefCell::new(&*a.borrow() * &*b.borrow()))),
            },
            (TensorData::F64(a), TensorData::F64(b)) => Tensor {
                data: TensorData::F64(Rc::new(RefCell::new(&*a.borrow() * &*b.borrow()))),
            },
            _ => panic!("Multiplication requires both tensors to have the same data type."),
        }
    }
}

impl Div for &Tensor {
    type Output = Tensor;
    
    fn div(self, other: &Tensor) -> Tensor {
        match (&self.data, &other.data) {
            (TensorData::F32(a), TensorData::F32(b)) => Tensor {
                data: TensorData::F32(Rc::new(RefCell::new(&*a.borrow() / &*b.borrow()))),
            },
            (TensorData::F64(a), TensorData::F64(b)) => Tensor {
                data: TensorData::F64(Rc::new(RefCell::new(&*a.borrow() / &*b.borrow()))),
            },
            _ => panic!("Division requires both tensors to have the same data type."),
        }
    }
}

impl Neg for &Tensor {
    type Output = Tensor;
    
    fn neg(self) -> Tensor {
        match &self.data {
            TensorData::F32(a) => Tensor {
                data: TensorData::F32(Rc::new(RefCell::new(-&*a.borrow()))),
            },
            TensorData::F64(a) => Tensor {
                data: TensorData::F64(Rc::new(RefCell::new(-&*a.borrow()))),
            },
        }
    }
}

impl AddAssign for Tensor {
    fn add_assign(&mut self, other: Self) {
        match (&mut self.data, &other.data) {
            (TensorData::F32(a), TensorData::F32(b)) => {
                *a.borrow_mut() += &*b.borrow();
            },
            (TensorData::F64(a), TensorData::F64(b)) => {
                *a.borrow_mut() += &*b.borrow();
            },
            _ => panic!("AddAssign requires both tensors to have the same data type."),
        }
    }
}

impl SubAssign for Tensor {
    fn sub_assign(&mut self, other: Self) {
        match (&mut self.data, &other.data) {
            (TensorData::F32(a), TensorData::F32(b)) => {
                *a.borrow_mut() -= &*b.borrow();
            },
            (TensorData::F64(a), TensorData::F64(b)) => {
                *a.borrow_mut() -= &*b.borrow();
            },
            _ => panic!("SubAssign requires both tensors to have the same data type."),
        }
    }
}

impl MulAssign for Tensor {
    fn mul_assign(&mut self, other: Self) {
        match (&mut self.data, &other.data) {
            (TensorData::F32(a), TensorData::F32(b)) => {
                *a.borrow_mut() *= &*b.borrow();
            },
            (TensorData::F64(a), TensorData::F64(b)) => {
                *a.borrow_mut() *= &*b.borrow();
            },
            (TensorData::F64(a), TensorData::F32(b)) => {
                let scalar = b.borrow()[IxDyn(&[])] as f64;
                *a.borrow_mut() *= scalar;
            },
            (TensorData::F32(a), TensorData::F64(b)) => {
                let scalar = b.borrow()[IxDyn(&[])] as f32;
                *a.borrow_mut() *= scalar;
            }
        }
    }
}

impl DivAssign for Tensor {
    fn div_assign(&mut self, other: Self) {
        match (&mut self.data, &other.data) {
            (TensorData::F32(a), TensorData::F32(b)) => {
                *a.borrow_mut() /= &*b.borrow();
            },
            (TensorData::F64(a), TensorData::F64(b)) => {
                *a.borrow_mut() /= &*b.borrow();
            },
            _ => panic!("DivAssign requires both tensors to have the same data type."),
        }
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        match (&self.data, &other.data) {
            (TensorData::F32(a), TensorData::F32(b)) => {
                let a_array = a.borrow();
                let b_array = b.borrow();
                
                if a_array.shape() != b_array.shape() {
                    return false;
                }
                
                a_array.iter().zip(b_array.iter()).all(|(x, y)| (x - y).abs() < std::f32::EPSILON)
            },
            (TensorData::F64(a), TensorData::F64(b)) => {
                let a_array = a.borrow();
                let b_array = b.borrow();
                
                if a_array.shape() != b_array.shape() {
                    return false;
                }
                
                a_array.iter().zip(b_array.iter()).all(|(x, y)| (x - y).abs() < std::f64::EPSILON)
            },
            _ => false,
        }
    }
}

impl Eq for Tensor {}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, arr3};
    use approx::assert_relative_eq;

    // Helper function to create a 2D tensor
    fn create_2d_tensor() -> Tensor {
        Tensor::new_f32(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn())
    }

    // Helper function to create a 3D tensor
    fn create_3d_tensor() -> Tensor {
        Tensor::new_f32(arr3(&[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).into_dyn())
    }

    #[test]
    fn test_new_f32() {
        let tensor = create_2d_tensor();
        assert_eq!(tensor.shape(), vec![2, 2]);
    }

    #[test]
    fn test_new_f64() {
        let tensor = Tensor::new_f64(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());
        assert_eq!(tensor.shape(), vec![2, 2]);
    }

    #[test]
    fn test_scalar() {
        let tensor = Tensor::scalar(5.0);
        assert_eq!(tensor.shape(), vec![]);
        assert_eq!(tensor.to_scalar().unwrap(), 5.0);
    }

    #[test]
    fn test_slice() {
        let tensor = create_3d_tensor();
        let sliced = tensor.slice(&[SliceInfoElem::Index(1), SliceInfoElem::NewAxis, SliceInfoElem::NewAxis]).unwrap();
        assert_eq!(sliced.shape(), vec![2, 2]);
    }

    #[test]
    fn test_clip() {
        let tensor = create_2d_tensor();
        let clipped = tensor.clip(1.5, 3.5).unwrap();
        let expected = arr2(&[[1.5, 2.0], [3.0, 3.5]]);
        assert_relative_eq!(clipped.get_f32().unwrap().borrow().as_slice().unwrap(), expected.as_slice().unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_reshape() {
        let tensor = create_2d_tensor();
        let reshaped = tensor.reshape(&[1, 4]).unwrap();
        assert_eq!(reshaped.shape(), vec![1, 4]);
    }

    #[test]
    fn test_broadcast() {
        let tensor = Tensor::scalar(2.0);
        let broadcasted = tensor.broadcast(&[2, 2]).unwrap();
        assert_eq!(broadcasted.shape(), vec![2, 2]);
    }

    #[test]
    fn test_sum_axis() {
        let tensor = create_2d_tensor();
        let sum = tensor.sum_axis(0).unwrap();
        let expected = arr1(&[4.0, 6.0]);
        assert_relative_eq!(sum.get_f32().unwrap().borrow().as_slice().unwrap(), expected.as_slice().unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_reciprocal() {
        let tensor = create_2d_tensor();
        let reciprocal = tensor.reciprocal().unwrap();
        let expected = arr2(&[[1.0, 0.5], [1.0/3.0, 0.25]]);
        assert_relative_eq!(reciprocal.get_f32().unwrap().borrow().as_slice().unwrap(), expected.as_slice().unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_sign() {
        let tensor = Tensor::new_f32(arr2(&[[-1.0, 0.0], [2.0, -3.0]]).into_dyn());
        let sign = tensor.sign().unwrap();
        let expected = arr2(&[[-1.0, 0.0], [1.0, -1.0]]);
        assert_relative_eq!(sign.get_f32().unwrap().borrow().as_slice().unwrap(), expected.as_slice().unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_abs() {
        let tensor = Tensor::new_f32(arr2(&[[-1.0, 0.0], [2.0, -3.0]]).into_dyn());
        let abs = tensor.abs().unwrap();
        let expected = arr2(&[[1.0, 0.0], [2.0, 3.0]]);
        assert_relative_eq!(abs.get_f32().unwrap().borrow().as_slice().unwrap(), expected.as_slice().unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_pow() {
        let tensor = create_2d_tensor();
        let pow = tensor.pow(2.0).unwrap();
        let expected = arr2(&[[1.0, 4.0], [9.0, 16.0]]);
        assert_relative_eq!(pow.get_f32().unwrap().borrow().as_slice().unwrap(), expected.as_slice().unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_sqrt() {
        let tensor = create_2d_tensor();
        let sqrt = tensor.sqrt().unwrap();
        let expected = arr2(&[[1.0, 2.0_f32.sqrt()], [3.0_f32.sqrt(), 2.0]]);
        assert_relative_eq!(sqrt.get_f32().unwrap().borrow().as_slice().unwrap(), expected.as_slice().unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_exp() {
        let tensor = Tensor::new_f32(arr2(&[[0.0, 1.0], [2.0, 3.0]]).into_dyn());
        let exp = tensor.exp().unwrap();
        let expected = arr2(&[[1.0, 2.718281828], [7.389056099, 20.08553692]]);
        assert_relative_eq!(exp.get_f32().unwrap().borrow().as_slice().unwrap(), expected.as_slice().unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_log() {
        let tensor = create_2d_tensor();
        let log = tensor.log().unwrap();
        let expected = arr2(&[[0.0, 0.693147181], [1.098612289, 1.386294361]]);
        assert_relative_eq!(log.get_f32().unwrap().borrow().as_slice().unwrap(), expected.as_slice().unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_sin_cos_tan() {
        let tensor = Tensor::new_f32(arr1(&[0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI]).into_dyn());
        let sin = tensor.sin().unwrap();
        let cos = tensor.cos().unwrap();
        let tan = tensor.tan().unwrap();
        
        let expected_sin = arr1(&[0.0, 1.0, 0.0]);
        let expected_cos = arr1(&[1.0, 0.0, -1.0]);
        let expected_tan = arr1(&[0.0, f32::INFINITY, 0.0]);
        
        assert_relative_eq!(sin.get_f32().unwrap().borrow().as_slice().unwrap(), expected_sin.as_slice().unwrap(), epsilon = 1e-6);
        assert_relative_eq!(cos.get_f32().unwrap().borrow().as_slice().unwrap(), expected_cos.as_slice().unwrap(), epsilon = 1e-6);
        assert_relative_eq!(tan.get_f32().unwrap().borrow().as_slice().unwrap(), expected_tan.as_slice().unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_mean() {
        let tensor = create_2d_tensor();
        let mean = tensor.mean(None).unwrap();
        assert_eq!(mean.to_scalar().unwrap(), 2.5);

        let mean_axis = tensor.mean(Some(0)).unwrap();
        let expected = arr1(&[2.0, 3.0]);
        assert_relative_eq!(mean_axis.get_f32().unwrap().borrow().as_slice().unwrap(), expected.as_slice().unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_std() {
        let tensor = create_2d_tensor();
        let std = tensor.std(None).unwrap();
        assert_relative_eq!(std.to_scalar().unwrap(), 1.118033989, epsilon = 1e-6);

        let std_axis = tensor.std(Some(0)).unwrap();
        let expected = arr1(&[1.0, 1.0]);
        assert_relative_eq!(std_axis.get_f32().unwrap().borrow().as_slice().unwrap(), expected.as_slice().unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_var() {
        let tensor = create_2d_tensor();
        let var = tensor.var(None).unwrap();
        assert_relative_eq!(var.to_scalar().unwrap(), 1.25, epsilon = 1e-6);

        let var_axis = tensor.var(Some(0)).unwrap();
        let expected = arr1(&[1.0, 1.0]);
        assert_relative_eq!(var_axis.get_f32().unwrap().borrow().as_slice().unwrap(), expected.as_slice().unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_max_min() {
        let tensor = create_2d_tensor();
        let max = tensor.max(None).unwrap();
        let min = tensor.min(None).unwrap();
        assert_eq!(max.to_scalar().unwrap(), 4.0);
        assert_eq!(min.to_scalar().unwrap(), 1.0);

        let max_axis = tensor.max(Some(0)).unwrap();
        let min_axis = tensor.min(Some(0)).unwrap();
        let expected_max = arr1(&[3.0, 4.0]);
        let expected_min = arr1(&[1.0, 2.0]);
        assert_relative_eq!(max_axis.get_f32().unwrap().borrow().as_slice().unwrap(), expected_max.as_slice().unwrap(), epsilon = 1e-6);
        assert_relative_eq!(min_axis.get_f32().unwrap().borrow().as_slice().unwrap(), expected_min.as_slice().unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_argmax_argmin() {
        let tensor = create_2d_tensor();
        let argmax = tensor.argmax(None).unwrap();
        let argmin = tensor.argmin(None).unwrap();
        assert_eq!(argmax.to_scalar().unwrap(), 3.0);
        assert_eq!(argmin.to_scalar().unwrap(), 0.0);

        let argmax_axis = tensor.argmax(Some(0)).unwrap();
        let argmin_axis = tensor.argmin(Some(0)).unwrap();
        let expected_argmax = arr1(&[1.0, 1.0]);
        let expected_argmin = arr1(&[0.0, 0.0]);
        assert_relative_eq!(argmax_axis.get_f32().unwrap().borrow().as_slice().unwrap(), expected_argmax.as_slice().unwrap(), epsilon = 1e-6);
        assert_relative_eq!(argmin_axis.get_f32().unwrap().borrow().as_slice().unwrap(), expected_argmin.as_slice().unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_sum() {
        let tensor = create_2d_tensor();
        let sum = tensor.sum(None).unwrap();
        assert_eq!(sum.to_scalar().unwrap(), 10.0);

        let sum_axis = tensor.sum(Some(0)).unwrap();
        let expected = arr1(&[4.0, 6.0]);
        assert_relative_eq!(sum_axis.get_f32().unwrap().borrow().as_slice().unwrap(), expected.as_slice().unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_dot() {
        let a = Tensor::new_f32(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());
        let b = Tensor::new_f32(arr2(&[[5.0, 6.0], [7.0, 8.0]]).into_dyn());
        let dot = a.dot(&b).unwrap();
        let expected = arr2(&[[19.0, 22.0], [43.0, 50.0]]);
        assert_relative_eq!(dot.get_f32().unwrap().borrow().as_slice().unwrap(), expected.as_slice().unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::new_f32(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());
        let b = Tensor::new_f32(arr2(&[[5.0, 6.0], [7.0, 8.0]]).into_dyn());
        let matmul = a.matmul(&b).unwrap();
        let expected = arr2(&[[19.0, 22.0], [43.0, 50.0]]);
        assert_relative_eq!(matmul.get_f32().unwrap().borrow().as_slice().unwrap(), expected.as_slice().unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_transpose() {
        let tensor = create_2d_tensor();
        let transposed = tensor.transpose().unwrap();
        let expected = arr2(&[[1.0, 3.0], [2.0, 4.0]]);
        assert_relative_eq!(transposed.get_f32().unwrap().borrow().as_slice().unwrap(), expected.as_slice().unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_relu() {
        let tensor = Tensor::new_f32(arr2(&[[-1.0, 0.0], [1.0, 2.0]]).into_dyn());
        let relu = tensor.relu().unwrap();
        let expected = arr2(&[[0.0, 0.0], [1.0, 2.0]]);
        assert_relative_eq!(relu.get_f32().unwrap().borrow().as_slice().unwrap(), expected.as_slice().unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_sigmoid() {
        let tensor = Tensor::new_f32(arr2(&[[-1.0, 0.0], [1.0, 2.0]]).into_dyn());
        let sigmoid = tensor.sigmoid().unwrap();
        let expected = arr2(&[[0.26894142, 0.5], [0.73105858, 0.88079708]]);
        assert_relative_eq!(sigmoid.get_f32().unwrap().borrow().as_slice().unwrap(), expected.as_slice().unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_softmax() {
        let tensor = Tensor::new_f32(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());
        let softmax = tensor.softmax().unwrap();
        let expected = arr2(&[[0.03205860, 0.08714432], [0.23688282, 0.64391426]]);
        assert_relative_eq!(softmax.get_f32().unwrap().borrow().as_slice().unwrap(), expected.as_slice().unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_gt_lt() {
        let a = Tensor::new_f32(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());
        let b = Tensor::new_f32(arr2(&[[2.0, 1.0], [3.0, 5.0]]).into_dyn());
        let gt = a.gt(&b).unwrap();
        let lt = a.lt(&b).unwrap();
        let expected_gt = arr2(&[[0.0, 1.0], [0.0, 0.0]]);
        let expected_lt = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        assert_relative_eq!(gt.get_f32().unwrap().borrow().as_slice().unwrap(), expected_gt.as_slice().unwrap(), epsilon = 1e-6);
        assert_relative_eq!(lt.get_f32().unwrap().borrow().as_slice().unwrap(), expected_lt.as_slice().unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_ones_zeros() {
        let ones = Tensor::ones(&[2, 2], "f32").unwrap();
        let zeros = Tensor::zeros(&[2, 2], "f32").unwrap();
        let expected_ones = arr2(&[[1.0, 1.0], [1.0, 1.0]]);
        let expected_zeros = arr2(&[[0.0, 0.0], [0.0, 0.0]]);
        assert_relative_eq!(ones.get_f32().unwrap().borrow().as_slice().unwrap(), expected_ones.as_slice().unwrap(), epsilon = 1e-6);
        assert_relative_eq!(zeros.get_f32().unwrap().borrow().as_slice().unwrap(), expected_zeros.as_slice().unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_randn() {
        let random = Tensor::randn(&[1000], 0.0, 1.0).unwrap();
        let mean = random.mean(None).unwrap();
        let std = random.std(None).unwrap();
        assert_relative_eq!(mean.to_scalar().unwrap(), 0.0, epsilon = 0.1);
        assert_relative_eq!(std.to_scalar().unwrap(), 1.0, epsilon = 0.1);
    }

   
    #[test]
    fn test_where_cond() {
        let condition_data = vec![1.0, 0.0, 0.0, 1.0];
        let condition_shape = vec![2, 2];
        let condition = Tensor::new_f32(ArrayD::from_shape_vec(IxDyn(&condition_shape), condition_data).unwrap());

        let true_case_data = vec![1.0, 2.0, 3.0, 4.0];
        let true_case_shape = vec![2, 2];
        let true_case = Tensor::new_f32(ArrayD::from_shape_vec(IxDyn(&true_case_shape), true_case_data).unwrap());

        let false_case_data = vec![5.0, 6.0, 7.0, 8.0];
        let false_case_shape = vec![2, 2];
        let false_case = Tensor::new_f32(ArrayD::from_shape_vec(IxDyn(&false_case_shape), false_case_data).unwrap());

        let result = condition.where_cond(&condition, &true_case, &false_case).unwrap();

        let expected_data = vec![1.0, 6.0, 7.0, 4.0];
        let expected_shape = vec![2, 2];
        let expected = ArrayD::from_shape_vec(IxDyn(&expected_shape), expected_data).unwrap();

        assert_relative_eq!(result.get_f32().unwrap().borrow().as_slice().unwrap(), expected.as_slice().unwrap(), epsilon = 1e-6);
    }

    #[test]
    fn test_select() {
        let a = Tensor::new_f32(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());
        let condition = Tensor::new_f32(arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn());
        let b = Tensor::new_f32(arr2(&[[5.0, 6.0], [7.0, 8.0]]).into_dyn());
        let result = a.select(&condition, &b).unwrap();
        let expected = arr2(&[[1.0, 6.0], [7.0, 4.0]]);
        assert_relative_eq!(result.get_f32().unwrap().borrow().as_slice().unwrap(), expected.as_slice().unwrap(), epsilon = 1e-6);
    }
}