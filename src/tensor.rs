use ndarray::{Axis, ArrayD, IxDyn, SliceInfo, SliceInfoElem};
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign};
use std::rc::Rc;
use std::cell::RefCell;

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

#[derive(Debug, Clone)]
pub struct Tensor {
    data: TensorData,
}

impl Tensor {
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
        
        let (m, n) = (self.shape()[0], self.shape()[1]);
        
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
            },
            _ => Err(TensorError::UnsupportedDtype("Transpose requires a tensor of type F32 or F64".to_string())),
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
            },
            _ => Err(TensorError::UnsupportedDtype("ReLU requires a tensor of type F32 or F64".to_string())),
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
            _ => Err(TensorError::UnsupportedDtype("Sigmoid requires a tensor of type F32 or F64".to_string())),
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
            _ => Err(TensorError::UnsupportedDtype("Softmax requires a tensor of type F32 or F64".to_string())),
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
            _ => panic!("MulAssign requires both tensors to have the same data type."),
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