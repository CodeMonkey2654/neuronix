use std::collections::HashMap;
use crate::tensor::Tensor;
use crate::variable::Variable;
use crate::graph::ComputationGraph;

pub trait Optimizer {
    fn step(&mut self, graph: &mut ComputationGraph) -> Result<(), String>;
    fn zero_grad(&mut self);
    fn update(&mut self, param: &Variable, grad: &Tensor) -> Result<(), String>;
}

pub struct SGD {
    learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        SGD { learning_rate }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, graph: &mut ComputationGraph) -> Result<(), String> {
        for var in graph.trainable_variables() {
            if let Some(grad) = var.node.borrow().gradient() {
                self.update(&var, grad)?;
            }
        }
        Ok(())
    }

    fn zero_grad(&mut self) {
        // SGD doesn't need to store gradients, so this is a no-op
    }

    fn update(&mut self, param: &Variable, grad: &Tensor) -> Result<(), String> {
        let value = param.node.borrow().value().ok_or("Parameter value not found")?.clone();
        let updated_value = &value - &(grad * &Tensor::scalar(self.learning_rate));
        param.node.borrow_mut().set_value(updated_value);
        Ok(())
    }
}

pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: usize,
    m: HashMap<usize, Tensor>,
    v: HashMap<usize, Tensor>,
}

impl Adam {
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, graph: &mut ComputationGraph) -> Result<(), String> {
        self.t += 1;
        for var in graph.trainable_variables() {
            if let Some(grad) = var.node.borrow().gradient() {
                self.update(&var, grad)?;
            }
        }
        Ok(())
    }

    fn zero_grad(&mut self) {
        self.m.clear();
        self.v.clear();
        self.t = 0;
    }

    fn update(&mut self, param: &Variable, grad: &Tensor) -> Result<(), String> {
        let var_id = param.node.borrow().id();
        let value = param.node.borrow().value().ok_or("Parameter value not found")?.clone();
        let m = self.m.entry(var_id).or_insert_with(|| Tensor::zeros(&grad.shape(), "f32").unwrap());
        let v = self.v.entry(var_id).or_insert_with(|| Tensor::zeros(&grad.shape(), "f32").unwrap());

        *m = &(grad * &Tensor::scalar(1.0 - self.beta1)) + &(&*m * &Tensor::scalar(self.beta1));
        *v = &(&(grad * grad) * &Tensor::scalar(1.0 - self.beta2)) + &(&*v * &Tensor::scalar(self.beta2));

        let m_hat = &*m / &Tensor::scalar(1.0 - self.beta1.powi(self.t as i32));
        let v_hat = &*v / &Tensor::scalar(1.0 - self.beta2.powi(self.t as i32));

        let sqrt_v_hat = v_hat.sqrt().map_err(|e| e.to_string())?;
        let denom = &sqrt_v_hat + &Tensor::scalar(self.epsilon);
        let step = &(&m_hat / &denom) * &Tensor::scalar(self.learning_rate);
        let updated_value = &value - &step;
        param.node.borrow_mut().set_value(updated_value);
        Ok(())
    }
}

pub struct RMSprop {
    learning_rate: f32,
    alpha: f32,
    epsilon: f32,
    v: HashMap<usize, Tensor>,
}

impl RMSprop {
    pub fn new(learning_rate: f32, alpha: f32, epsilon: f32) -> Self {
        RMSprop {
            learning_rate,
            alpha,
            epsilon,
            v: HashMap::new(),
        }
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self, graph: &mut ComputationGraph) -> Result<(), String> {
        for var in graph.trainable_variables() {
            if let Some(grad) = var.node.borrow().gradient() {
                self.update(&var, grad)?;
            }
        }
        Ok(())
    }

    fn zero_grad(&mut self) {
        self.v.clear();
    }

    fn update(&mut self, param: &Variable, grad: &Tensor) -> Result<(), String> {
        let var_id = param.node.borrow().id();
        let value = param.node.borrow().value().ok_or("Parameter value not found")?.clone();
        let v = self.v.entry(var_id).or_insert_with(|| Tensor::zeros(&value.shape(), "f32").unwrap());

        *v = &(&(grad * grad) * &Tensor::scalar(1.0 - self.alpha)) + &(&*v * &Tensor::scalar(self.alpha));

        let sqrt_v = v.sqrt().map_err(|e| e.to_string())?;
        let denom = &sqrt_v + &Tensor::scalar(self.epsilon);
        let scaled_grad = &(grad / &denom) * &Tensor::scalar(self.learning_rate);
        let updated_value = &value - &scaled_grad;
        param.node.borrow_mut().set_value(updated_value);
        Ok(())
    }
}

pub struct Momentum {
    learning_rate: f32,
    momentum: f32,
    velocity: HashMap<usize, Tensor>,
}

impl Momentum {
    pub fn new(learning_rate: f32, momentum: f32) -> Self {
        Momentum {
            learning_rate,
            momentum,
            velocity: HashMap::new(),
        }
    }
}

impl Optimizer for Momentum {
    fn step(&mut self, graph: &mut ComputationGraph) -> Result<(), String> {
        for var in graph.trainable_variables() {
            if let Some(grad) = var.node.borrow().gradient() {
                self.update(&var, grad)?;
            }
        }
        Ok(())
    }

    fn zero_grad(&mut self) {
        self.velocity.clear();
    }

    fn update(&mut self, param: &Variable, grad: &Tensor) -> Result<(), String> {
        let var_id = param.node.borrow().id();
        let value = param.node.borrow().value().ok_or("Parameter value not found")?.clone();
        let v = self.velocity.entry(var_id).or_insert_with(|| Tensor::zeros(&value.shape(), "f32").unwrap());

        *v = &(grad * &Tensor::scalar(self.learning_rate)) + &(&*v * &Tensor::scalar(self.momentum));

        let updated_value = &value - v;
        param.node.borrow_mut().set_value(updated_value);
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::variable::Variable;
    use crate::graph::ComputationGraph;
    use crate::op::{Identity, Op};

    fn create_test_graph() -> ComputationGraph {
        let mut graph = ComputationGraph::new();
        let var1 = Variable::from_tensor(0, Tensor::new(&[3], &[1.0, 2.0, 3.0], "f32").unwrap(), true);
        let var2 = Variable::from_tensor(1, Tensor::new(&[3], &[4.0, 5.0, 6.0], "f32").unwrap(), true);
        graph.add_node_with_variables(std::rc::Rc::new(Identity), vec![&var1, &var2], vec![var1.clone(), var2.clone()]);
        graph
    }

    fn set_gradients(graph: &mut ComputationGraph) {
        for var in graph.trainable_variables() {
            let grad = Tensor::new(&[3], &[0.1, 0.2, 0.3], "f32").unwrap();
            var.node.borrow_mut().set_gradient(grad);
        }
    }

    #[test]
    fn test_sgd_optimizer() {
        let mut graph = create_test_graph();
        let mut optimizer = SGD::new(0.1);
        
        set_gradients(&mut graph);
        
        optimizer.step(&mut graph).unwrap();
        
        let var1 = &graph.trainable_variables()[0];
        let var2 = &graph.trainable_variables()[1];
        
        let expected1 = Tensor::new(&[3], &[0.99, 1.98, 2.97], "f32").unwrap();
        let expected2 = Tensor::new(&[3], &[3.99, 4.98, 5.97], "f32").unwrap();
        
        assert!(var1.node.borrow().value().unwrap().is_close(&expected1, 1e-6));
        assert!(var2.node.borrow().value().unwrap().is_close(&expected2, 1e-6));
        
        optimizer.zero_grad();
    }

    #[test]
    fn test_adam_optimizer() {
        let mut graph = create_test_graph();
        let mut optimizer = Adam::new(0.1, 0.9, 0.999, 1e-8);
        
        set_gradients(&mut graph);
        
        optimizer.step(&mut graph).unwrap();
        
        let var1 = &graph.trainable_variables()[0];
        let var2 = &graph.trainable_variables()[1];
        
        // These expected values are approximate due to the complexity of Adam
        let expected1 = Tensor::new(&[3], &[0.9, 1.9, 2.9], "f32").unwrap();
        let expected2 = Tensor::new(&[3], &[3.9, 4.9, 5.9], "f32").unwrap();
        
        assert!(var1.node.borrow().value().unwrap().is_close(&expected1, 1e-1));
        assert!(var2.node.borrow().value().unwrap().is_close(&expected2, 1e-1));
        
        optimizer.zero_grad();
        assert_eq!(optimizer.t, 0);
        assert!(optimizer.m.is_empty());
        assert!(optimizer.v.is_empty());
    }

    #[test]
    fn test_rmsprop_optimizer() {
        let mut graph = create_test_graph();
        let mut optimizer = RMSprop::new(0.1, 0.9, 1e-8);
        
        set_gradients(&mut graph);
        
        optimizer.step(&mut graph).unwrap();
        
        let var1 = &graph.trainable_variables()[0];
        let var2 = &graph.trainable_variables()[1];
        
        // These expected values are approximate due to the complexity of RMSprop
        let expected1 = Tensor::new(&[3], &[0.95, 1.95, 2.95], "f32").unwrap();
        let expected2 = Tensor::new(&[3], &[3.95, 4.95, 5.95], "f32").unwrap();
        
        assert!(var1.node.borrow().value().unwrap().is_close(&expected1, 1e-1));
        assert!(var2.node.borrow().value().unwrap().is_close(&expected2, 1e-1));
        
        optimizer.zero_grad();
        assert!(optimizer.v.is_empty());
    }

    #[test]
    fn test_momentum_optimizer() {
        let mut graph = create_test_graph();
        let mut optimizer = Momentum::new(0.1, 0.9);
        
        set_gradients(&mut graph);
        
        optimizer.step(&mut graph).unwrap();
        
        let var1 = &graph.trainable_variables()[0];
        let var2 = &graph.trainable_variables()[1];
        
        let expected1 = Tensor::new(&[3], &[0.99, 1.98, 2.97], "f32").unwrap();
        let expected2 = Tensor::new(&[3], &[3.99, 4.98, 5.97], "f32").unwrap();
        
        assert!(var1.node.borrow().value().unwrap().is_close(&expected1, 1e-6));
        assert!(var2.node.borrow().value().unwrap().is_close(&expected2, 1e-6));
        
        optimizer.zero_grad();
        assert!(optimizer.velocity.is_empty());
    }

    #[test]
    fn test_multiple_steps() {
        let mut graph = create_test_graph();
        let mut optimizer = SGD::new(0.1);
        
        for _ in 0..5 {
            set_gradients(&mut graph);
            optimizer.step(&mut graph).unwrap();
        }
        
        let var1 = &graph.trainable_variables()[0];
        let var2 = &graph.trainable_variables()[1];
        
        let expected1 = Tensor::new(&[3], &[0.95, 1.90, 2.85], "f32").unwrap();
        let expected2 = Tensor::new(&[3], &[3.95, 4.90, 5.85], "f32").unwrap();
        
        assert!(var1.node.borrow().value().unwrap().is_close(&expected1, 1e-6));
        assert!(var2.node.borrow().value().unwrap().is_close(&expected2, 1e-6));
    }

    #[test]
    fn test_optimizer_with_no_gradients() {
        let mut graph = create_test_graph();
        let mut optimizer = SGD::new(0.1);
        
        // Don't set gradients
        
        optimizer.step(&mut graph).unwrap();
        
        let var1 = &graph.trainable_variables()[0];
        let var2 = &graph.trainable_variables()[1];
        
        let expected1 = Tensor::new(&[3], &[1.0, 2.0, 3.0], "f32").unwrap();
        let expected2 = Tensor::new(&[3], &[4.0, 5.0, 6.0], "f32").unwrap();
        
        assert!(var1.node.borrow().value().unwrap().is_close(&expected1, 1e-6));
        assert!(var2.node.borrow().value().unwrap().is_close(&expected2, 1e-6));
    }

    #[test]
    fn test_optimizer_error_handling() {
        let mut graph = create_test_graph();
        let mut optimizer = SGD::new(0.1);
        
        // Create a variable without a value
        let invalid_var = Variable::from_tensor(0, Tensor::zeros(&[3], "f32").unwrap(), true);
        graph.add_node_with_variables(std::rc::Rc::new(Identity), vec![&invalid_var], vec![invalid_var.clone()]);
        
        set_gradients(&mut graph);
        
        // The last variable doesn't have a value, so this should return an error
        assert!(optimizer.step(&mut graph).is_err());
    }

    #[test]
    fn test_adam_convergence() {
        let mut graph = ComputationGraph::new();
        let var = Variable::from_tensor(0, Tensor::new(&[1], &[1.0], "f32").unwrap(), true);
        graph.add_node_with_variables(std::rc::Rc::new(Identity), vec![&var], vec![var.clone()]);
        
        let mut optimizer = Adam::new(0.1, 0.9, 0.999, 1e-8);
        
        for _ in 0..1000 {
            let var = &graph.trainable_variables()[0];
            let value = var.node.borrow().value().unwrap().clone(); // Create a longer-lived value
            let _loss = (&value.clone() - &Tensor::scalar(0.0)).pow(2.0); // Use cloned value
            
            optimizer.step(&mut graph).unwrap();
        }
        
        let expected_value = Tensor::new(&[], &[0.0], "f32").unwrap(); // Store in a variable
        let trainable_variable = graph.trainable_variables()[0].clone();
        let trainable_variable_value = trainable_variable.node.borrow();
        let final_value = trainable_variable_value.value().unwrap().clone();
        let expected_value_clone = expected_value.clone(); // Clone to ensure it lives long enough
        assert!(final_value.is_close(&expected_value_clone, 1e-4)); // Use the cloned variable
    }
}