pub mod tensor;
pub mod variable;
pub mod node;
pub mod op;
pub mod layer;
pub mod optimizer;
pub mod graph;

// Re-export commonly used items
pub use tensor::Tensor;
pub use variable::Variable;
pub use node::Node;
pub use op::Op;
pub use layer::Layer;
pub use optimizer::Optimizer;
pub use graph::ComputationGraph;

// Optional: Re-export specific implementations if they are frequently used
pub use layer::{Dense, Sequential};
pub use optimizer::SGD;
pub use op::{Add, MatMul, ReLU, Conv2D, MaxPool2D, Flatten, Softmax, MeanSquaredError};
