use ndarray::prelude::*;

/// Creates a padding mask for sequences.
/// `sequence`: Array of shape (batch_size, seq_len)
/// `pad_token`: The token used for padding in the sequences.
pub fn create_padding_mask(sequence: &Array2<usize>, pad_token: usize) -> Array2<f64> {
    sequence.mapv(|token| if token == pad_token { 1.0 } else { 0.0 })
}

/// Creates a look-ahead mask for sequences to prevent the decoder from attending to future positions.
/// `size`: The size of the mask (usually equal to the target sequence length).
pub fn create_look_ahead_mask(size: usize) -> Array2<f64> {
    let mut mask = Array2::<f64>::zeros((size, size));
    for i in 0..size {
        for j in 0..size {
            if j > i {
                mask[[i, j]] = 1.0;
            }
        }
    }
    mask
}

/// Combines padding and look-ahead masks into a single mask.
/// Positive values in the combined mask indicate positions to be masked.
pub fn combine_masks(padding_mask: &ArrayD<f64>, look_ahead_mask: &ArrayD<f64>) -> ArrayD<f64> {
    padding_mask + look_ahead_mask
}
