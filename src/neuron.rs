use rand::Rng;

#[derive(Debug, Clone)]
pub struct Neuron {
    bias: f64,
    weights: Box<[f64]>,
}

impl Neuron {
    pub fn new(inputs: usize) -> Self {
        Self {
            bias: rand::rng().random_range(-0.5..0.5),
            weights: (0..inputs)
                .map(|_| rand::rng().random_range(-0.5..0.5))
                .collect(),
        }
    }

    pub fn apply(self, inputs: &[f64]) -> f64 {
        inputs
            .iter()
            .zip(self.weights.iter())
            .map(|(input, weight)| input * weight)
            .sum::<f64>()
            + self.bias
    }
}
