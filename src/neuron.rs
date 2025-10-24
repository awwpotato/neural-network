use rand::Rng;

#[derive(Debug, Clone)]
pub struct Neuron {
    pub name: Option<String>,
    pub bias: f64,
    pub weights: Box<[f64]>,
}

impl Neuron {
    pub fn new(inputs: usize) -> Self {
        Self {
            name: None,
            bias: rand::rng().random_range(-0.5..0.5),
            weights: (0..inputs)
                .map(|_| rand::rng().random_range(-0.5..0.5))
                .collect(),
        }
    }

    pub fn new_with_name(inputs: usize, name: impl ToString) -> Self {
        Self {
            name: Some(name.to_string()),
            bias: rand::rng().random_range(-0.5..0.5),
            weights: (0..inputs)
                .map(|_| rand::rng().random_range(-0.5..0.5))
                .collect(),
        }
    }

    pub fn apply(&self, inputs: &[f64]) -> f64 {
        inputs
            .iter()
            .zip(self.weights.iter())
            .map(|(input, weight)| input * weight)
            .sum::<f64>()
            + self.bias
    }

    pub fn update_weights(&mut self, err_signal: &f64, input_values: &[f64], learning_rate: &f64) {
        self.weights = self
            .weights
            .iter()
            .zip(input_values.iter())
            .map(|(weight, input)| weight + err_signal * input * learning_rate)
            .collect();
    }
}
