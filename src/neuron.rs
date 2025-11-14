use rand::Rng;

#[derive(Debug, Clone)]
pub struct Neuron {
    pub name: Option<String>,
    pub bias: f64,
    pub weights: Box<[f64]>,
    pub err_signal: Option<f64>,
    pub correct_answer: Option<f64>,
    pub temp_output: Option<f64>,
}

impl Neuron {
    pub fn new(inputs: usize) -> Self {
        Self {
            name: None,
            bias: rand::rng().random_range(-0.5..0.5),
            weights: (0..inputs)
                .map(|_| rand::rng().random_range(-0.5..0.5))
                .collect(),
            err_signal: None,
            correct_answer: None,
            temp_output: None,
        }
    }

    pub fn new_with_name(inputs: usize, name: impl ToString) -> Self {
        Self {
            name: Some(name.to_string()),
            bias: rand::rng().random_range(-0.5..0.5),
            weights: (0..inputs)
                .map(|_| rand::rng().random_range(-0.5..0.5))
                .collect(),
            err_signal: None,
            correct_answer: None,
            temp_output: None,
        }
    }

    pub fn apply(&mut self, inputs: &[f64]) -> f64 {
        println!("neuron weights: {:?}, bias: {:?}", self.weights, self.bias);

        let answer = inputs
            .iter()
            .zip(self.weights.iter())
            .map(|(input, weight)| input * weight)
            .sum::<f64>()
            + self.bias;

        self.temp_output = Some(answer);

        answer
    }

    pub fn update_weights(&mut self, input_values: &[f64], learning_rate: &f64) {
        self.bias += learning_rate
            * self
                .err_signal
                .expect("err_signal must be set to update weights");
        self.weights = self
            .weights
            .iter()
            .zip(input_values.iter())
            .map(|(weight, input)| weight + self.err_signal.unwrap() * input * learning_rate)
            .collect();
    }
}
