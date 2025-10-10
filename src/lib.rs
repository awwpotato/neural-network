use rand::Rng;

type Layer = Vec<Neuron>;

#[derive(Debug, Clone)]
struct Network {
    inputs: usize,
    output_layer: Layer,
    hidden_layers: Vec<Layer>,
}

impl Network {
    fn new(
        inputs: usize,
        num_hidden_layers: usize,
        hidden_layer_width: usize,
        output_neurons: usize,
    ) -> Self {
        Self {
            inputs,
            output_layer: (0..output_neurons)
                .map(|_| Neuron::new(hidden_layer_width))
                .collect(),
            hidden_layers: (0..num_hidden_layers)
                .map(|_| {
                    (0..size).map(|i| Neuron::new(if i == 0 { inputs } else { hidden_layer_width }))
                })
                .collect(),
        }
    }

    fn train(&mut self) {
        todo!()
    }

    fn run(&mut self) {
        todo!()
    }
}

#[derive(Debug, Clone)]
struct Neuron {
    bias: f64,
    weights: Vec<f64>,
}

impl Neuron {
    fn new(inputs: usize) -> Self {
        Self {
            bias: rand::rng().gen_range(-0.5..0.5),
            weights: (0..inputs)
                .map(|_| rand::rng().gen_range(-0.5..0.5))
                .collect(),
        }
    }

    fn apply(self, inputs: &[f64]) -> f64 {
        inputs
            .iter()
            .zip(self.weights.iter())
            .map(|(input, weight)| input * weight)
            .sum::<f64>()
            + self.bias
    }
}
