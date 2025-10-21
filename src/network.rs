use crate::{neuron::Neuron, series::Series};
use rayon::prelude::*;

pub type Layer = Box<[Neuron]>;

#[derive(Debug, Clone)]
pub struct Network {
    inputs: usize,
    output_layer: Layer,
    output_names: Box<[String]>,
    hidden_layers: Box<[Layer]>,
}

impl Network {
    pub fn new(
        inputs: usize,
        num_hidden_layers: usize,
        hidden_layer_width: usize,
        output_neurons: usize,
        output_names: Box<[impl ToString]>,
    ) -> Self {
        Self {
            inputs,
            output_names: output_names.iter().map(|s| s.to_string()).collect(),
            output_layer: (0..output_neurons)
                .map(|_| Neuron::new(hidden_layer_width))
                .collect(),
            hidden_layers: (0..num_hidden_layers)
                .map(|_| {
                    (0..hidden_layer_width)
                        .map(|i| Neuron::new(if i == 0 { inputs } else { hidden_layer_width }))
                        .collect()
                })
                .collect(),
        }
    }

    fn internal_run(&self, inputs: &[f64]) -> &[f64] {
        todo!()
    }

    fn outputs_to_output(&self, outputs: &[f64]) -> &str {
        let max = outputs.iter().max_by(|x, y| x.total_cmp(y)).unwrap();
        let index = outputs.iter().position(|i| i == max).unwrap();
        &self.output_names[index]
    }

    pub fn train(&mut self, data: &[Series]) {
        let _ = data.par_iter().map(|series| {
            let outputs = self.internal_run(&series.data);
            (&series.answer, self.outputs_to_output(outputs), outputs)
        });
    }

    pub fn run(&self, inputs: &[f64]) -> &str {
        self.outputs_to_output(self.internal_run(inputs))
    }
}
