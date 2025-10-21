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
        output_names: &[impl ToString],
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

    pub fn train(&mut self, data: &[Series]) {
        let _ = data.par_iter().map(|series| {
            let (pred, outputs) = self.run(&series.data);
            (&series.answer, pred, outputs)
        });
    }

    pub fn run(&self, inputs: &[f64]) -> (&str, Vec<f64>) {
        let hidden_layer_outputs: Vec<f64> = self
            .hidden_layers
            .iter()
            .fold(inputs.to_vec(), |inputs, layer| {
                layer.iter().map(|neuron| neuron.apply(&inputs)).collect()
            });

        let outputs: Vec<f64> = self
            .output_layer
            .iter()
            .map(|neuron| neuron.apply(&hidden_layer_outputs))
            .collect();

        let max = outputs
            .iter()
            .max_by(|x, y| x.total_cmp(y))
            .expect("failed to find max output");
        let index = outputs
            .iter()
            .position(|x| x == max)
            .expect("failed to find index of max output");

        (&self.output_names[index], outputs)
    }
}
