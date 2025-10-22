use crate::{neuron::Neuron, series::Series};
use rayon::prelude::*;

pub type Layer = Box<[Neuron]>;

#[derive(Debug, Clone)]
pub struct Network {
    inputs: usize,
    output_layer: Layer,
    hidden_layers: Box<[Layer]>,
}

impl Network {
    pub fn new(
        inputs: usize,
        num_hidden_layers: usize,
        hidden_layer_width: usize,
        output_neurons: usize,
        output_names: &[impl ToString + Copy],
    ) -> Self {
        Self {
            inputs,
            output_layer: (0..output_neurons)
                .map(|i| Neuron::new_with_name(hidden_layer_width, output_names[i]))
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
        let _ = data.iter().map(|series| {
            self.train_on_example(series);
        });
    }

    fn train_on_example(&mut self, series: &Series) {
        let (_output_name, outputs) = self.run(&series.data);
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

        (&self.output_layer[index].name.as_ref().unwrap(), outputs)
    }
}
