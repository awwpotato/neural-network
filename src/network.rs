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

    pub fn train(&mut self, data: &[Series], learning_rating: f64) {
        let _ = data.iter().map(|series| {
            self.train_on_example(series, learning_rating);
        });
    }

    fn train_on_example(&mut self, series: &Series, learning_rating: f64) {
        let (output_name, outputs) = self.run_with_info(&series.data);
        let _ = self
            .output_layer
            .iter_mut()
            .zip(outputs.iter())
            .map(|(neuron, result)| {
                let correct_res = (*neuron.name.as_ref().unwrap() == series.answer) as u8;
                let answer_err_signal = (correct_res as f64 - result) * result * (1.0 - result);
                neuron.weights = neuron
                    .weights
                    .iter()
                    .map(|old_weight| old_weight + answer_err_signal)
                    .collect();

                todo!()
            });
    }

    pub fn run(&self, inputs: &[f64]) -> &str {
        self.run_with_info(inputs).0
    }

    fn run_with_info(&self, inputs: &[f64]) -> (&str, Vec<f64>) {
        let mut hidden_layer_output_tracking: Vec<Vec<f64>> = Vec::new();

        let hidden_layer_outputs: Vec<f64> =
            self.hidden_layers
                .iter()
                .enumerate()
                .fold(inputs.to_vec(), |inputs, (i, layer)| {
                    layer
                        .iter()
                        .enumerate()
                        .map(|(j, neuron)| {
                            let output = neuron.apply(&inputs);
                            hidden_layer_output_tracking[i][j] = output;
                            output
                        })
                        .collect()
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
