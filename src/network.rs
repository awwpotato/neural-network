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

    fn train_on_example(&mut self, series: &Series, learning_rate: f64) {
        let (output_name, outputs) = self.run_with_info(&series.data);

        let output_layer_error_signals: Vec<f64> = self
            .output_layer
            .iter_mut()
            .zip(outputs)
            .map(|(neuron, result)| {
                neuron.correct_answer =
                    Some(((*neuron.name.as_ref().unwrap() == series.answer) as u8).into());
                neuron.err_signal = Some((correct_res as f64 - result) * result * (1.0 - result));

                neuron.update_weights(
                    &self.hidden_layers[self.hidden_layers.len() - 1]
                        .iter()
                        .map(|n| n.temp_output.unwrap())
                        .collect::<Vec<f64>>(),
                    &learning_rate,
                );

                answer_err_signal
            })
            .collect();

        let _ = self.hidden_layers.iter_mut().fold(
            output_layer_error_signals,
            |error_signals, layer| {
                let _ = layer.iter_mut().map(|neuron| {});
                todo!()
            },
        );
    }

    pub fn run(&mut self, inputs: &[f64]) -> &str {
        self.run_with_info(inputs).0
    }

    fn run_with_info(&mut self, inputs: &[f64]) -> (&str, Vec<f64>) {
        let hidden_layer_outputs: Vec<f64> = self.hidden_layers.iter_mut().enumerate().fold(
            inputs.to_vec(),
            |inputs, (i, layer)| {
                layer
                    .iter_mut()
                    .enumerate()
                    .map(|(j, neuron)| neuron.apply(&inputs))
                    .collect()
            },
        );

        let outputs: Vec<f64> = self
            .output_layer
            .iter_mut()
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
