use crate::{neuron::Neuron, series::Series};
use rayon::prelude::*;

pub type Layer = Box<[Neuron]>;

#[derive(Debug, Clone)]
pub struct Network {
    inputs: usize,
    output_layer: Layer,
    hidden_layers: Box<[Layer]>,
    cached_inputs: Option<Box<[f64]>>,
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
            cached_inputs: None,
        }
    }

    pub fn train(&mut self, data: &[Series], learning_rating: f64, target_err_percent: f64) {
        loop {
            data.iter().for_each(|series| {
                self.train_on_example(series, learning_rating);
            });

            if self.err_percentage(data) > target_err_percent {
                break;
            }
        }
    }

    pub fn err_percentage(&mut self, data: &[Series]) -> f64 {
        let mut signals = Vec::new();

        for series in data {
            let (output_name, _outputs) = self.run_with_info(&series.data);
            signals.push(((output_name == series.answer) as u8) as f64);
        }

        signals.iter().sum::<f64>() / signals.len() as f64
    }

    fn train_on_example(&mut self, series: &Series, learning_rate: f64) {
        let (output_name, outputs) = self.run_with_info(&series.data);
        self.cached_inputs = Some(series.data.clone());

        let mut temp_err_signal: Vec<f64> = self
            .output_layer
            .iter_mut()
            .zip(outputs)
            .map(|(neuron, result)| {
                neuron.correct_answer =
                    Some(((*neuron.name.as_ref().unwrap() == series.answer) as u8).into());
                neuron.err_signal =
                    Some((neuron.correct_answer.unwrap() - result) * result * (1.0 - result));

                neuron.update_weights(
                    &self.hidden_layers[self.hidden_layers.len() - 1]
                        .iter()
                        .map(|n| n.temp_output.unwrap())
                        .collect::<Vec<f64>>(),
                    &learning_rate,
                );

                neuron.err_signal.unwrap()
            })
            .collect();

        println!("temp err signal {:?}", temp_err_signal);

        // let mut temp_hidden_layer_err_signal: Vec<f64> = {
        //     let temp_data: Vec<(f64, Vec<f64>)> = self.hidden_layers[self.hidden_layers.len() - 1]
        //         .iter()
        //         .enumerate()
        //         .map(|(i, neuron)| {
        //             let err_signal = output_layer_err_signals
        //                 .iter()
        //                 .zip(self.output_layer.iter())
        //                 .map(|(err_signal, neuron)| err_signal * neuron.weights[i])
        //                 .sum::<f64>()
        //                 * neuron.temp_output.unwrap()
        //                 * (1.0 - neuron.temp_output.unwrap());
        //
        //             let inputs_values = self.hidden_layers[self.hidden_layers.len() - 2]
        //                 .iter()
        //                 .map(|n| n.temp_output.unwrap())
        //                 .collect::<Vec<f64>>();
        //
        //             (err_signal, inputs_values)
        //         })
        //         .collect();
        //
        //     self.hidden_layers[self.hidden_layers.len() - 1]
        //         .iter_mut()
        //         .zip(temp_data)
        //         .map(|(neuron, (err_signal, input_values))| {
        //             neuron.err_signal = Some(err_signal);
        //             neuron.update_weights(&input_values, &learning_rate);
        //
        //             err_signal
        //         })
        //         .collect()
        // };

        for layer_index in (1..(self.hidden_layers.len() - 1)).rev() {
            let temp: Vec<(f64, Vec<f64>)> = self.hidden_layers[layer_index]
                .iter()
                .enumerate()
                .map(|(neuron_index, neuron)| {
                    let err_signal = temp_err_signal
                        .iter()
                        .zip(self.hidden_layers[layer_index + 1].iter())
                        .map(|(err_signal, n)| err_signal * n.weights[neuron_index])
                        .sum::<f64>()
                        * neuron.temp_output.unwrap()
                        * (1.0 - neuron.temp_output.unwrap());

                    println!("err_signal: {:?}", err_signal);

                    (
                        err_signal,
                        self.hidden_layers[layer_index - 1]
                            .iter()
                            .map(|n| n.temp_output.unwrap())
                            .collect::<Vec<f64>>(),
                    )
                })
                .collect();
            temp_err_signal = self.hidden_layers[layer_index]
                .iter_mut()
                .zip(temp)
                .map(|(neuron, (err_signal, input_values))| {
                    neuron.err_signal = Some(err_signal);

                    neuron.update_weights(&input_values, &learning_rate);

                    err_signal
                })
                .collect();
        }

        {
            let temp: Vec<f64> = self.hidden_layers[0]
                .iter()
                .enumerate()
                .map(|(neuron_index, neuron)| {
                    temp_err_signal
                        .iter()
                        .zip(if self.hidden_layers.len() > 1 {
                            self.hidden_layers[1].iter()
                        } else {
                            self.output_layer.iter()
                        })
                        .map(|(err_signal, n)| err_signal * n.weights[neuron_index])
                        .sum::<f64>()
                        * neuron.temp_output.unwrap()
                        * (1.0 - neuron.temp_output.unwrap())
                })
                .collect();

            self.hidden_layers[0]
                .iter_mut()
                .zip(temp)
                .for_each(|(neuron, err_signal)| {
                    neuron.err_signal = Some(err_signal);
                    neuron.update_weights(self.cached_inputs.as_ref().unwrap(), &learning_rate);
                });
        }
    }

    pub fn run(&mut self, inputs: &[f64]) -> &str {
        self.run_with_info(inputs).0
    }

    fn run_with_info(&mut self, inputs: &[f64]) -> (&str, Vec<f64>) {
        let hidden_layer_outputs: Vec<f64> =
            self.hidden_layers
                .iter_mut()
                .fold(inputs.to_vec(), |inputs, layer| {
                    layer
                        .iter_mut()
                        .map(|neuron| neuron.apply(&inputs))
                        .collect()
                });

        let outputs: Vec<f64> = self
            .output_layer
            .iter_mut()
            .map(|neuron| neuron.apply(&hidden_layer_outputs))
            .collect();

        assert!(outputs.iter().all(|x| !x.is_nan()));

        let max = outputs
            .iter()
            .max_by(|x, y| x.total_cmp(y))
            .expect("failed to find max output");
        let index = outputs
            .iter()
            .position(|x| x == max)
            .expect("failed to find index of max output");

        (self.output_layer[index].name.as_ref().unwrap(), outputs)
    }
}
