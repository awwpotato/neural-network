use crate::{neuron::Neuron, series::Series};
use std::ops::Div;

pub type Layer = Box<[Neuron]>;

#[derive(Debug, Clone)]
pub struct Network {
    output_layer: Layer,
    hidden_layers: Box<[Layer]>,
    cached_inputs: Option<Box<[f64]>>,
}

impl Network {
    pub fn new(
        inputs: usize,
        num_hidden_layers: usize,
        hidden_layer_width: usize,
        output_names: &[impl ToString + Copy],
    ) -> Self {
        Self {
            output_layer: (0..output_names.len())
                .map(|i| Neuron::new_with_name(hidden_layer_width, Some(output_names[i])))
                .collect(),
            hidden_layers: (0..num_hidden_layers)
                .map(|i| {
                    (0..hidden_layer_width)
                        .map(|_| Neuron::new(if i == 0 { inputs } else { hidden_layer_width }))
                        .collect()
                })
                .collect(),
            cached_inputs: None,
        }
    }

    pub fn train(
        &mut self,
        data: &[Series],
        learning_rate: f64,
        target_err: f64,
        epoch_info_frequency: u32,
    ) {
        let mut epoch_no = 0;
        let mut exceptable_err_times = 0;
        loop {
            data.iter().for_each(|series| {
                self.train_on_example(series, learning_rate);
            });

            epoch_no += 1;

            let err_percent = self.err_percentage(data);

            if epoch_no % epoch_info_frequency == 0 {
                println!("epoch: {} err_percent: {}", epoch_no, err_percent);
            }

            match err_percent > target_err {
                true => {
                    if exceptable_err_times > 5 {
                        break;
                    } else {
                        exceptable_err_times += 1;
                    }
                }
                false => exceptable_err_times = 0,
            }
        }
    }

    pub fn err_percentage(&mut self, inputs: &[Series]) -> f64 {
        inputs
            .iter()
            .map(|series| {
                let (output_name, _outputs) = self.run_with_info(&series.inputs);
                u8::from(output_name == series.answer) as f64
            })
            .sum::<f64>()
            .div(inputs.len() as f64)
    }

    fn train_on_example(&mut self, series: &Series, learning_rate: f64) {
        let _ = self.run_with_info(&series.inputs);
        self.cached_inputs = Some(series.inputs.clone());

        self.set_err_signals(series);
        self.update_weights(&series.inputs, &learning_rate);
    }

    fn set_err_signals(&mut self, series: &Series) {
        let layer_err_signal = |above_layer: Box<[Neuron]>, layer: &mut Box<[Neuron]>| {
            layer
                .into_iter()
                .enumerate()
                .for_each(|(neuron_index, neuron)| {
                    neuron.err_signal = Some(
                        above_layer
                            .iter()
                            .map(|n| n.err_signal.unwrap() * n.weights[neuron_index])
                            .sum::<f64>()
                            * neuron.temp_output.unwrap()
                            * (1.0 - neuron.temp_output.unwrap()),
                    );
                });
        };

        self.output_layer.iter_mut().for_each(|neuron| {
            let result = neuron.temp_output.unwrap();

            neuron.correct_answer =
                Some(((*neuron.name.as_ref().unwrap() == series.answer) as u8).into());
            neuron.err_signal =
                Some((neuron.correct_answer.unwrap() - result) * result * (1.0 - result));
        });

        // // handle hidden_layers, but leave last hidden layer for special case
        // for layer_index in (1..self.hidden_layers.len()).rev() {
        //     layer_err_signal(
        //         self.hidden_layers[layer_index - 1].clone(),
        //         &mut self.hidden_layers[layer_index],
        //     )
        // }

        layer_err_signal(
            if self.hidden_layers.len() > 1 {
                self.hidden_layers[1].clone()
            } else {
                self.output_layer.clone()
            },
            &mut self.hidden_layers[0],
        );
    }

    fn update_weights(&mut self, inputs: &[f64], learning_rate: &f64) {
        let pre_layer = |layer: &mut Box<[Neuron]>, layer_inputs: &[f64]| {
            layer
                .iter_mut()
                .enumerate()
                .for_each(|(neuron_index, neuron)| {
                    neuron.update_weights(layer_inputs, learning_rate)
                })
        };

        self.hidden_layers[0].iter_mut().for_each(|neuron| {
            neuron.update_weights(inputs, learning_rate);
        });

        // handle hidden_layers, but leave last hidden layer for special case
        for layer_index in 1..self.hidden_layers.len() {
            let below_layer = match layer_index == self.hidden_layers.len() - 1 {
                true => self.output_layer.clone(),
                false => self.hidden_layers[layer_index - 1].clone(),
            };
            let layer_inputs = below_layer
                .iter()
                .map(|n| n.temp_output.unwrap())
                .collect::<Vec<f64>>();

            pre_layer(&mut self.hidden_layers[layer_index], &layer_inputs);
        }

        self.output_layer.iter_mut().for_each(|neuron| {
            neuron.update_weights(
                &self.hidden_layers[self.hidden_layers.len() - 1]
                    .iter()
                    .map(|n| n.temp_output.unwrap())
                    .collect::<Vec<f64>>(),
                learning_rate,
            );
        });
    }

    /// return the name of the output the neural network rated highest
    pub fn run(&mut self, inputs: &[f64]) -> &str {
        self.run_with_info(inputs).0
    }

    /// return the name of the output the neural network rated highest and
    /// all of the output's ratings
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
