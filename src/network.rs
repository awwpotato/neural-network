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
        output_neurons: usize,
        output_names: &[impl ToString + Copy],
    ) -> Self {
        Self {
            output_layer: (0..output_neurons)
                .map(|i| Neuron::new_with_name(hidden_layer_width, Some(output_names[i])))
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

    pub fn train(
        &mut self,
        data: &[Series],
        learning_rate: f64,
        target_err: f64,
        epoch_info_frequency: u32,
    ) {
        let mut epoch_no = 0;
        loop {
            data.iter().for_each(|series| {
                self.train_on_example(series, learning_rate);
            });

            epoch_no += 1;

            let err_percent = self.err_percentage(data);

            if epoch_no % epoch_info_frequency == 0 {
                println!("epoch: {} err_percent: {}", epoch_no, err_percent);
            }

            if err_percent > target_err {
                break;
            }
        }
    }

    pub fn err_percentage(&mut self, inputs: &[Series]) -> f64 {
        inputs
            .iter()
            .map(|series| {
                let (output_name, _outputs) = self.run_with_info(&series.data);
                u8::from(output_name == series.answer) as f64
            })
            .sum::<f64>()
            .div(inputs.len() as f64)
    }

    fn train_on_example(&mut self, series: &Series, learning_rate: f64) {
        let _ = self.run_with_info(&series.data);
        self.cached_inputs = Some(series.data.clone());

        // handle output layer
        self.output_layer.iter_mut().for_each(|neuron| {
            let result = neuron.temp_output.unwrap();

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
        });

        // handle hidden_layers, but leave last hidden layer for special case
        for layer_index in (1..self.hidden_layers.len()).rev() {
            Self::back_propogate(
                self.hidden_layers[layer_index - 1].clone(),
                &mut self.hidden_layers[layer_index].clone(),
                &self.hidden_layers[layer_index - 1]
                    .iter()
                    .map(|n| n.temp_output.unwrap())
                    .collect::<Vec<f64>>(),
                &learning_rate,
            );
        }

        // special case for last hidden layer, because there isn't a layer below
        Self::back_propogate(
            if self.hidden_layers.len() > 1 {
                self.hidden_layers[1].clone()
            } else {
                self.output_layer.clone()
            },
            &mut self.hidden_layers[0],
            self.cached_inputs.as_ref().unwrap(),
            &learning_rate,
        );
    }

    fn back_propogate(
        above_layer: Box<[Neuron]>,
        current_layer: &mut [Neuron],
        inputs: &[f64],
        learning_rate: &f64,
    ) {
        current_layer
            .iter_mut()
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

                neuron.update_weights(inputs, learning_rate);
            });
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
