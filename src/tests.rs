use crate::{network::Network, series::Series};
use std::fs::read_to_string;

#[test]
pub fn test_and() {
    let training_data = vec![
        Series::new([0.0, 0.0], "0"),
        Series::new([1.0, 0.0], "0"),
        Series::new([0.0, 1.0], "0"),
        Series::new([1.0, 1.0], "1"),
    ];

    let mut network = Network::new(2, 1, 2, &["0", "1"]);

    network.train(&training_data, 1.0, 0.9, 1);
}

#[test]
pub fn test_xor() {
    let training_data = vec![
        Series::new([0.0, 0.0], "0"),
        Series::new([1.0, 0.0], "1"),
        Series::new([0.0, 1.0], "1"),
        Series::new([1.0, 1.0], "0"),
    ];

    let mut network = Network::new(2, 1, 2, &["0", "1"]);

    network.train(&training_data, 0.05, 0.9, 1000);
}

#[test]
pub fn test_digits() {
    let training_data: Vec<Series> = {
        let file = read_to_string("./data/digits-train.txt").unwrap();
        file.lines()
            .map(|line_str| {
                let mut line: Vec<&str> = line_str.split(",").collect();
                let correct = line.pop().unwrap();
                let inputs: Vec<f64> = line
                    .iter()
                    .map(|input| f64::from(input.parse::<u8>().unwrap()))
                    .collect();
                Series::new(Box::from(inputs), correct)
            })
            .collect()
    };

    println!("file loaded");

    let output_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
    let mut network = Network::new(64, 1, 100, &output_names);

    println!("network built");

    network.train(&training_data, 0.05, 0.99, 1);
}
