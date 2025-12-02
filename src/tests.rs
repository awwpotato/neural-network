use crate::{network::Network, series::Series};

#[test]
pub fn test_and() {
    let training_data = vec![
        Series::new([0.0, 0.0], "1"),
        Series::new([1.0, 0.0], "0"),
        Series::new([0.0, 1.0], "0"),
        Series::new([1.0, 1.0], "1"),
    ];

    let mut network = Network::new(2, 1, 4, 2, &["0", "1"]);

    network.train(&training_data, 0.5, 0.9);

    assert_eq!(network.run(&[0.0, 0.0]), "1");
    assert_eq!(network.run(&[1.0, 0.0]), "0");
    assert_eq!(network.run(&[0.0, 1.0]), "0");
    assert_eq!(network.run(&[1.0, 1.0]), "1");
}

#[test]
pub fn test_xor() {
    let training_data = vec![
        Series::new([0.0, 0.0], "0"),
        Series::new([1.0, 0.0], "1"),
        Series::new([0.0, 1.0], "1"),
        Series::new([1.0, 1.0], "0"),
    ];

    let mut network = Network::new(2, 1, 4, 2, &["0", "1"]);

    network.train(&training_data, 0.05, 0.9);

    assert_eq!(network.run(&[0.0, 0.0]), "0");
    assert_eq!(network.run(&[1.0, 0.0]), "1");
    assert_eq!(network.run(&[0.0, 1.0]), "1");
    assert_eq!(network.run(&[1.0, 1.0]), "0");
}
