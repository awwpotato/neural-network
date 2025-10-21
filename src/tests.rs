use crate::{network::Network, series::Series};

#[test]
pub fn test_xor() {
    let training_data = vec![
        Series::new([0.0, 0.0], "0"),
        Series::new([1.0, 0.0], "1"),
        Series::new([0.0, 1.0], "1"),
        Series::new([1.0, 1.0], "0"),
    ];

    let mut network = Network::new(2, 1, 4, 2, ["0".into(), "1".into()]);

    network.train(training_data);

    assert_eq!(network.run(&[0.0, 0.0]), "0");
    assert_eq!(network.run(&[1.0, 0.0]), "1");
    assert_eq!(network.run(&[0.0, 1.0]), "1");
    assert_eq!(network.run(&[1.0, 1.0]), "0");
}
