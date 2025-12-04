pub struct Series {
    pub inputs: Box<[f64]>,
    pub answer: String,
}

impl Series {
    pub fn new(inputs: impl Into<Box<[f64]>>, answer: impl ToString) -> Self {
        Self {
            inputs: inputs.into(),
            answer: answer.to_string(),
        }
    }
}
