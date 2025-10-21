pub struct Series {
    pub data: Box<[f64]>,
    pub answer: String,
}

impl Series {
    pub fn new(data: impl Into<Box<[f64]>>, answer: impl ToString) -> Self {
        Self {
            data: data.into(),
            answer: answer.to_string(),
        }
    }
}
