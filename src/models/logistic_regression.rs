use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct LogisticRegression {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub learning_rate: f64,
    #[serde(default)]
    pub initialized: bool,
}

impl LogisticRegression {
    pub fn new(num_features: usize, learning_rate: f64) -> Self {
        let weights = vec![0.0; num_features];
        LogisticRegression {
            weights,
            bias: 0.0, // Inicializado com zero
            learning_rate,
            initialized: false,
        }
    }

    pub fn train(&mut self, x: &[Vec<f64>], y: &[f64], epochs: usize) {
        for _ in 0..epochs {
            for (features, &target) in x.iter().zip(y.iter()) {
                let prediction = self.sigmoid(self.weighted_sum(features));
                let error = prediction - target;

                // Atualiza os pesos
                for (weight, &feature) in self.weights.iter_mut().zip(features.iter()) {
                    *weight -= self.learning_rate * error * feature;
                }

                // Atualiza o bias
                self.bias -= self.learning_rate * error;
            }
        }
        self.initialized = true;
    }

    pub fn predict_raw(&self, features: &[f64]) -> f64 {
        self.sigmoid(self.weighted_sum(features))
    }

    pub fn predict(&self, features: &[f64]) -> u8 {
        if self.predict_raw(features) > 0.5 {
            1
        } else {
            0
        }
    }

    pub fn weighted_sum(&self, features: &[f64]) -> f64 {
        self.weights
            .iter()
            .zip(features)
            .map(|(w, xi)| w * xi)
            .sum::<f64>()
            + self.bias // Adicionando o bias ao somatório ponderado
    }

    pub fn sigmoid(&self, z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }
}
