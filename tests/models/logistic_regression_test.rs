use creditrisk_api_rs::LogisticRegression;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_new_model() {
        let model = LogisticRegression::new(3, 0.01);

        assert_eq!(model.weights.len(), 3);
        assert_eq!(model.weights, vec![0.0, 0.0, 0.0]);
        assert_eq!(model.bias, 0.0);
        assert_eq!(model.learning_rate, 0.01);
        assert_eq!(model.initialized, false);
    }

    #[test]
    fn test_sigmoid() {
        let model = LogisticRegression::new(1, 0.01);

        assert_relative_eq!(model.sigmoid(0.0), 0.5, epsilon = 1e-10);
        assert_relative_eq!(model.sigmoid(10.0), 0.9999546021312976, epsilon = 1e-10);
        assert_relative_eq!(model.sigmoid(-10.0), 4.53978687024e-5, epsilon = 1e-10);
    }

    #[test]
    fn test_weighted_sum() {
        let mut model = LogisticRegression::new(3, 0.01);
        model.weights = vec![0.5, -0.5, 0.3];
        model.bias = 0.2;

        let features = vec![2.0, 1.0, -1.0];
        let result = model.weighted_sum(&features);

        assert_relative_eq!(result, 0.4, epsilon = 1e-10);
    }

    #[test]
    fn test_predict_raw() {
        let mut model = LogisticRegression::new(2, 0.01);
        model.weights = vec![1.0, -1.0];
        model.bias = 0.0;

        let features = vec![0.5, 0.5];
        let prediction = model.predict_raw(&features);

        assert_relative_eq!(prediction, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_predict() {
        let mut model = LogisticRegression::new(2, 0.01);
        model.weights = vec![2.0, -1.0];
        model.bias = 0.0;

        assert_eq!(model.predict(&[1.0, 0.0]), 1); // sigmoid(2.0) > 0.5
        assert_eq!(model.predict(&[0.0, 2.0]), 0); // sigmoid(-2.0) < 0.5
    }

    #[test]
    fn test_training_xor() {
        // XOR problem
        let x = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let y = vec![0.0, 1.0, 1.0, 0.0];

        // Note: XOR não é linearmente separável, então um modelo de regressão
        // logística simples não conseguirá aprender perfeitamente
        // Este teste é para garantir que o treinamento executa sem erros
        let mut model = LogisticRegression::new(2, 0.1);
        model.train(&x, &y, 100);

        assert!(model.initialized);

        // Verificamos se pelo menos o modelo melhorou em relação à inicialização aleatória
        let predictions = x
            .iter()
            .map(|features| model.predict_raw(features))
            .collect::<Vec<_>>();
        assert_eq!(0.5089674323325066, predictions[0]);
        assert_eq!(0.49881000229313843, predictions[1]);
        assert_eq!(0.486960500675949, predictions[2]);
        assert_eq!(0.4768157864551614, predictions[3]);
    }

    #[test]
    fn test_training_linearly_separable() {
        // Problema linearmente separável: y = 1 se x1 + x2 > 1, senão y = 0
        let x = vec![
            vec![0.0, 0.0],
            vec![0.0, 0.5],
            vec![0.5, 0.0],
            vec![0.5, 0.5],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.5, 1.0],
            vec![1.0, 0.5],
        ];
        let y = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let mut model = LogisticRegression::new(2, 0.1);
        model.train(&x, &y, 1000);

        // Para um problema linearmente separável, esperamos alta precisão
        for (i, features) in x.iter().enumerate() {
            let prediction = model.predict(features);
            assert_eq!(
                prediction as f64, y[i],
                "Falha na classificação do exemplo {}",
                i
            );
        }
    }

    #[test]
    fn test_serialization() {
        let mut model = LogisticRegression::new(2, 0.01);
        model.weights = vec![0.5, -0.3];
        model.bias = 0.2;
        model.initialized = true;

        let serialized = serde_json::to_string(&model).unwrap();
        let deserialized: LogisticRegression = serde_json::from_str(&serialized).unwrap();

        assert_eq!(model.weights, deserialized.weights);
        assert_eq!(model.bias, deserialized.bias);
        assert_eq!(model.learning_rate, deserialized.learning_rate);
        assert_eq!(model.initialized, deserialized.initialized);
    }
}
