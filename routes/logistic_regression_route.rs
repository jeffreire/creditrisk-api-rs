use std::sync::Arc;
use axum::{
    extract::State,
    routing::post,
    Json, Router,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Serialize, Deserialize};
use tokio::sync::Mutex;
use crate::models::logistic_regression::LogisticRegression;

/// Erros que podem ocorrer nas operações da API
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("Número de features incompatível: esperado {expected}, recebido {received}")]
    FeatureMismatch { expected: usize, received: usize },
    
    #[error("Requisição inválida: {0}")]
    InvalidRequest(String),

    #[error("Modelo não inicializado: {0}")]
    ModelNotReady(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match &self {
            ApiError::FeatureMismatch { .. } => StatusCode::BAD_REQUEST,
            ApiError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::ModelNotReady(_) => StatusCode::BAD_REQUEST,
        };
        
        (status, self.to_string()).into_response()
    }
}


/// Estruturas para processamento de solicitações

#[derive(Debug, Deserialize)]
pub struct PredictionRequest {
    features: Vec<f64>,
    learning_rate: Option<f64>,
    reconfigure: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct PredictionResponse {
    pub predicted: u8,
    pub confidence: f64,
}

#[derive(Debug, Deserialize)]
pub struct ModelConfigRequest {
    num_features: usize,
    learning_rate: f64,
}

#[derive(Debug, Deserialize)]
pub struct TrainingRequest {
    features: Vec<Vec<f64>>,
    targets: Vec<f64>,
    epochs: usize,
}

#[derive(Debug, Deserialize)]
pub struct SaveModelRequest {
    filepath: String,
}

#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    filepath: String,
}

/// Handlers para os endpoints da API

/// Processa requisições de predição
pub async fn predict(
    State(model): State<Arc<Mutex<LogisticRegression>>>,
    Json(payload): Json<PredictionRequest>,
) -> Result<Json<PredictionResponse>, ApiError> {
    let mut model = model.lock().await;
    
    // Verificar se o modelo foi inicializado
    if !model.initialized {
        return Err(ApiError::ModelNotReady(
            "O modelo não foi treinado ou carregado. Use /train ou /load primeiro.".to_string()
        ));
    }

    // Validação do número de features
    if payload.features.len() != model.weights.len() {
        return Err(ApiError::FeatureMismatch { 
            expected: model.weights.len(), 
            received: payload.features.len() 
        });
    }
    
    // Reconfigura o modelo se solicitado
    if let Some(lr) = payload.learning_rate {
        if payload.reconfigure.unwrap_or(false) {
            *model = LogisticRegression::new(model.weights.len(), lr);
        }
    }
    
    // Realiza a predição
    let raw_prediction = model.predict_raw(&payload.features);
    let prediction = model.predict(&payload.features);
    
    Ok(Json(PredictionResponse { 
        predicted: prediction,
        confidence: raw_prediction,
    }))
}

/// Configura o modelo com novos parâmetros
pub async fn configure_model(
    State(model): State<Arc<Mutex<LogisticRegression>>>,
    Json(config): Json<ModelConfigRequest>,
) -> impl IntoResponse {
    let mut model_lock = model.lock().await;
    *model_lock = LogisticRegression::new(config.num_features, config.learning_rate);
    StatusCode::OK
}

/// Treina o modelo com dados fornecidos
pub async fn train_model(
    State(model): State<Arc<Mutex<LogisticRegression>>>,
    Json(payload): Json<TrainingRequest>,
) -> Result<StatusCode, ApiError> {
    let mut model_lock = model.lock().await;
    
    // Validações
    if payload.features.is_empty() || payload.targets.is_empty() {
        return Err(ApiError::InvalidRequest(
            "Conjuntos de treinamento vazios".to_string()
        ));
    }
    
    if payload.features.len() != payload.targets.len() {
        return Err(ApiError::InvalidRequest(
            format!("Número de amostras incompatível: {} features vs {} targets", 
                payload.features.len(), payload.targets.len())
        ));
    }
    
    // Verifica se cada amostra tem o número correto de features
    for (i, sample) in payload.features.iter().enumerate() {
        if sample.len() != model_lock.weights.len() {
            return Err(ApiError::InvalidRequest(
                format!("Amostra {} tem {} features, esperado {}", 
                    i, sample.len(), model_lock.weights.len())
            ));
        }
    }
    
    model_lock.train(&payload.features, &payload.targets, payload.epochs);
    Ok(StatusCode::OK)
}

/// Salva o modelo em arquivo
pub async fn save_model(
    State(model): State<Arc<Mutex<LogisticRegression>>>,
    Json(payload): Json<SaveModelRequest>,
) -> Result<StatusCode, ApiError> {
    let model_lock = model.lock().await;
    
    // Serializa o modelo para JSON
    let serialized = serde_json::to_string(&*model_lock)
        .map_err(|e| ApiError::InvalidRequest(format!("Erro ao serializar modelo: {}", e)))?;
    
    // Escreve no arquivo
    tokio::fs::write(&payload.filepath, serialized)
        .await
        .map_err(|e| ApiError::InvalidRequest(format!("Erro ao salvar arquivo: {}", e)))?;
    
    Ok(StatusCode::OK)
}

/// Carrega o modelo a partir de um arquivo
pub async fn load_model(
    State(model): State<Arc<Mutex<LogisticRegression>>>,
    Json(payload): Json<LoadModelRequest>,
) -> Result<StatusCode, ApiError> {
    // Lê o arquivo
    let file_content = tokio::fs::read(&payload.filepath)
        .await
        .map_err(|e| ApiError::InvalidRequest(format!("Erro ao ler arquivo: {}", e)))?;
    
    // Deserializa
    let mut loaded_model: LogisticRegression = serde_json::from_slice(&file_content)
        .map_err(|e| ApiError::InvalidRequest(format!("Erro ao deserializar modelo: {}", e)))?;
    
    // Garantir que o modelo esteja marcado como inicializado
    loaded_model.initialized = true;

    // Substitui o modelo atual
    let mut model_lock = model.lock().await;
    *model_lock = loaded_model;
    
    Ok(StatusCode::OK)
}


/// Configura as rotas para este módulo
pub fn routes(model: Arc<Mutex<LogisticRegression>>) -> Router {
    Router::new()
        .route("/predict", post(predict))
		.route("/configure", post(configure_model))
        .route("/train", post(train_model))
        .route("/save-model", post(save_model))
        .route("/load-model", post(load_model))
        .with_state(model)
}