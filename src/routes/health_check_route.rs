use axum::{Json, Router, response::IntoResponse, routing::get};
use serde::Serialize;

pub async fn health_check() -> impl IntoResponse {
    #[derive(Serialize)]
    struct HealthResponse {
        status: &'static str,
        version: &'static str,
        timestamp: String,
    }

    let response = HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    Json(response)
}

pub fn routes() -> Router {
    Router::new().route("/health", get(health_check))
}
