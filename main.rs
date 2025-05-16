mod routes;
mod models;

use axum::{Router, body::Body};
use tokio::sync::Mutex;
use std::{net::SocketAddr, sync::Arc};
use tower_http::trace::TraceLayer;
use crate::models::logistic_regression::LogisticRegression;

// use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() {
    // tracing_subscriber::registry()
    //     .with(tracing_subscriber::fmt::layer())
    //     .init();
    
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let model = Arc::new(Mutex::new(LogisticRegression::new( 3, 0.01)));

    let app = Router::new()
        .merge(routes::logistic_regression_route::routes(model))
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(|req: &axum::http::Request<Body>| {
                    tracing::info_span!(
                        "request",
                        method = %req.method(),
                        uri = %req.uri(),
                    )
                })
                .on_response(|res: &axum::http::Response<Body>, latency: std::time::Duration, _span: &tracing::Span| {
                    tracing::info!(
                        status = %res.status(),
                        latency = ?latency,
                        "resposta enviada"
                    );
                }),
        );

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    tracing::info!("Servidor rodando em http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}