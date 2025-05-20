# Stage 1: Builder
FROM rustlang/rust:nightly as builder

WORKDIR /usr/src/app

# Instalar dependências de compilação
RUN apt-get update && \
    apt-get install -y pkg-config libssl-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Criar projeto
RUN cargo new --lib creditrisk-api-rs
WORKDIR /usr/src/app/creditrisk-api-rs
COPY Cargo.toml Cargo.lock ./

# Criar uma estrutura básica de arquivos
RUN mkdir -p src/models src/routes

# Copiar código fonte real
COPY src ./src
COPY main.rs ./

# Garantir que lib.rs está correto
RUN echo 'pub mod models;\npub mod routes;\npub use models::logistic_regression::LogisticRegression;' > src/lib.rs

# Compilar aplicação
RUN cargo build --release

# Listar binários gerados para debug
RUN ls -la /usr/src/app/creditrisk-api-rs/target/release/

# Stage 2: Runtime
FROM ubuntu:22.04

WORKDIR /app

# Instalar dependências necessárias
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Copiar binário compilado - usando uma expressão mais ampla para capturar o binário correto
COPY --from=builder /usr/src/app/creditrisk-api-rs/target/release/creditrisk-api-rs ./creditrisk-api-rs

# Criar diretório para modelos
RUN mkdir -p /app/models

# Configurar usuário não-root
RUN groupadd -r appuser && \
    useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app

USER appuser
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

CMD ["./creditrisk-api-rs"]