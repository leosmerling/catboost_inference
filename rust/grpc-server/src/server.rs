use std::thread_local;

use tonic::{transport::Server, Request, Response, Status};
use catboost;

pub mod cb {
    tonic::include_proto!("cb"); // The string specified here must match the proto package name
}

mod predict;
use predict::predict;

#[derive(Debug, Default)]
pub struct CatboostInferenceService {}

#[tonic::async_trait]
impl cb::inference_server::Inference for CatboostInferenceService {
    async fn predict(
        &self,
        request: Request<cb::PredictRequest>,
    ) -> Result<Response<cb::PredictResponse>, Status> {
        // println!("Got a request: {:?}", request);

        let reply = predict(request.into_inner());

        // println!("Reply with: {:?}", reply);
        Ok(Response::new(reply)) // Send back our formatted greeting
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse()?;
    let service = CatboostInferenceService::default();

    Server::builder()
        .add_service(cb::inference_server::InferenceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
