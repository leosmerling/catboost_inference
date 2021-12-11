use std::time::Instant;

use crate::cb::{PredictRequest, PredictResponse, Features, Prediction};

thread_local! {
    pub static MODEL: catboost::Model = load_model();
}

fn load_model() -> catboost::Model {
    println!("Loading model...");
    catboost::Model::load("model.cbm").unwrap()
}

pub fn preprocess(features: &Vec<Features>) -> (Vec<Vec<String>>, Vec<Vec<f32>>) {
    let cat_features = features.iter().map(
        |f| vec![f.cat_feature1.clone(), f.cat_feature2.clone(), f.cat_feature3.clone()]
    ).collect();

    let float_features = features.iter().map(
        |f| vec![f.float_feature4, f.float_feature5]
    ).collect();

    (cat_features, float_features)
}

pub fn predict(request: PredictRequest) -> PredictResponse {
    
    let start = Instant::now();
    let (cat_features, float_features ) = preprocess(&request.features);
    let preprocess_latency = start.elapsed().as_nanos() as u64;

    let start = Instant::now();
    let pred = MODEL.with( |model| 
            model.calc_model_prediction(float_features, cat_features).unwrap()
    );
    let model_latency = start.elapsed().as_nanos() as u64;
    
    PredictResponse {
        predictions: pred.iter().map(|score| Prediction { score:  *score as f32 }).collect(),
        preprocess_latency,
        model_latency,
    }
}
