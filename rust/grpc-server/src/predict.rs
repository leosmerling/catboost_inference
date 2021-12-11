use crate::cb::{PredictRequest, PredictResponse};

thread_local! {
    pub static MODEL: catboost::Model = load_model();
}

fn load_model() -> catboost::Model {
    println!("Loading model...");
    catboost::Model::load("model.cbm").unwrap()
}

pub fn predict(request: PredictRequest) -> PredictResponse {
    PredictResponse { 
        score: MODEL.with( |model|
            model.calc_model_prediction(
                vec![vec![request.float_feature4, request.float_feature5]],
                vec![vec![request.cat_feature1, request.cat_feature2, request.cat_feature3]]
            ).unwrap()[0] as f32
        )
    }
}
