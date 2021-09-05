use std::time::{Duration, Instant};

// Bring catboost module into the scope
use catboost;

fn get_float_features(batch_size: u32) -> Vec<Vec<f32>> {
    (0 .. batch_size).map({
        |_| vec![0.33, 0.5]
    }).collect()
}

fn get_cat_features(batch_size: u32) -> Vec<Vec<String>> {
    (0 .. batch_size).map({
        |_| vec![String::from("1"), String::from("2"), String::from("3")]
    }).collect()
}


fn run(batch_size: u32, n_iterations: u32, show_every: u32) {
    // Load the trained model
    println!("Loading model...");

    let model = catboost::Model::load("/home/leo/catboost_memleak/rust/model.cbm").unwrap();

    println!("Number of cat features {}", model.get_cat_features_count());
    println!("Number of float features {}", model.get_float_features_count());

    let mut show = 0u32;
    let mut start = Instant::now();

    for i in 0u32 .. n_iterations {
        let prediction = model
            .calc_model_prediction(
                get_float_features(batch_size),
                get_cat_features(batch_size)
            )
            .unwrap();
        
        if (show % show_every) == 0 {
            let elapsed = start.elapsed();
            println!(
                "iter {}, prediction {:?}, time {:?}, time/call {:?}, time/item: {:?}",
                i, prediction[0], elapsed, elapsed / show_every, elapsed / batch_size / show_every
            );
            start = Instant::now();
        }


        show = show + 1;

    }
}

fn main() {
    // let features = get_float_features(3);
    // println!("{:?}", features)

    run(50, 100000, 1000);
}
