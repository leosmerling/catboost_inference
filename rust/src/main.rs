use std::time::Instant;

use simple_process_stats::ProcessStats;
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


async fn run(batch_size: u32, n_iterations: u32, show_every: u32) {
    // Load the trained model
    println!("Loading model...");

    let model = catboost::Model::load("/home/leo/catboost_memleak/rust/model.cbm").unwrap();

    println!("Number of cat features {}", model.get_cat_features_count());
    println!("Number of float features {}", model.get_float_features_count());

    let mut start = Instant::now();
    let mut maxmem= 0u64;

    for i in 0u32 .. n_iterations + 1 {
        let prediction = model
            .calc_model_prediction(
                get_float_features(batch_size),
                get_cat_features(batch_size)
            )
            .unwrap();
        
        if i > 0 && (i % show_every) == 0 {
            let elapsed = start.elapsed();
            let process_stats = ProcessStats::get().await.expect("failed stats");
            maxmem = process_stats.memory_usage_bytes.max(maxmem);
            println!(
                "iter {}, prediction {:?}, elapsed {:?}, time/call {:?}, time/item: {:?}, mem: {}kb, maxmem: {}kb",
                i, prediction[0], elapsed, elapsed / show_every, elapsed / batch_size / show_every, process_stats.memory_usage_bytes / 1024, maxmem / 1024
            );
            start = Instant::now();
        }
    }
}

#[tokio::main]
async fn main() {
    run(200, 100000, 1000).await;
}
