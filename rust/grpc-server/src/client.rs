use std::time::{Duration, Instant};

use tonic::{Request, Response, Status};

pub mod cb {
    tonic::include_proto!("cb"); // The string specified here must match the proto package name
}

const REPORT: usize = 1000;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================================================================================================================================");
    println!("Usage: cb-client http://host:port batch_size iterations timeout_ms");
    let host = std::env::args().nth(1).expect("http://host:port missing");
    let b: usize = std::env::args().nth(2).expect("missing number batch_size").parse().unwrap();
    let n: usize = std::env::args().nth(3).expect("missing number iterations").parse().unwrap();
    let timeout: u64 = std::env::args().nth(4).expect("missing timeout_ms").parse().unwrap();
    println!(
        "Host:{} BatchSize:{} Iterations:{} Timeout:{}",
        host.clone(), b, n, timeout
    );
    println!("========================================================================================================================================================");

    let mut client = cb::inference_client::InferenceClient::connect(host).await?;
    
    let mut lat = vec![0u128; n];
    let mut model_lat = vec![0u128; n];
    let mut prep_lat = vec![0u128; n];
    for i in 1..n {
        let request = cb::PredictRequest {
            features: vec![cb::Features {
                cat_feature1: "A".to_string(),
                cat_feature2: "B".to_string(),
                cat_feature3: "C".to_string(),
                float_feature4: 0.5,
                float_feature5: 0.33,
            }; b] 
        };
        let start = Instant::now();
        let response = client.predict(request).await?.into_inner();
        lat[i] = start.elapsed().as_nanos();
        model_lat[i] = response.model_latency as u128;
        prep_lat[i] = response.preprocess_latency as u128;

        if i % REPORT == 0 {
            log_stats(">Prepr", i, n, timeout, &prep_lat, i - REPORT, REPORT);
            log_stats(">Model", i, n, timeout, &model_lat, i - REPORT, REPORT);
            log_stats("Client", i, n, timeout, &lat, i - REPORT, REPORT);
            println!("--------------------------------------------------------------------------------------------------------------------------------------------------------");
        }
    }

    println!("REPORT =================================================================================================================================================");
    log_stats(">Prepr", n, n, timeout, &prep_lat, 0, n);
    log_stats(">Model", n, n, timeout, &model_lat, 0, n);
    log_stats("Client", n, n, timeout, &lat, 0, n);
    println!("========================================================================================================================================================");
    Ok(())
}

fn log_stats(title: &str, i: usize, n: usize, timeout: u64, latencies: &Vec<u128>, skip: usize, take: usize) {
    let mean = latencies.iter().skip(skip).take(take).sum::<u128>() / (i - skip) as u128;
    let max = *latencies.iter().skip(skip).take(take).max().unwrap();
    let count = latencies.iter().skip(skip).take(take).collect::<Vec<&u128>>().len();
    let timeouts = latencies.iter().skip(skip).take(take).filter(
        |x| Duration::from_nanos(**x as u64) > Duration::from_millis(timeout)
    ).collect::<Vec<&u128>>().len();
    let success_ratio = 100.0 - 100.0 * (timeouts as f32 / n as f32);

    let ps: Vec<String> = percentiles(
        vec![0.95, 0.99, 0.999], latencies, skip, take
    ).iter().map(
        |(p, x)| format!("p{:.1}={:.3}ms", 100.0 * p, *x as f64 * 1e-6)
    ).collect();

    println!(
        "{}: \t{}\tMean={:.3}ms\tMax={:.3}ms\tCount={}\tTimeouts={}\tSucc={:0>3.3}%\t{}",
        title,
        i,
        mean as f64 * 1e-6,
        max as f64 * 1e-6,
        count,
        timeouts,
        success_ratio,
        ps.join("\t")
    );
}

fn percentiles(ps: Vec<f64>, latencies: &Vec<u128>, skip: usize, take: usize) -> Vec<(f64, u128)> {
    let mut sorted: Vec<&u128> = latencies.iter().skip(skip).take(take).collect();
    sorted.sort();
    ps.iter().map( |p|
        (*p, *sorted[(sorted.len() as f64 * p) as usize])
    ).collect()
}
