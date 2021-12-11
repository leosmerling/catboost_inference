use std::time::{Duration, Instant};

use tonic::{Request, Response, Status};

pub mod cb {
    tonic::include_proto!("cb"); // The string specified here must match the proto package name
}

const REPORT: usize = 1000;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Usage: cb-client http://host:port iterations timeout_ms");

    let host = std::env::args().nth(1).expect("http://host:port missing");
    let n: usize = std::env::args().nth(2).expect("missing number iterations").parse().unwrap();
    let timeout: u64 = std::env::args().nth(3).expect("missing timeout_ms").parse().unwrap();
    let mut client = cb::inference_client::InferenceClient::connect(host).await?;

    let mut latencies = vec![0u128; n];
    for i in 1..n {
        let request = Request::new(cb::PredictRequest {
            cat_feature1: "A".to_string(),
            cat_feature2: "B".to_string(),
            cat_feature3: "C".to_string(),
            float_feature4: 0.5,
            float_feature5: 0.33,
        });
        let start = Instant::now();
        let response = client.predict(request).await?;
        latencies[i] = start.elapsed().as_nanos();

        if i % REPORT == 0 {
            log_stats(i, n, timeout, &latencies, i - REPORT, REPORT);
        }
    }

    println!("REPORT ==================================================================================================================================");
    log_stats(n, n, timeout, &latencies, 0, n);
    println!("=========================================================================================================================================");
    Ok(())
}

fn log_stats(i: usize, n: usize, timeout: u64, latencies: &Vec<u128>, skip: usize, take: usize) {
    let mean = Duration::from_nanos((latencies.iter().skip(skip).take(take).sum::<u128>() / (i - skip) as u128) as u64);
    let max = Duration::from_nanos(*latencies.iter().skip(skip).take(take).max().unwrap() as u64);
    let count = latencies.iter().skip(skip).take(take).collect::<Vec<&u128>>().len();
    let timeouts = latencies.iter().skip(skip).take(take).filter(
        |x| Duration::from_nanos(**x as u64) > Duration::from_millis(timeout)
    ).collect::<Vec<&u128>>().len();
    let success_ratio = 100.0 - 100.0 * (timeouts as f32 / n as f32);

    let ps: Vec<String> = percentiles(
        vec![0.95, 0.99, 0.999], latencies, skip, take
    ).iter().map(
        |(p, x)| format!("p{:.1}: {:?}", 100.0 * p, x)
    ).collect();

    println!(
        "{} -> {} Mean: {:?} Max: {:?} Count: {} Timeouts: {} Success: {}% {}",
        skip,
        i,
        mean,
        max,
        count,
        timeouts,
        success_ratio,
        ps.join(" ")
    );
}

fn percentiles(ps: Vec<f64>, latencies: &Vec<u128>, skip: usize, take: usize) -> Vec<(f64, Duration)> {
    let mut sorted: Vec<&u128> = latencies.iter().skip(skip).take(take).collect();
    sorted.sort();
    ps.iter().map( |p|
        (*p, Duration::from_nanos(*sorted[(sorted.len() as f64 * p) as usize] as u64))
    ).collect()
}
