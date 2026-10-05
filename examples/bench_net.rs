//! Times the actor-critic on a backend chosen by cargo feature: forward at rollout batch size,
//! forward+backward at minibatch size. Warms up first and syncs the device before reading the clock.
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_mario_rl::backend::{Inner, Train as B};
use burn_mario_rl::entities::ENT_LEN;
use burn_mario_rl::ppo::{ActorCritic, NetConfig};
use std::time::Instant;

fn main() {
    let device = Default::default();
    println!("backend {}", burn_mario_rl::backend::name());
    let cfg = NetConfig::from_env();
    burn_mario_rl::obs::set(cfg.trunk.obs_kind());
    let (oc, oh, ow) = burn_mario_rl::obs::dims();
    println!("net {}", cfg.to_line());
    let model = ActorCritic::<B>::new(&cfg, &device);
    for (batch, backward) in [(32usize, false), (128, false), (256, true), (256, true)] {
        let obs = Tensor::<B, 4>::random([batch, oc, oh, ow], burn::tensor::Distribution::Uniform(0.0, 255.0), &device);
        let ent = Tensor::<B, 2>::zeros([batch, ENT_LEN], &device);
        let run = || {
            let (logits, value) = model.forward(obs.clone(), ent.clone());
            if backward {
                let _grads = (logits.mean() + value.mean()).backward();
            } else {
                let _ = logits.into_data();
            }
            <Inner as Backend>::sync(&device).ok();
        };
        for _ in 0..4 {
            run(); // warm up: kernels compile per shape on first use
        }
        let reps = 10;
        let start = Instant::now();
        for _ in 0..reps {
            run();
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        println!("batch {batch:4} {}: {ms:8.1} ms  ({:.3} ms/sample)", if backward { "fwd+bwd" } else { "forward " }, ms / batch as f64);
    }
}
