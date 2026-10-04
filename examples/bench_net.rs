//! Times the actor-critic on a backend chosen by cargo feature: forward at rollout batch size,
//! forward+backward at minibatch size.
use burn::tensor::Tensor;
use burn_mario_rl::env::{OBS_H, OBS_W, STACK};
use burn_mario_rl::ppo::ActorCritic;
use std::time::Instant;

use burn_mario_rl::backend::Train as B;

fn main() {
    let device = Default::default();
    let model = ActorCritic::<B>::new(&device);
    for (batch, backward) in [(10usize, false), (96, false), (320, true)] {
        let obs = Tensor::<B, 4>::random([batch, STACK, OBS_H, OBS_W], burn::tensor::Distribution::Uniform(0.0, 255.0), &device);
        let reps = 5;
        let _ = model.forward(obs.clone()); // warm up
        let start = Instant::now();
        for _ in 0..reps {
            let (logits, value) = model.forward(obs.clone());
            if backward {
                let loss = logits.mean() + value.mean();
                let _ = loss.backward();
            } else {
                let _ = logits.into_data();
            }
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        println!("batch {batch:4} {}: {ms:8.1} ms  ({:.2} ms/sample)", if backward { "fwd+bwd" } else { "forward " }, ms / batch as f64);
    }
}
