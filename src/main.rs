mod data;
mod inference;
mod model;
mod training;

use crate::model::ModelConfig;
use burn::backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu};
use burn::data::dataset::Dataset;
use burn::optim::AdamConfig;

fn main() {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let artifact_dir = "/tmp/guide";

    let device = burn::backend::wgpu::WgpuDevice::default();
    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        crate::training::TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device,
    );

    let device = burn::backend::wgpu::WgpuDevice::default();
    crate::inference::infer::<MyBackend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    )
}
