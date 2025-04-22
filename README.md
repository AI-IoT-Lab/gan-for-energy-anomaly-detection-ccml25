# gan-for-energy-anomaly-detection
Code repository for the paper Generative Adversarial Networks for Unsupervised Anomaly Detection in Energy Time series data which was accepted at ICLR 2025 Workshop: Tackling Climate Change with Machine Learning

## LEAD Dataset

This dataset contains hourly electricity meter readings and anomaly annotations for various commercial buildings over a period of up to one year. The data is structured as follows:

- `building_id`: Unique identifier for each building.
- `timestamp`: Hourly timestamp of the meter reading.
- `meter_reading`: Actual electricity meter reading value.
- `anomaly`: Binary indicator of whether the timestamp (reading) is considered anomalous (1) or not (0).

The dataset covers readings from 200 buildings, with each building having approximately 8,747 data points. Anomaly annotations are provided to mark specific timestamps within each building's time series where anomalous readings were detected.

Here's a small example of the dataset:

| building_id | timestamp       | meter_reading | anomaly |
|-------------|-----------------|---------------|---------|
| 1           | 01-01-2016 00:00| 100.5         | 0       |
| 1           | 01-01-2016 01:00| 98.2          | 0       |
| 1           | 01-01-2016 02:00| 95.7          | 0       |
| 2           | 01-01-2016 00:00| 200.1         | 0       |
| 2           | 01-01-2016 01:00| 203.4         | 1       |
| 2           | 01-01-2016 02:00| 197.8         | 0       |


## Repository Structure

```yaml
.
├───dataset                         <- Contains the 200 buildings dataset.
├───experimental                    <- Contains code for baseline models
│   ├───1d-autoencoder
│   ├───CNN VAE
│   ├───Isolation Forest
│   ├───LOF
├───model_input                     <- Training data tensors are stored here
├───plots                           <- Plots illustrating different processes
│   ├───anom_detect
│   ├───gen_model
│   ├───sample_reconstruct
│   ├───test_segments
│   └───train_segments
├───test_out                        <- Reconstruction pickle file are stored for test segments
└───trained_out                     <- Trained Models are stored here 

```

## Config JSON File Details

Given below is the config file with default values.

```yaml
{
    "data": {
        "dataset_path": "dataset/15_builds_dataset.csv",
        "train_path": "model_input/",
        "only_building": 1304
    },
    "training": {
        "batch_size": 128,
        "num_epochs": 200,
        "latent_dim": 100,
        "w_gan_training": true,
        "n_critic": 5,
        "clip_value": 0.01,
        "betaG": 0.5,
        "betaD": 0.5,
        "lrG": 0.0002,
        "lrD": 0.0002
    },
    "preprocessing": {
        "normalize": true,
        "plot_segments": true,
        "store_segments": true,
        "window_size": 48
    },
    "recon": {
        "use_dtw": false,
        "iters": 1000,
        "use_eval_mode": true
    }
}
```

### Data Section
- `dataset_path`: Path to the dataset file (`"dataset/200_builds_dataset.csv"`) (Needs to be uploaded)
- `train_path`: Path where the training data or model inputs are stored (`"model_input/"`)
- `only_building`: Particular building identifier or index (`1304`)

### Training Section
- `batch_size`: Number of samples per batch during the training process (`128`)
- `num_epochs`: Number of training epochs (`200`)
- `latent_dim`: Dimensionality of the latent space in the model (`100`)
- `w_gan_training`: Indicates whether to use Wasserstein GAN (WGAN) training (`true`)
- `n_critic`: Number of critic iterations per generator iteration in WGAN training (`5`)
- `clip_value`: Clipping value for the critic's weights in WGAN training (`0.01`)
- `betaG` and `betaD`: Beta values for the generator and discriminator, respectively (`0.5`)
- `lrG` and `lrD`: Learning rates for the generator and discriminator, respectively (`0.0002`)

### Preprocessing Section
- `normalize`: Indicates whether to normalize the sements (transform all the readings in a segment to be in the [-1,1] range). (`true`)
- `plot_segments`: Specifies whether to plot the segments (`true`)
- `store_segments`: Indicates whether to store the segments (`true`)
- `window_size`: Size of the window for data preprocessing (`48`)

### Reconstruction (recon) Section
- `use_dtw`: Will work on this later. Make sure it is False to use MSE!(`false`)
- `iters`: Number of iterations  used by the gradient descent algorithm in noise space for rconstruction (`1000`)
- `use_eval_mode`: Indicates whether to use evaluation mode of the Generator is used during reconstruction (`true`)

## Anomaly Detection on the Entire Dataset

Anomaly detection process for the entire set of 200 buildings follow the same steps. Each building gets its own GAN model. The process is automated by the `run.py` script where the reconstruction pickle files are obtained for each building by running the `preprocessing.py`, `training.py` and `testing.py` scripts on loop.

1. Set up the appropriate configuration in `config.json`
2. Run `run.py` (It runs 3 scripts and creates reconstruction data pickle files)
3. Run `anom_detect_gan.py` 
4. Run `plotting.py` to create plots for the anomaly detection


## Baseline Methodologies


The directory "experimental" contains code for comparisons with other DL and ML methods.

It includes:
1. Convolutional Variational Autoencoder
2. 1D CNN Autoencoder
3. Local Outlier Factor
4. Isolation Forest

## Configurations for models

1. DCGAN with Conventional Loss

```yaml
{
    "data": {
        "dataset_path": "dataset/200_builds_dataset.csv",
        "train_path": "model_input/",
        "only_building": 1304
    },
    "training": {
        "batch_size": 128,
        "num_epochs": 200,
        "latent_dim": 100,
        "w_gan_training": false,
        "n_critic": 5,
        "clip_value": 0.01,
        "betaG": 0.5,
        "betaD": 0.5,
        "lrG": 0.0002,
        "lrD": 0.0002
    },
    "preprocessing": {
        "normalize": true,
        "plot_segments": true,
        "store_segments": true,
        "window_size": 48
    },
    "recon": {
        "use_dtw": false,
        "iters": 1000,
        "use_eval_mode": true
    }
}
```

2. DCGAN with Wasserstein Loss

```yaml
{
    "data": {
        "dataset_path": "dataset/200_builds_dataset.csv",
        "train_path": "model_input/",
        "only_building": 1304
    },
    "training": {
        "batch_size": 128,
        "num_epochs": 200,
        "latent_dim": 100,
        "w_gan_training": True,
        "n_critic": 5,
        "clip_value": 0.01,
        "betaG": 0.5,
        "betaD": 0.5,
        "lrG": 0.0002,
        "lrD": 0.0002
    },
    "preprocessing": {
        "normalize": true,
        "plot_segments": true,
        "store_segments": true,
        "window_size": 48
    },
    "recon": {
        "use_dtw": false,
        "iters": 1000,
        "use_eval_mode": true
    }
}
```

3. Other Baseline models

```yaml
{
    "data": {
        "dataset_path": "dataset/200_builds_dataset.csv",
        "train_path": "model_input/",
        "only_building": 1304
    },
    "training": {
        "batch_size": 128,
        "num_epochs": 200,
        "latent_dim": 100,
        "w_gan_training": false,
        "n_critic": 5,
        "clip_value": 0.01,
        "betaG": 0.5,
        "betaD": 0.5,
        "lrG": 0.0002,
        "lrD": 0.0002
    },
    "preprocessing": {
        "normalize": true,
        "plot_segments": true,
        "store_segments": true,
        "window_size": 48
    },
    "recon": {
        "use_dtw": false,
        "iters": 1000,
        "use_eval_mode": true
    }
}
```
a. Set the above configuration

b. cd experimental/

c. Run `run.py`

d. Run `anom_detect.py`
