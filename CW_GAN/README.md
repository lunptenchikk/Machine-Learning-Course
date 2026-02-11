# Conditional GANs – DCGAN vs WGAN vs WGAN-GP

This project compares three types of conditional GANs:

* Conditional DCGAN
* Conditional WGAN
* Conditional WGAN-GP

The goal was to understand how switching from standard GAN loss (BCE) to Wasserstein distance affects training stability and image quality.

## What I Did

I implemented all three models from scratch in TensorFlow/Keras and trained them on MNIST.
For each model:

* The generator takes random noise + a class label (one-hot encoded).
* The discriminator/critic receives the image along with the class label.
* Training metrics are logged.
* A 10×10 grid of generated images is saved after each epoch.

For WGAN:

* Used weight clipping to enforce the Lipschitz constraint.

For WGAN-GP:

* Replaced clipping with gradient penalty (λ = 10).
* Used the Adam optimizer as suggested in the original paper.

## Datasets

* MNIST – used for comparing the three architectures.
* KMNIST – used separately to generate higher-quality samples for a visual competition.

All images are normalized to [-1, 1] to match the generator’s `tanh` output.

## How to Run

python train.py --mode dcgan
python train.py --mode wgan
python train.py --mode wgangp

Outputs are saved in:
out/
  dcgan/
  wgan/
  wgangp/


Each folder contains training metrics and generated image grids.

## Final Thoughts

WGAN-GP was the most stable and produced the best-looking samples.
WGAN improved over classical GAN but was sensitive to weight clipping.
DCGAN worked fine on MNIST but showed more instability.

This project helped me understand the practical differences between adversarial losses and why Wasserstein-based GANs are often preferred in practice.
Thanks for attention!
