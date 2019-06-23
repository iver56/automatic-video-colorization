# Noise2Noise

This is an unofficial and partial Keras implementation of "Noise2Noise: Learning Image Restoration without Clean Data" [1].

It is a fork of https://github.com/yu4u/noise2noise adapted for video colorization purposes

There are several things different from the original paper
(but not a fatal problem to see how the noise2noise training framework works):
- Training dataset (orignal: ImageNet, this repository: [2])
- Model (original: RED30 [3], this repository: SRResNet [4] or UNet [5])

## Dependencies
- Keras >= 2.1.2, TensorFlow, NumPy, OpenCV

## Train Noise2Noise

##### Model architectures
With `--model unet`, UNet model can be trained instead of SRResNet.

##### Resume training
With `--weight path/to/weight/file`, training can be resumed with trained weights.


### Noise Models
Using `source_noise_model`, `target_noise_model`, and `val_noise_model` arguments,
arbitrary noise models can be set for source images, target images, and validation images respectively.
Default values are taken from the experiment in [1].

- Gaussian noise
  - gaussian,min_stddev,max_stddev (e.g. gaussian,0,50)
- Clean target
  - clean

You can see how the gaussian noise model works by running:

```bash
python noise_model.py
```

## References

[1] J. Lehtinen, J. Munkberg, J. Hasselgren, S. Laine, T. Karras, M. Aittala, 
T. Aila, "Noise2Noise: Learning Image Restoration without Clean Data," in Proc. of ICML, 2018.

[2] J. Kim, J. K. Lee, and K. M. Lee, "Accurate Image Super-Resolution Using Very Deep Convolutional Networks," in Proc. of CVPR, 2016.

[3] X.-J. Mao, C. Shen, and Y.-B. Yang, "Image
Restoration Using Convolutional Auto-Encoders with
Symmetric Skip Connections," in Proc. of NIPS, 2016.

[4] C. Ledig, et al., "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network," in Proc. of CVPR, 2017.

[5] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," in MICCAI, 2015.
