# Conditional Diffusion for Grayscale Image Colorization

This project trains a diffusion model to colorize grayscale images. A grayscale image is used as a conditioning signal by concatenating it as an additional channel to the noisy RGB image during training. At inference, the model takes a grayscale image and generates a colorized version.

The backbone is a U-Net based diffusion model trained on face images from the CelebA-HQ dataset.

## Dataset Generation

To generate and push the processed dataset to HuggingFace, first login:

```bash
huggingface-cli login
```

Then run:

```bash
python scripts/generate_dataset.py \
  --source-repo korexyz/celeba-hq-256x256 \
  --output-repo <your-hf-username>/celebahq-128-gray
```

For a quick test run with a small sample:

```bash
python scripts/generate_dataset.py \
  --source-repo korexyz/celeba-hq-256x256 \
  --output-repo <your-hf-username>/celebahq-128-gray \
  --max-samples 10
```

**Source dataset:** `korexyz/celeba-hq-256x256` — CelebA-HQ face images at 256x256.

**Generated dataset:** Each sample contains a 128x128 RGB color image and its grayscale version. Images are resized from 256x256 using LANCZOS resampling.

| Split      | Samples |
|------------|---------|
| Train      | 28,000  |
| Validation | 1,000   |
| Test       | 1,000   |
