# Agentic Conditional Fashion Product Image Generator using DDPM

This project is an upgraded Generative AI image generation system. It uses a **class-conditional diffusion model** trained on the Kaggle **Fashion Product Images Small** dataset and adds an agentic workflow around the model.

## Project Aim

The aim is to generate higher-quality RGB fashion product images based on a selected category such as `Apparel`, `Accessories`, `Footwear`, or `Personal Care`.

The system includes:

- Conditional DDPM image generation
- Classifier-Free Guidance
- Intent Agent
- Planner Agent
- Generation Agent
- Evaluator Agent
- Improvement Agent
- Report Agent
- Fashion product classifier for evaluation
- Real vs generated image comparison
- Training loss graph
- Streamlit UI

## Research Papers / References

- **Denoising Diffusion Probabilistic Models** by Ho, Jain, and Abbeel
- **Classifier-Free Diffusion Guidance** by Ho and Salimans

## Dataset

This project uses Kaggle's Fashion Product Images Small dataset.

Dataset page:
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

Expected dataset structure:

```text
data/fashion_products/
├── styles.csv
└── images/
    ├── 1163.jpg
    ├── 1164.jpg
    └── ...
```

## Setup

```bash
cd agentic_fashion_products_ddpm_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Download Dataset

Try:

```bash
python download_dataset.py
```

If download fails, manually download the dataset from Kaggle and place `styles.csv` and the `images/` folder in:

```text
data/fashion_products/
```

## Train Classifier

```bash
python train_classifier.py --epochs 5 --batch_size 64 --image_size 64 --max_images 12000
```

This saves:

```text
checkpoints/fashion_product_classifier.pt
```

## Train Diffusion Model

Fast demo training:

```bash
python train_diffusion.py --epochs 10 --batch_size 32 --timesteps 100 --image_size 64 --max_images 12000
```

Better quality:

```bash
python train_diffusion.py --epochs 30 --batch_size 32 --timesteps 200 --image_size 64 --max_images 20000
```

This saves:

```text
checkpoints/conditional_fashion_product_ddpm.pt
outputs/training_loss.csv
```

## Run App

```bash
python -m streamlit run app.py
```

## Generate from Terminal

```bash
python generate.py --class_name Footwear --num_images 8 --guidance_scale 2.5
```

## Architecture

```text
User Prompt
   ↓
Intent Agent
   ↓
Planner Agent
   ↓
Conditional DDPM Generation Agent
   ↓
Classifier Evaluator Agent
   ↓
Improvement Agent
   ↓
Report Agent
   ↓
Generated Product Images
```

## GitHub Notes

Do not upload dataset or checkpoint files. They are ignored in `.gitignore`.

Upload only the source code. The user can run `download_dataset.py` and train locally.
