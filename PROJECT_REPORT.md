# Project Report: Agentic Conditional Fashion Product Image Generator using DDPM

## Abstract

This project implements an agent-assisted generative AI system for fashion product image generation. Unlike a basic unconditional diffusion model, this system uses class-conditional generation to produce product images based on selected categories. It also adds classifier-free guidance, an evaluator classifier, and multiple functional agents that plan, generate, evaluate, improve, and report the final output.

## Problem Statement

Basic image generation models often produce random images without user control. The goal of this project is to generate fashion product images that match a user-selected category and provide measurable evaluation feedback.

## Dataset

The project uses the Kaggle Fashion Product Images Small dataset. The dataset contains product images and metadata such as category labels in `styles.csv`. Images are resized to 64x64 RGB for training.

## Methodology

The diffusion model is trained using a forward noising process and a reverse denoising process. During training, Gaussian noise is added to product images at random timesteps. The denoising model learns to predict the noise added to the image.

## Conditional Generation

The model receives three inputs:

1. Noisy image
2. Timestep
3. Class label

This allows the generator to produce category-specific fashion product images.

## Classifier-Free Guidance

During training, labels are randomly dropped and replaced with a null label. During sampling, conditional and unconditional predictions are combined:

```text
guided_prediction = unconditional + guidance_scale × (conditional - unconditional)
```

This improves control over the generated category.

## Agentic Workflow

The project includes the following agents:

| Agent | Responsibility |
|---|---|
| Intent Agent | Parses user request and detects product category |
| Planner Agent | Chooses generation settings |
| Generation Agent | Runs the conditional DDPM model |
| Evaluator Agent | Uses a classifier to check generated image category confidence |
| Improvement Agent | Regenerates images if confidence is low |
| Report Agent | Summarizes workflow and results |

## Expected Output

The app displays:

- Generated product images
- Agent workflow log
- Classifier confidence
- Real vs generated comparison
- Training loss graph

## Conclusion

This project demonstrates a stronger generative AI system than a simple image generator by combining conditional diffusion, classifier-free guidance, image classification evaluation, and an agentic workflow.
