import torch
from diffusion import Diffusion


def generate_images(model, plan, image_size, device):
    diffusion = Diffusion(timesteps=plan["timesteps"], device=device)
    labels = torch.full((plan["num_images"],), plan["class_index"], dtype=torch.long, device=device)
    images = diffusion.sample(
        model,
        n=plan["num_images"],
        labels=labels,
        image_size=image_size,
        guidance_scale=plan["guidance_scale"],
    )
    return {
        "agent": "Generation Agent",
        "images": images,
        "message": f"Generated {plan['num_images']} image(s) for {plan['class_name']}.",
    }
