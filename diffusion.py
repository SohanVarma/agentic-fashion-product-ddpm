import torch
import torch.nn.functional as F


class Diffusion:
    def __init__(self, timesteps=100, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.timesteps, size=(n,), device=self.device)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    @torch.no_grad()
    def sample(self, model, n, labels, image_size=64, channels=3, guidance_scale=2.5):
        model.eval()
        x = torch.randn((n, channels, image_size, image_size), device=self.device)
        labels = labels.to(self.device)
        null_labels = torch.full_like(labels, model.null_class)

        for i in reversed(range(1, self.timesteps)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            cond_pred = model(x, t, labels)
            if guidance_scale > 0:
                uncond_pred = model(x, t, null_labels)
                predicted_noise = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                predicted_noise = cond_pred

            alpha = self.alphas[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.betas[t][:, None, None, None]
            noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise

        x = (x.clamp(-1, 1) + 1) / 2
        return x
