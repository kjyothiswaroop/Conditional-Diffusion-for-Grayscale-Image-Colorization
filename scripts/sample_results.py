import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from train import Diffusion

PROJECT_ROOT = Path(__file__).parent.parent


def main(n: int, diffusion_steps: int, out: Path):
    model = Diffusion(inference=True)
    checkpoint = torch.load(PROJECT_ROOT / "checkpoints" / "latest.pt", map_location=model.device)
    model.ema_network.load_state_dict(checkpoint["ema_network"])
    model.ema_network.eval()

    dataset = load_dataset(model.dataset_name, split="test").with_format("torch")
    samples = [dataset[i] for i in range(n)]

    color = torch.stack([s["color"] for s in samples]).to(model.device)
    gray = torch.stack([s["gray"] for s in samples]).to(model.device)

    gray_norm = model._normalize_images(gray)
    gray_3ch = gray_norm.expand(-1, 3, -1, -1) if gray_norm.shape[1] == 1 else gray_norm

    generated = model.reverse_diffusion(gray_norm, diffusion_steps)

    display_gray = model._denormalize_images(gray_3ch)
    display_gen = model._denormalize_images(generated)
    display_color = model._denormalize_images(model._normalize_images(color))

    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))
    if n == 1:
        axes = [[axes[r]] for r in range(3)]

    row_labels = ["Input (Gray)", "Generated", "Ground Truth"]
    for col in range(n):
        axes[0][col].imshow(display_gray[col].permute(1, 2, 0).cpu(), cmap="gray")
        axes[1][col].imshow(display_gen[col].permute(1, 2, 0).cpu())
        axes[2][col].imshow(display_color[col].permute(1, 2, 0).cpu())
        for row in range(3):
            axes[row][col].axis("off")
            if col == 0:
                axes[row][col].set_title(row_labels[row], fontsize=10, pad=4)

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3, help="Number of test images")
    parser.add_argument("--steps", type=int, default=100, help="Diffusion steps")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "samples" / "results.png")
    args = parser.parse_args()
    main(args.n, args.steps, args.out)
