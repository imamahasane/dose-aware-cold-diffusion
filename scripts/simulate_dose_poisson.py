from __future__ import annotations
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
from dacd.data.ct_ops import CTOps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="lodopab", help="name for bookkeeping")
    ap.add_argument("--root", type=str, required=True, help="root with images/*.pt")
    ap.add_argument("--doses", type=float, nargs="+", default=[0.5, 0.25, 0.125, 0.05])
    ap.add_argument("--angles", type=int, default=512)
    ap.add_argument("--det", type=int, default=512)
    args = ap.parse_args()
    root = Path(args.root)
    images = sorted((root / "images").glob("*.pt"))
    assert len(images) > 0, f"No images in {root}/images"
    x0 = torch.load(images[0])
    H, W = x0.shape[-2:]
    ct = CTOps((H, W), angles=args.angles, det_count=args.det, device="cuda" if torch.cuda.is_available() else "cpu")
    (root / "sinograms_clean").mkdir(parents=True, exist_ok=True)
    
    for p in tqdm(images, desc="Simulating doses"):
        xgt = torch.load(p).unsqueeze(0)  # [1,1,H,W]
        yclean = ct.project(xgt)
        torch.save(yclean.cpu(), root / "sinograms_clean" / p.name)

        for d in args.doses:
            yd = torch.poisson(d * torch.clamp(yclean, min=0))
            fbp = ct.fbp(yd)
            sd = root / f"sinograms_d{d}"
            xd = root / f"fbp_d{d}"
            sd.mkdir(parents=True, exist_ok=True)
            xd.mkdir(parents=True, exist_ok=True)
            torch.save(yd.cpu(), sd / p.name)
            torch.save(fbp.cpu().squeeze(0), xd / p.name)
    print("Dose simulation complete.")

if __name__ == "__main__":
    main()
