from __future__ import annotations
import argparse
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from dacd.data.datamodules import DataCfg, make_loader
from dacd.models.dacd_model import DACDModel
from dacd.eval.metrics import psnr, ssim, rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--angles", type=int, default=512)
    ap.add_argument("--det", type=int, default=512)
    args = ap.parse_args()
    data_cfg = DataCfg(root=args.data_root)
    loader = make_loader(data_cfg, split="val", batch_size=2, shuffle=False)
    x0 = next(iter(loader))
    H, W = x0["xgt"].shape[-2:]
    model = DACDModel((H, W), args.angles, args.det, device="cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=model.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    metrics = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            xgt = batch["xgt"].to(model.device)
            xfbp = batch["xfbpd"].to(model.device)
            yd = batch["yd"].to(model.device)
            xhat = model.infer(xfbp, yd)
            metrics.append({
                "psnr": psnr(xhat, xgt).mean().item(),
                "ssim": ssim(xhat, xgt).mean().item(),
                "rmse": rmse(xhat, xgt).mean().item(),
            })
    
    df = pd.DataFrame(metrics)
    out = Path(args.ckpt).with_suffix(".eval.csv")
    df.to_csv(out, index=False)
    print(df.describe())

if __name__ == "__main__":
    main()
