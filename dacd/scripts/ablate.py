from __future__ import annotations
import argparse
import torch
from dacd.data.datamodules import DataCfg, make_loader
from dacd.models.dacd_model import DACDModel
from dacd.eval.metrics import psnr, ssim, rmse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--angles", type=int, default=512)
    ap.add_argument("--det", type=int, default=512)
    ap.add_argument("--toggle", type=str, nargs="+", default=[], help="disable modules: DAP PEMPP DCSA PHYS")
    args = ap.parse_args()
    data_cfg = DataCfg(root=args.data_root)
    loader = make_loader(data_cfg, split="val", batch_size=1, shuffle=False)
    batch0 = next(iter(loader))
    H, W = batch0["xgt"].shape[-2:]
    model = DACDModel((H, W), args.angles, args.det, device="cuda" if torch.cuda.is_available() else "cpu").eval()

    if "DAP" in args.toggle:
        model.dap.forward = lambda x: (torch.zeros(x.size(0), device=x.device), torch.zeros(x.size(0), 64, device=x.device))
    if "PEMPP" in args.toggle:
        model.pempp.forward = lambda x, e, t: torch.zeros(x.size(0), 32, x.size(2), x.size(3), device=x.device)
    if "DCSA" in args.toggle:
        model.dcsa.forward = lambda e: torch.full((e.size(0),), 8, dtype=torch.int64, device=e.device)
    if "PHYS" in args.toggle:
        @torch.no_grad()
        def infer_no_phys(xfbp, yd):
            xfbp = xfbp.to(model.device)
            sev, ed = model.dap(xfbp)
            steps = model.dcsa(ed)
            T = int(steps.max().item())
            xt = xfbp
            for tt in range(T, 0, -1):
                tvec = torch.full((xfbpt := xfbp.shape[0],), float(tt), device=model.device)
                et = torch.sin(torch.linspace(0, 1, 64, device=model.device))[None, :].expand(xfbp.size(0), -1)
                prior = model.pempp(xt, ed, et)
                denoised = model.unet(xt, prior=prior)
                xt = denoised
            return xt
        model.infer = infer_no_phys

    ps, ss, rr = [], [], []
    with torch.no_grad():
        for batch in loader:
            xgt = batch["xgt"].to(model.device)
            xfbp = batch["xfbpd"].to(model.device)
            yd = batch["yd"].to(model.device)
            xhat = model.infer(xfbp, yd)
            ps.append(psnr(xhat, xgt).mean().item())
            ss.append(ssim(xhat, xgt).mean().item())
            rr.append(rmse(xhat, xgt).mean().item())
    print(f"Ablation ({args.toggle}): PSNR={sum(ps)/len(ps):.3f}, SSIM={sum(ss)/len(ss):.4f}, RMSE={sum(rr)/len(rr):.4f}")

if __name__ == "__main__":
    main()
