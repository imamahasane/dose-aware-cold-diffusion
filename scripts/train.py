from __future__ import annotations
import argparse
from dacd.data.datamodules import DataCfg, make_loader
from dacd.engine.trainer import TrainCfg, Trainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--angles", type=int, default=512)
    ap.add_argument("--det", type=int, default=512)
    ap.add_argument("--stage", type=str, default="a", choices=["a","b"])
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    data_cfg = DataCfg(root=args.data_root)
    loader = make_loader(data_cfg, split="train", batch_size=args.batch_size)
    out_dir = args.out or (f"outputs/stage_{args.stage}")
    tcfg = TrainCfg(
        data_root=args.data_root,
        angles=args.angles,
        det=args.det,
        epochs=args.epochs,
        batch_size=args.batch_size,
        stage=args.stage,
        out_dir=out_dir,
    )
    trainer = Trainer(tcfg, loader)
    trainer.fit(args.epochs)

if __name__ == "__main__":
    main()
