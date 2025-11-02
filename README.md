# DACD – Dose-Aware Cold Diffusion with Physics Consistency

Reproduction of DACD for generalizable LDCT reconstruction.

## Quickstart
```bash
conda env create -f env.yml
conda activate dacd

# Simulate doses & FBP baselines (LoDoPaB demo)
python scripts/simulate_dose_poisson.py --dataset lodopab --root /data/lodopab --doses 0.5 0.25 0.125 0.05

# Train (Stage A, then Stage B)
python scripts/train.py --data-root /data/lodopab --stage a --epochs 100 --batch-size 8
python scripts/train.py --data-root /data/lodopab --stage b --epochs 200 --batch-size 8 --resume outputs/stage_a/best.pt

# Eval
python scripts/eval.py --data-root /data/lodopab --ckpt outputs/stage_b/best.pt
```
Research use only. Not for clinical deployment.
