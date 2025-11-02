PY?=python

install:
	$(PY) -m pip install -e . -r requirements.txt
	pre-commit install

format:
	pre-commit run -a

simulate:
	$(PY) scripts/simulate_dose_poisson.py --root /data/lodopab

train:
	$(PY) scripts/train.py --data-root /data/lodopab --epochs 1 --batch-size 1

eval:
	$(PY) scripts/eval.py --data-root /data/lodopab --ckpt outputs/stage_a/best.pt
