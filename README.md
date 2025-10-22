# softmax-smallnum (Python)

Small-number/digit recognition using **multinomial Softmax regression** with optional **PCA**.

- Pure NumPy implementation of Softmax training (batch gradient descent)
- Optional PCA (from-scratch) for dimensionality reduction
- Train/eval on your CSV (label,f1,...,fd) or on MNIST via `scikit-learn`

## Install
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start (toy CSV included)
```bash
python -m softmax_smallnum.cli \
  --train data/sample_train.csv --test data/sample_test.csv \
  --epochs 200 --alpha 0.3 --lambda_factor 1e-4 \
  --temp 1.0 --pca 0 --save model_softmax.npz
```

## MNIST (needs internet on first run)
```bash
python -m softmax_smallnum.cli --mnist \
  --epochs 150 --alpha 0.3 --lambda_factor 1e-4 \
  --temp 1.0 --pca 50 --save mnist_softmax_pca50.npz
```

## Predict on a CSV using a saved model
```bash
python -m softmax_smallnum.cli --predict data/sample_test.csv --load model_softmax.npz
```

## Project layout
```
softmax_smallnum/
  softmax.py    # core softmax
  pca.py        # PCA helpers
  data.py       # CSV & MNIST loaders
  utils.py      # plot/save/load
  train.py      # end-to-end training
  cli.py        # command-line interface
data/
  sample_train.csv
  sample_test.csv
```
