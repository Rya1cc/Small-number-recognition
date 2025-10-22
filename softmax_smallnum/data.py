import numpy as np

def load_csv(path):
    labels, feats = [], []
    with open(path, 'r') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts=line.split(',')
            labels.append(int(parts[0]))
            feats.append([float(x) for x in parts[1:]])
    X = np.array(feats, dtype=float)
    y = np.array(labels, dtype=int)
    return X, y

def load_mnist_from_sklearn(limit_train=None, limit_test=None, random_state=0):
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist['data'] / 255.0
    y = mnist['target'].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=1/7, random_state=random_state, stratify=y)
    if limit_train is not None:
        Xtr, ytr = Xtr[:limit_train], ytr[:limit_train]
    if limit_test is not None:
        Xte, yte = Xte[:limit_test], yte[:limit_test]
    return Xtr, ytr, Xte, yte
