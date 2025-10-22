import argparse
from .data import load_csv, load_mnist_from_sklearn
from .train import train_softmax
from .utils import plot_cost, save_model, load_model
from .pca import project_onto_PC
from .softmax import get_classification

def main():
    ap = argparse.ArgumentParser(description="Softmax small-number recognizer")
    ap.add_argument("--train", type=str, help="CSV path (label,f1..fd) for training")
    ap.add_argument("--test", type=str, help="CSV path for testing")
    ap.add_argument("--mnist", action="store_true", help="Use MNIST from sklearn (internet required)")
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-test", type=int, default=None)

    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--lambda_factor", type=float, default=1e-4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--pca", type=int, default=0)

    ap.add_argument("--save", type=str, default=None, help="Path to save model npz")
    ap.add_argument("--load", type=str, default=None, help="Load model npz to predict")
    ap.add_argument("--predict", type=str, default=None, help="CSV to predict with loaded model")
    ap.add_argument("--no-show", action="store_true", help="Do not show plots")
    args = ap.parse_args()

    if args.load and args.predict:
        theta, meta = load_model(args.load)
        X, _ = load_csv(args.predict)
        if meta.get("pca_components", 0) and meta["pcs"] is not None:
            X = project_onto_PC(X, meta["pcs"], meta["pca_components"], meta["means"])
        yhat = get_classification(X, theta, meta["temp"])
        for i, pred in enumerate(yhat):
            print(f"{i},{pred}")
        return

    if args.mnist:
        Xtr, ytr, Xte, yte = load_mnist_from_sklearn(args.limit_train, args.limit_test)
    else:
        if not args.train or not args.test:
            ap.error("Provide --train and --test CSVs or use --mnist")
        Xtr, ytr = load_csv(args.train)
        Xte, yte = load_csv(args.test)

    theta, costs, test_err, mod3_err, meta = train_softmax(
        Xtr, ytr, Xte, yte,
        temp=args.temp, alpha=args.alpha, lambda_factor=args.lambda_factor,
        iters=args.epochs, k=int(max(ytr.max(), yte.max()) + 1),
        pca_components=args.pca, record_cost=True
    )

    print(f"Test error: {test_err:.4f}  (accuracy {(1.0 - test_err)*100:.2f}%)")
    if mod3_err is not None:
        print(f"Mod-3 test error: {mod3_err:.4f}")

    if args.save:
        save_model(args.save, theta, meta)
        print(f"Saved model to {args.save}")

    if not args.no_show:
        plot_cost(costs, show=True)

if __name__ == "__main__":
    main()
