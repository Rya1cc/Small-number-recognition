from .softmax import softmax_regression, compute_test_error, compute_test_error_mod3
from .pca import center_data, principal_components, project_onto_PC

def train_softmax(
    Xtr, ytr, Xte, yte,
    temp=1.0, alpha=0.3, lambda_factor=1e-4, iters=150, k=10,
    pca_components=0, record_cost=True
):
    means = None
    pcs = None
    if pca_components and pca_components > 0:
        Xtr_c, means = center_data(Xtr)
        pcs = principal_components(Xtr_c)
        Xtr = project_onto_PC(Xtr, pcs, pca_components, means)
        Xte = project_onto_PC(Xte, pcs, pca_components, means)

    theta, costs = softmax_regression(
        Xtr, ytr, temp_parameter=temp, alpha=alpha,
        lambda_factor=lambda_factor, k=k, num_iterations=iters, record_cost=record_cost
    )
    test_err = compute_test_error(Xte, yte, theta, temp)

    mod3_err = None
    if ytr.max() <= 9 and ytr.min() >= 0:
        yte_m = (yte % 3).astype(int)
        mod3_err = compute_test_error_mod3(Xte, yte_m, theta, temp)

    meta = {
        "temp": temp, "alpha": alpha, "lambda_factor": lambda_factor,
        "iters": iters, "k": k, "pca_components": int(pca_components or 0),
        "means": means, "pcs": pcs
    }
    return theta, costs, test_err, mod3_err, meta
