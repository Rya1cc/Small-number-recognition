import numpy as np

def augment_feature_vector(X):
    column_of_ones = np.ones([len(X), 1])
    return np.hstack((column_of_ones, X))

def compute_probabilities(X, theta, temp_parameter):
    scores = (theta @ X.T) / float(temp_parameter)
    scores -= np.max(scores, axis=0, keepdims=True)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
    return probs

def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    n = X.shape[0]
    probs = compute_probabilities(X, theta, temp_parameter)
    p_correct = probs[Y, np.arange(n)]
    eps = 1e-15
    data_loss = -np.sum(np.log(p_correct + eps)) / n
    reg = (lambda_factor / 2.0) * np.sum(theta ** 2)
    return data_loss + reg

def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    n = X.shape[0]
    tau = float(temp_parameter)
    P = compute_probabilities(X, theta, tau)
    grad_scores = P.copy()
    grad_scores[Y, np.arange(n)] -= 1
    grad_theta = (grad_scores @ X) / (n * tau) + lambda_factor * theta
    return theta - alpha * grad_theta

def softmax_regression(X, Y, temp_parameter=1.0, alpha=0.3, lambda_factor=1e-4, k=10, num_iterations=150, record_cost=True):
    X_aug = augment_feature_vector(X)
    theta = np.zeros([k, X_aug.shape[1]])
    costs = []
    for i in range(num_iterations):
        if record_cost:
            costs.append(compute_cost_function(X_aug, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X_aug, Y, theta, alpha, lambda_factor, temp_parameter)
    if record_cost:
        costs.append(compute_cost_function(X_aug, Y, theta, lambda_factor, temp_parameter))
    return theta, costs

def get_classification(X, theta, temp_parameter):
    X_aug = augment_feature_vector(X)
    probs = compute_probabilities(X_aug, theta, temp_parameter)
    return np.argmax(probs, axis=0)

def compute_test_error(X, Y, theta, temp_parameter):
    assigned = get_classification(X, theta, temp_parameter)
    return 1.0 - np.mean(assigned == Y)

def update_y_mod3(train_y, test_y):
    return (train_y % 3).astype(int), (test_y % 3).astype(int)

def compute_test_error_mod3(X, Y_mod3, theta, temp_parameter):
    pred_digits = get_classification(X, theta, temp_parameter)
    pred_mod3 = (pred_digits % 3).astype(int)
    return 1.0 - np.mean(pred_mod3 == Y_mod3)
