import numpy as np

def center_data(X):
    means = X.mean(axis=0)
    return X - means, means

def principal_components(centered_data):
    scatter = centered_data.T @ centered_data
    eigvals, eigvecs = np.linalg.eig(scatter)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    return eigvecs.real

def project_onto_PC(X, pcs, n_components, feature_means):
    Xc = X - feature_means
    V = pcs[:, :n_components]
    return Xc @ V

def reconstruct_PC(x_pca, pcs, n_components, feature_means):
    return (x_pca @ pcs[:, :n_components].T) + feature_means
