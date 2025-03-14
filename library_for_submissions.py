from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from scipy.sparse import coo_matrix
from itertools import product
from collections import defaultdict
import pandas as pd
import cvxopt
cvxopt.solvers.options["show_progress"] = False
import optuna
from scipy.linalg import solve

def load_data(seq_file: str, label_file: str) -> tuple:
    """
    加载多组CSV文件并合并数据
    
    参数:
        seq_files: 多个x.csv文件路径列表（如["x.csv", "x1.csv"]）
        label_files: 多个y.csv文件路径列表（如["y.csv", "y1.csv"]）
    
    返回:
        (sequences, labels): 合并后的序列列表和标签数组
    """
    # 合并所有数据
    df_seq = pd.read_csv(seq_file)
    df_label = pd.read_csv(label_file)
    
    merged = pd.merge(df_seq, df_label, on='Id', how='inner')
    if merged.empty:
        print(f"警告: {seq_file}和{label_file}中没有匹配的ID")

    # 提取数据
    sequences = merged['seq'].values
    labels = np.where(merged['Bound'] == 1, 1, -1)
    return sequences, labels

def save_to_csv(df_0, df_1, df_2, suffixe = ""):
    res = pd.concat([df_0, df_1, df_2], ignore_index = True)
    res["Bound"] = res["Bound"].astype("int")
    res[["Id", "Bound"]].to_csv(f"./kernel_data_challenge/results/Yte_{suffixe}.csv", index = False)
    return


def generate_mismatch_neighbors(kmer, m, alphabet = "ACGT"):
    """
    Generate all possible k-mers that are within 'm' mismatches of the given k-mer.
    
    Args:
        kmer (str): The original k-mer.
        m (int): Maximum allowed mismatches.
        alphabet (str): Possible characters in the k-mers (e.g., "ACGT").
    
    Returns:
        set: A set of k-mers within m mismatches.
    """
    if m == 0:
        return {kmer}  # No mismatches allowed, return the k-mer itself

    n = len(kmer)
    mismatch_neighbors = set()

    # Generate all possible positions and substitutions up to m mismatches
    def generate(pos, mismatches, current_kmer):
        if mismatches > m:  # Stop if we exceed the allowed mismatches
            return
        if pos == n:  # If we processed all positions, add the modified k-mer
            mismatch_neighbors.add("".join(current_kmer))
            return

        # Keep the original character (no mismatch at this position)
        generate(pos + 1, mismatches, current_kmer)

        # Try all possible mismatches at the current position
        original_char = current_kmer[pos]
        for char in alphabet:
            if char != original_char:  # Only substitute if it's different
                current_kmer[pos] = char
                generate(pos + 1, mismatches + 1, current_kmer)
                current_kmer[pos] = original_char  # Restore original

    generate(0, 0, list(kmer))
    return mismatch_neighbors

def compute_feature_vector(seq, k, m, alphabet, neighbor_cache):
    """
    Compute the feature vector for a given sequence using the mismatch kernel.

    Args:
        seq (str): Input DNA sequence.
        k (int): Length of k-mers.
        m (int): Maximum number of mismatches allowed.
        alphabet (str): Alphabet set (e.g., "ACGT").
        neighbor_cache (dict): Dictionary to cache computed mismatch neighborhoods.

    Returns:
        dict: Feature vector where keys are k-mers and values are their frequencies.
    """
    feature_vector = defaultdict(int)

    # Iterate over all k-mers in the sequence
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]  # Extract k-mer from the sequence

        # Check if neighbors are cached
        if kmer in neighbor_cache:
            neighbors = neighbor_cache[kmer]
        else:
            neighbors = generate_mismatch_neighbors(kmer, m, alphabet)
            neighbor_cache[kmer] = neighbors  # Cache the result

        # Update feature vector for all mismatch neighbors
        for neighbor in neighbors:
            feature_vector[neighbor] += 1  # Count occurrences

    return feature_vector

def compute_mismatch_kernel(sequences, k, m=1, alphabet="ACGT"):
    """
    Compute the mismatch kernel matrix for a set of sequences.
    
    Args:
        sequences (list of str): List of input DNA sequences.
        k (int): Length of k-mers.
        m (int): Maximum number of mismatches allowed.
        alphabet (str): Alphabet set (default: "ACGT").
    
    Returns:
        np.ndarray: The normalized mismatch kernel matrix.
    """
    n = len(sequences)

    # Create a shared neighbor cache
    neighbor_cache = {}

    # Compute feature vectors in parallel
    feature_vectors = list(
        Parallel(n_jobs=-1)(
            delayed(compute_feature_vector)(seq, k, m, alphabet, neighbor_cache)
            for seq in tqdm(sequences, total=n, desc="Computing feature vectors")
        )
    )

    # Build the global vocabulary from all feature vectors
    all_kmers = set()
    
    # Collect all k-mers from feature vectors in parallel
    all_kmers = set().union(*Parallel(n_jobs=-1)(
        delayed(lambda fv: set(fv.keys()))(fv) for fv in tqdm(feature_vectors, desc="Collecting k-mers")
    ))

    all_kmers = sorted(all_kmers)
    kmer_index = {kmer: idx for idx, kmer in enumerate(all_kmers)}

    # Collect entries for the sparse matrix in parallel
    rows, cols, data = [], [], []
    results = Parallel(n_jobs=-1)(
        delayed(lambda i, fv: (
            [i] * len(fv),
            [kmer_index[kmer] for kmer in fv.keys()],
            list(fv.values())
        ))(i, fv) for i, fv in tqdm(enumerate(feature_vectors), total=len(feature_vectors), desc="Building sparse matrix entries")
    )

    for r, c, d in results:
        rows.extend(r)
        cols.extend(c)
        data.extend(d)

    # Build the sparse matrix (COO format) and convert to CSR
    X = coo_matrix((data, (rows, cols)), shape=(n, len(all_kmers)), dtype=np.float32).tocsr()

    # Compute kernel matrix (dot product)
    K = X @ X.T
    print("K is computed")
    return K.toarray()


def train_svm(K, y, C=1.0):
    """Train an SVM using the precomputed kernel matrix K."""
    n = len(y)
    y = y.astype(float).reshape(-1, 1)  # Ensure y is a column vector

    # Construct the quadratic programming matrices
    P = cvxopt.matrix(np.outer(y, y) * K)  # P_ij = y_i * y_j * K_ij
    q = cvxopt.matrix(-np.ones(n))        # q_i = -1
    G = cvxopt.matrix(np.vstack((-np.eye(n), np.eye(n))))  # Constraints 0 <= α <= C
    h = cvxopt.matrix(np.hstack((np.zeros(n), C * np.ones(n))))
    A = cvxopt.matrix(y.T)  # Equality constraint sum(α_i * y_i) = 0
    b = cvxopt.matrix(0.0)

    # Solve the quadratic program
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    alphas = np.ravel(solution['x'])

    # Support vectors: α > 0
    sv_indices = alphas > 1e-5
    support_vectors = np.where(sv_indices)[0]
    alphas = alphas[sv_indices]
    support_y = y[sv_indices].flatten()

    # Compute the bias term (intercept)
    bias = np.mean(support_y - np.sum(alphas * support_y * K[support_vectors][:, support_vectors], axis=1))

    return alphas, support_vectors, bias

def predict_svm(K_test, alphas, support_vectors, support_y, bias):
    """Make predictions using the trained SVM."""
    return np.sign(np.sum(alphas * support_y * K_test[:, support_vectors], axis=1) + bias)


def predict_mismatch_svm(X_train_path, 
                         Y_train_path, 
                         X_test_path, 
                         best_k, 
                         best_m, 
                         best_C):
    "Predict with mismatch kernel and SVM"
    X_train, Y_train = load_data(X_train_path, Y_train_path)

    df_pred = pd.read_csv(X_test_path)
    X_test = df_pred["seq"].values

    # Train final model with best hyperparameters
    sequences = np.concatenate([X_train, X_test])
    kernel_matrix = compute_mismatch_kernel(sequences, best_k, m = best_m)
    K_train = kernel_matrix[0:len(X_train), 0:len(X_train)]
    K_test = kernel_matrix[len(X_train):len(X_train)+len(X_test), 0:len(X_train)]
    # K_test = spectrum_kernel_matrix(X_test, X_train, best_k)
    alphas, support_vectors, bias = train_svm(K_train, Y_train, best_C)

    # Predict on test set
    predictions = predict_svm(K_test, alphas, support_vectors, Y_train[support_vectors], bias)

    # Convert {-1,1} predictions to {0,1}
    predictions = (predictions + 1) // 2

    df_pred["Bound"] = predictions

    return df_pred


# ===== Kernel Ridge Regression (KRR) =====
def train_kernel_ridge_regression(K_train, y_train, lambda_reg=1.0):
    """Train Kernel Ridge Regression: Solves (K + λI)α = y."""
    n = K_train.shape[0]

    alpha = solve(K_train + lambda_reg * np.eye(n), y_train, assume_a='pos')
    return alpha

def predict_kernel_ridge_regression(K_test, alpha):
    """Predict using Kernel Ridge Regression."""
    return np.sign(K_test @ alpha)  # Predict {-1,1}

def predict_mismatch_krr(X_train_path, 
                         Y_train_path, 
                         X_test_path, 
                         best_k, 
                         best_m, 
                         best_lambda):
    """Predicts with Kernel Ridge Regression and mismatch kernel."""

    # Load data
    X_train, Y_train = load_data(X_train_path, Y_train_path)

    df_test = pd.read_csv(X_test_path)
    X_test = df_test["seq"].values

    # Train final model with best hyperparameters
    sequences = np.concatenate([X_train, X_test])
    kernel_matrix = compute_mismatch_kernel(sequences, best_k, m = best_m)
    K_train = kernel_matrix[0:len(X_train), 0:len(X_train)]
    K_test = kernel_matrix[len(X_train):len(X_train)+len(X_test), 0:len(X_train)]

    alpha = train_kernel_ridge_regression(K_train, Y_train, best_lambda)

    # Predict on test set
    predictions = predict_kernel_ridge_regression(K_test, alpha)

    # Convert {-1,1} predictions to {0,1}
    df_test["Bound"] = (predictions + 1) // 2

    return df_test