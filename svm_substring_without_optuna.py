import numpy as np
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import cvxopt
import torch
import os
import numpy as np
import hashlib

def get_kernel_cache_filename(X_type, kernel_name, params, shape):
    param_str = "_".join([f"{k}={v}" for k, v in sorted(params.items())])
    shape_str = f"{shape[0]}x{shape[1]}"
    
    unique_hash = hashlib.md5((param_str + shape_str).encode()).hexdigest()[:8]
    return f"kernel_matrices/{X_type}_{kernel_name}_{unique_hash}.npy"

def save_kernel_matrix(K, X_type, kernel_name, params):
    os.makedirs("kernel_matrices", exist_ok=True)
    filename = get_kernel_cache_filename(X_type, kernel_name, params, K.shape)
    np.save(filename, K)
    print(f"Saved kernel matrix to {filename}")

def load_kernel_matrix(X_type, kernel_name, params, expected_shape):
    filename = get_kernel_cache_filename(X_type, kernel_name, params, expected_shape)
    if os.path.exists(filename):
        K = np.load(filename)
        if K.shape == expected_shape:
            print(f"Loaded cached kernel matrix from {filename}")
            return K
        else:
            print(f"Warning: Cached matrix shape {K.shape} != expected {expected_shape}")
    return None

def ssk_kernel(X, Y, k, lambda_decay=0.9):
    len_X, len_Y = len(X), len(Y)
    K = np.zeros((k+1, len_X+1, len_Y+1))

    # initialization
    K[0, :, :] = 1

    for m in range(1, k+1):
        sum_K = np.cumsum(K[m-1, :, :], axis=0)  
        sum_K = np.cumsum(sum_K, axis=1)      
        
        for i in range(1, len_X+1):
            for j in range(1, len_Y+1):
                K[m, i, j] = lambda_decay * (
                    K[m, i-1, j] + K[m, i, j-1] - 
                    lambda_decay * K[m, i-1, j-1] 
                )
                # if two characters are same
                if X[i-1] == Y[j-1]:
                    K[m, i, j] += lambda_decay**2 * sum_K[i-1, j-1]
    return K[k, len_X, len_Y]

#svm
def compute_kernel_for_pair(args):
    """
    kernel value for one pair (i, j) 
    """
    i, j, X, kernel_func, k = args
    score = kernel_func(X[i], X[j], k)
    return i, j, score

def compute_kernel_matrix(X_train, kernel_func, k, num_workers=8,reuse=False):
    params = {
        "kernel": kernel_func.__name__,
        "k": k,
        "n_samples": len(X_train)
    }
    # reload K if calculated
    if reuse:
        K = load_kernel_matrix("train", kernel_func.__name__, params, (len(X_train), len(X_train)))
        print("reuse")
        if K is not None:
            print("reuse k")
            return K
        
    n = len(X_train)
    K_matrix = np.zeros((n, n))

    tasks = [(i, j, X_train, kernel_func, k) for i in range(n) for j in range(i, n)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(compute_kernel_for_pair, task) for task in tasks]
        for future in as_completed(futures):
            i, j, score = future.result()
            K_matrix[i, j] = score
            K_matrix[j, i] = score  
    
    D = np.diag(1 / np.sqrt(np.diag(K_matrix)))
    K_norm = D @ K_matrix @ D
    K_norm += 1e-5 * np.eye(len(K_norm))
    save_kernel_matrix(K_matrix, "train", kernel_func.__name__, params)
    return K_norm

def compute_val_kernel_for_pair(args):
    i, j, val_sequences, train_sequences, k, lambda_decay = args
    score = ssk_kernel(val_sequences[i], train_sequences[j], k, lambda_decay)
    return i, j, score


def compute_kernel_val_matrix(val_sequences, train_sequences, k, lambda_decay=0.9, num_workers=8):
    num_val = len(val_sequences)
    num_train = len(train_sequences)
    K_val_train = np.zeros((num_val, num_train))
    
    tasks = [
        (i, j, val_sequences, train_sequences, k, lambda_decay)
        for i in range(num_val)
        for j in range(num_train)
    ]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(compute_val_kernel_for_pair, task) for task in tasks]
        for future in as_completed(futures):
            i, j, score = future.result()
            K_val_train[i, j] = score

    return K_val_train

def train_svm(K, y, C=1.0, tol=1e-4, max_iter=1000):
    # check K matrix
    eigenvalues = np.linalg.eigvalsh(K)
    print("min eigenvalues :", np.min(eigenvalues)) 
    print("nb conditions:", np.linalg.cond(K)) 

    n = len(y)
    y = y.astype(float).reshape(-1, 1)  
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['abstol'] = tol
    cvxopt.solvers.options['reltol'] = tol
    cvxopt.solvers.options['feastol'] = tol
    cvxopt.solvers.options['maxiters'] = max_iter

    # Construct the quadratic programming matrices
    P = cvxopt.matrix(np.outer(y, y) * K)
    q = cvxopt.matrix(-np.ones(n))         # q_i = -1
    G = cvxopt.matrix(np.vstack((-np.eye(n), np.eye(n))))  # Constraints 0 <= α <= C
    h = cvxopt.matrix(np.hstack((np.zeros(n), C * np.ones(n))))
    A = cvxopt.matrix(y.T)  # Equality constraint sum(α_i * y_i) = 0
    b = cvxopt.matrix(0.0)
    try:
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    except Exception as e:
        print("QP solver failed:", e)
        raise
    # Solve the quadratic program
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    print("solve")
    alphas = np.ravel(solution['x'])

    # Support vectors: α > 0
    threshold = np.max(alphas) * 1e-4
    sv_indices = alphas > threshold
    support_vectors = np.where(sv_indices)[0]
    alphas = alphas[sv_indices]
    support_y = y[sv_indices].flatten()

    # Compute the bias term (intercept)
    margin_sv = (alphas > 1e-5) & (alphas < C - 1e-5)
    if np.any(margin_sv):
        bias = np.mean(support_y[margin_sv] - np.dot(K[support_vectors][:, support_vectors], alphas * support_y)[margin_sv])
    else:
        bias = support_y[0] - np.dot(K[support_vectors[0], support_vectors], alphas * support_y)

    return alphas, support_vectors, bias


def predict_svm(K_test, alphas, support_vectors, support_y, bias, score_type = False):
    """Make predictions using the trained SVM."""
    score = np.dot(K_test[:, support_vectors], alphas * support_y) + bias
    if(score_type):
        return score
    return np.sign(score)
    
def load_data(seq_file: str, label_file: str) -> tuple:
    df_seq = pd.read_csv(seq_file)
    df_label = pd.read_csv(label_file)
    
    merged = pd.merge(df_seq, df_label, on='Id', how='inner')
    if merged.empty:
        print(f"Warning: {seq_file}and{label_file}don't have the same id")

    sequences = merged['seq'].values
    labels = np.where(merged['Bound'] == 1, 1, -1)
    return sequences, labels


def load_multiple_data(seq_files: list, label_files: list) -> tuple:
    if len(seq_files) != len(label_files):
        raise ValueError("Error : The number of seq_files and label_files should be same")
    
    merged_dfs = []
    for seq_file, label_file in zip(seq_files, label_files):
        df_seq = pd.read_csv(seq_file)
        df_label = pd.read_csv(label_file)
        
        merged = pd.merge(df_seq, df_label, on='Id', how='inner')
        if merged.empty:
            print(f"Warning: {seq_file}and{label_file}don't have the same id")
            continue
            
        merged_dfs.append(merged)
    
    full_df = pd.concat(merged_dfs, ignore_index=True)

    sequences = full_df['seq'].tolist()
    labels = np.where(full_df['Bound'] == 1, 1, -1)
    
    return sequences, labels

if __name__ == '__main__':
    t0 = time.time()
    seq_files =["data/Xtr0.csv","data/Xtr1.csv","data/Xtr2.csv"]
    label_files = ["data/Ytr0.csv","data/Ytr1.csv","data/Ytr2.csv"]
    seq_test_files = ["data/Xte0.csv","data/Xte1.csv","data/Xte2.csv"]
    y_test_file = "data/Ytrk_substring_train_all_at_once.csv"
    outputs=[]
    bestCs = [1, 1, 1]
    bestks = [7,7,7]

    for i in range(len(seq_files)):
        seq_file = seq_files[i]
        label_file = label_files[i]
        sequences_train, labels_train = load_data(seq_file, label_file)
        best_k = bestks[i]
        best_C = bestCs[i]
        # train 
        K_train = compute_kernel_matrix(sequences_train, ssk_kernel, best_k)
        alphas, support_vectors, bias = train_svm(K_train, labels_train, best_C)
        seq_test_file = seq_test_files[i]
        df_test = pd.read_csv(seq_test_file)
        if {'Id', 'seq'} - set(df_test.columns):
            raise ValueError(f"File {seq_test_file} doesn't have the ID column")
        
        # predict the test file
        test_sequences = df_test['seq'].tolist()
        test_ids = df_test['Id'].tolist()
        
        K_test = compute_kernel_val_matrix(test_sequences, sequences_train ,best_k)
        support_y = labels_train[support_vectors].flatten()  
        scores = predict_svm(K_test , alphas, support_vectors, support_y, bias, score_type=True)
        probs = 1 / (1 + np.exp(-scores)) 
        pred_labels = (probs > 0.5).astype(int)
        retsult = pd.DataFrame({
            'Id': test_ids,
            'Bound': pred_labels
        })
        
        print(f"Finished the prediction of : {seq_test_file}")
        outputs.append(retsult)

    # Merge results
    final_df = pd.concat(outputs, ignore_index=True)
    final_df.to_csv(y_test_file, index=False)
    print(f"The result is saved in: {y_test_file}")
    t1 = time.time()
    print('done in {:.3f} seconds'.format(t1 - t0))