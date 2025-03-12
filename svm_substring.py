import numpy as np
import pandas as pd
import time
import optuna
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

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

def compute_kernel_for_pair(args):
    i, j, X_train, kernel_func, k = args
    score = kernel_func(X_train[i], X_train[j], k)
    return i, j, score

def compute_kernel_matrix(X_train, kernel_func, k, num_workers=8):
    n = len(X_train)
    K_matrix = np.zeros((n, n))

    tasks = [(i, j, X_train, kernel_func, k) for i in range(n) for j in range(i, n)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(compute_kernel_for_pair, task) for task in tasks]
        for future in as_completed(futures):
            i, j, score = future.result()
            K_matrix[i, j] = score
            K_matrix[j, i] = score  

    return K_matrix

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

class KernelSVM:
    def __init__(self, C=1.0, max_iters=1000, tol=1e-3):
        self.C = C          
        self.max_iters = max_iters
        self.tol = tol      
        self.alpha = None   
        self.b = 0.0        
        self.X_train = None 
    
    def fit(self, K, y, X_train):
        self.X_train = X_train
        self.y_train = y
        n_samples = K.shape[0]
        self.alpha = np.zeros(n_samples)
        y = y.astype(np.float32)
        
        # SMO
        for _ in range(self.max_iters):
            alpha_prev = np.copy(self.alpha)
            
            for i in range(n_samples):
                # error
                Ei = (self.alpha * y).dot(K[i]) + self.b - y[i]
                
                # update when KKT conditions are violated
                if (y[i]*Ei < -self.tol and self.alpha[i] < self.C) or \
                   (y[i]*Ei > self.tol and self.alpha[i] > 0):
                    
                    j = np.random.choice(list(range(n_samples)))
                    Ej = (self.alpha * y).dot(K[j]) + self.b - y[j]
                    
                    # update alpha_i alpha_j
                    eta = K[i,i] + K[j,j] - 2*K[i,j]
                    if eta == 0:
                        continue
                        
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    
                    L, H = self._compute_L_H(alpha_i_old, alpha_j_old, y[i], y[j])
                    
                    self.alpha[j] = alpha_j_old + (y[j]*(Ei - Ej))/eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    self.alpha[i] = alpha_i_old + y[i]*y[j]*(alpha_j_old - self.alpha[j])
                    
                    # biais
                    b1 = self.b - Ei - y[i]*(self.alpha[i]-alpha_i_old)*K[i,i] - y[j]*(self.alpha[j]-alpha_j_old)*K[i,j]
                    b2 = self.b - Ej - y[i]*(self.alpha[i]-alpha_i_old)*K[i,j] - y[j]*(self.alpha[j]-alpha_j_old)*K[j,j]
                    self.b = (b1 + b2)/2
                    
            # check converge 
            if np.linalg.norm(self.alpha - alpha_prev) < self.tol:
                break
    
    def _compute_L_H(self, alpha_i, alpha_j, yi, yj):
        if yi != yj:
            L = max(0, alpha_j - alpha_i)
            H = min(self.C, self.C + alpha_j - alpha_i)
        else:
            L = max(0, alpha_j + alpha_i - self.C)
            H = min(self.C, alpha_j + alpha_i)
        return L, H

    def predict(self, valid_sequences, valid_labels, k ):
        K_test = compute_kernel_val_matrix(valid_sequences,self.X_train ,k)
        scores =  (self.alpha * self.y_train).dot(K_test.T) + self.b
        pred_labels = np.where(scores >= 0, 1, 0)
        true_predictions = sum(p == r for p, r in zip(pred_labels, valid_labels))  # 计算正确预测数
        return  true_predictions / len(valid_labels)

    def predict_optuna(self, valid_sequences, valid_labels, k , support_vectors, train_seq):
        K_test = compute_kernel_val_matrix(valid_sequences,train_seq,k)
        scores = self.alpha[support_vectors] * self.y_train[support_vectors]*K_test.T + self.b
        pred_labels = np.where(scores >= 0, 1, 0)
        true_predictions = sum(p == r for p, r in zip(pred_labels, valid_labels))  # 计算正确预测数
        return  true_predictions / len(valid_labels)


    def predict_file(self, file_path: str, k: int):
        try:
            df_test = pd.read_csv(file_path)
            if {'Id', 'seq'} - set(df_test.columns):
                raise ValueError(f" {file_path} lack ID or seq column")
            
            test_sequences = df_test['seq'].tolist()
            test_ids = df_test['Id'].tolist()
            
            K_test = compute_kernel_val_matrix(test_sequences,self.X_train ,k)
            scores =  (self.alpha * self.y_train).dot(K_test.T) + self.b
            pred_labels = np.where(scores >= 0, 1, 0)
            retsult = pd.DataFrame({
                'Id': test_ids,
                'Bound': pred_labels
            })
            
            print(f" Finished the prediction of  {file_path} ")
            
        except Exception as e:
            print(f"Error in  {file_path} exception message : {str(e)}")

        return retsult
    
    def _compute_test_kernel(self, test_sequences: list, k: int) -> np.ndarray:
        n_test = len(test_sequences)
        n_train = len(self.X_train)
        
        K_test = np.zeros((n_test, n_train))
        for i in range(n_test):
            for j in range(n_train):
                K_test[i,j] = ssk_kernel(test_sequences[i], self.X_train[j], k)
        return K_test
    
def load_data(seq_file: str, label_file: str) -> tuple:
    df_seq = pd.read_csv(seq_file)
    df_label = pd.read_csv(label_file)
    
    merged = pd.merge(df_seq, df_label, on='Id', how='inner')
    if merged.empty:
        print(f"警告: {seq_file}和{label_file}中没有匹配的ID")

    sequences = merged['seq'].values
    labels = np.where(merged['Bound'] == 1, 1, -1)
    return sequences, labels

def manual_kfold_split(X, y, n_splits=3, seed=42):
    """Manually splits X and y into K folds for cross-validation."""
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    folds = np.array_split(indices, n_splits)
    return folds

def cross_val_score_manual(X, y, k, C, n_splits=3):
    """Performs cross-validation without using sklearn."""
    folds = manual_kfold_split(X, y, n_splits)
    accuracies = []

    for i in range(n_splits):
        # print("i = ", i)
        val_indices = folds[i]  # Current fold is validation set
        train_indices = np.hstack([folds[j] for j in range(n_splits) if j != i])  # Rest are training

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        # Compute kernel matrices
        K_train = compute_kernel_matrix(X_train, ssk_kernel, k)

        svm = KernelSVM(C, max_iters=10000)
        svm.fit(K_train, y_train, X)
        # Train and predict

        accuracy = svm.predict_optuna(X_val, y_val, k, len(val_indices),X_train)
        print(accuracy[0])
        print(accuracy.shape)
        # Compute accuracy
        accuracies.append(accuracy[0])

    return np.mean(accuracies)

def objective(X_train,Y_train,trial):
    """Objective function for Optuna to optimize k and C."""
    k = trial.suggest_int("k", 4, 7)  # k-mer length between 1 and 6
    C = trial.suggest_loguniform("C", 0.01, 10)  # C in [0.01, 100]

    cross_val = cross_val_score_manual(X_train, Y_train, k, C)

    print(f"C = {C}, k = {k}: {cross_val}")
    return cross_val

if __name__ == '__main__':
    t0 = time.time()
    seq_files =["data/Xtr0.csv","data/Xtr1.csv","data/Xtr2.csv"]
    label_files = ["data/Ytr0.csv","data/Ytr1.csv","data/Ytr2.csv"]
    seq_test_files = ["data/Xte0.csv","data/Xte1.csv","data/Xte2.csv"]
    y_test_file = "data/Ytrk_with_optuna_new.csv"
    n_trials  = 4
    outputs=[]
    bestCs = [1.7259685252606225, 0.013650844380457251,0.2166487544857509]
    bestks = [7,6,7]
    for i in range(len(seq_files)):
        seq_file = seq_files[i]
        label_file = label_files[i]
        sequences_train, labels_train = load_data(seq_file, label_file)
        best_k = bestks[i]
        best_C = bestCs[i]
        K_train = compute_kernel_matrix(sequences_train, ssk_kernel, best_k)

        # training 
        
        study = optuna.create_study(direction="maximize")
        study.optimize(partial(objective, sequences_train, labels_train), n_trials=n_trials)

        # Best hyperparameters
        best_k = study.best_params["k"]
        best_C = study.best_params["C"]
        print(f" Best k: {best_k}, Best C: {best_C}")

        svm = KernelSVM(C=best_C, max_iters=1000)
        svm.fit(K_train, labels_train, sequences_train)

        # prediction
        seq_test_file = seq_test_files[i]
        output = svm.predict_file(
            file_path=seq_test_file,
            k=best_k
        )
        outputs.append(output)

    # Merge all results
    final_df = pd.concat(outputs, ignore_index=True)
    final_df.to_csv(y_test_file, index=False)
    print(f"The result is saved in : {y_test_file}")
    t1 = time.time()
    print('done in {:.3f} seconds'.format(t1 - t0))

#xte0
#Best k: 7, Best C: 1.7259685252606225
#precision of kernel 0 : 1.0

#Best k: 6, Best C: 0.013650844380457251
#precision of kernel 1 : 1.0

# Best k: 7, Best C: 0.2166487544857509
#precision of kernel 2 : 1.0

#done in 102945.497 seconds