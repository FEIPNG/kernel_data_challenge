import numpy as np
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import cvxopt
import numpy as np
from collections import defaultdict
import hashlib
from functools import partial


def sequence_to_kmer_graph(seq, k=3):
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    unique_kmers = list(set(kmers))
    node_id = {kmer: i for i, kmer in enumerate(unique_kmers)}  
    edges = defaultdict(list)
    for i in range(len(kmers)-1):
        curr = kmers[i]
        next_kmer = kmers[i+1]
        if curr in node_id and next_kmer in node_id:
            edges[node_id[curr]].append(node_id[next_kmer])
        else:
            print(f"Error abnormal neighbor: {curr}->{next_kmer}")
    return dict(edges), unique_kmers

def wl_kernel(graph1, graph2, h=3):
    edges1, labels1 = graph1
    edges2, labels2 = graph2
    kernel_value = 0

    current_labels1 = labels1.copy()
    current_labels2 = labels2.copy()

    common = set(current_labels1) & set(current_labels2)
    kernel_value += sum(current_labels1.count(lbl) * current_labels2.count(lbl) for lbl in common)

    for _ in range(h):
        
        new_labels1 = []
        for node in range(len(current_labels1)):
            neighbors = edges1.get(node, [])
            neighbor_labels = [current_labels1[nbr] for nbr in neighbors if nbr in edges1]
            new_label = hash(tuple([current_labels1[node]] + sorted(neighbor_labels)))
            new_labels1.append(new_label)
        
        new_labels2 = []
        for node in range(len(current_labels2)):
            neighbors = edges2.get(node, [])
            neighbor_labels = [current_labels2[nbr] for nbr in neighbors if nbr in edges2]
            new_label = hash(tuple([current_labels2[node]] + sorted(neighbor_labels)))
            new_labels2.append(new_label)
        
        count1 = defaultdict(int)
        count2 = defaultdict(int)
        for lbl in new_labels1:
            count1[lbl] += 1
        for lbl in new_labels2:
            count2[lbl] += 1
        common = set(count1.keys()) & set(count2.keys())
        kernel_value += sum(count1[lbl] * count2[lbl] for lbl in common)
        
        current_labels1 = new_labels1
        current_labels2 = new_labels2
    
    return kernel_value

def compute_wl_kernel_for_pair(args):
    i, j, graphs1, graphs2, h = args
    return i, j, wl_kernel(graphs1[i], graphs2[j], h=h)

def compute_wl_kernel_matrix(train_graphs, val_graphs=None, h=3, num_workers=8):
    if val_graphs is None:
        n = len(train_graphs)
        K = np.zeros((n, n))
        tasks = [(i, j, train_graphs, train_graphs, h) 
                for i in range(n) for j in range(i, n)]  
    else:
        n_val = len(val_graphs)
        n_train = len(train_graphs)
        K = np.zeros((n_val, n_train))
        tasks = [(i, j, val_graphs, train_graphs, h) 
                for i in range(n_val) for j in range(n_train)]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(compute_wl_kernel_for_pair, task) for task in tasks]
        for future in as_completed(futures):
            try:
                i, j, score = future.result()
                K[i][j] = score
            except KeyError as e:
                print(f"Error graphe ({i},{j}) , error message: {str(e)}")
                K[i][j] = 0 
    
    if val_graphs is None:
        np.fill_diagonal(K, K.diagonal() + 1e-5)
    return K

def validate_graph(graph):
    edges, labels = graph
    max_node_id = len(labels) - 1
    for node in edges:
        assert node <= max_node_id, f"error id : {node}， max node : {max_node_id}"
        for neighbor in edges[node]:
            assert neighbor <= max_node_id, f"error neighbor id : {neighbor}"

# svm
def train_svm(K, y, C=1.0, tol=1e-4, max_iter=1000):
    # check K matrices
    eigenvalues = np.linalg.eigvalsh(K)
    print("min eigenvalues:", np.min(eigenvalues)) 
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


if __name__ == '__main__':
    t0 = time.time()
    # files
    seq_files =["data/Xtr0.csv","data/Xtr1.csv","data/Xtr2.csv"]
    label_files = ["data/Ytr0.csv","data/Ytr1.csv","data/Ytr2.csv"]
    seq_test_files = ["data/Xte0.csv","data/Xte1.csv","data/Xte2.csv"]
    y_test_file = "data/Ytrk_kl.csv"
    outputs=[]
    bestCs = [1, 1, 1]
    bestks = [5,5,5]
    besths = [3,3,3]
    for i in range(len(seq_files)):
        # load file
        seq_file = seq_files[i]
        label_file = label_files[i]
        sequences_train, labels_train = load_data(seq_file, label_file)
        best_k = bestks[i]
        best_C = bestCs[i]
        best_h = besths[i]
        # create graphes
        train_graphs = [sequence_to_kmer_graph(s, k=best_k) for s in sequences_train]
        # check graphes
        for g in train_graphs:
            validate_graph(g)
        # calculate K
        K_train = compute_wl_kernel_matrix(train_graphs, h=best_h)
        # SVM
        alphas, support_vectors, bias = train_svm(K_train, labels_train, best_C)

        # predict for test file
        # load file
        seq_test_file = seq_test_files[i]
        df_test = pd.read_csv(seq_test_file)
        if {'Id', 'seq'} - set(df_test.columns):
            raise ValueError(f"file {seq_test_file} don't have the ID column")
        test_sequences = df_test['seq'].tolist()
        test_ids = df_test['Id'].tolist()
        
        # generate graph
        test_graphs = [sequence_to_kmer_graph(s, k=3) for s in test_sequences]     
        # K   
        K_test = compute_wl_kernel_matrix(test_graphs, train_graphs, h=3)
        # check graph
        for g in test_graphs:
            validate_graph(g)
        # predict 
        support_y = labels_train[support_vectors].flatten()  
        scores = predict_svm(K_test , alphas, support_vectors, support_y, bias, score_type=True)
        probs = 1 / (1 + np.exp(-scores)) 
        pred_labels = (probs > 0.5).astype(int)
        retsult = pd.DataFrame({
            'Id': test_ids,
            'Bound': pred_labels
        })
        
        print(f"Finished : {seq_test_file} ")
        outputs.append(retsult)

    # merge all results
    final_df = pd.concat(outputs, ignore_index=True)
    final_df.to_csv(y_test_file, index=False)
    print(f"The result is saved in {y_test_file}")
    t1 = time.time()
    print('done in {:.3f} seconds'.format(t1 - t0))