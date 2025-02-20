import numpy as np
import pandas as pd
import time

#substring kernel
def substring_kernel(s1: str, s2: str, k: int) -> int:
    """
    计算两个DNA序列的substring kernel值
    
    参数:
        s1, s2 (str): 输入的两个DNA序列
        k (int): 子串长度
    
    返回:
        int: kernel值（两个序列k-length子串的共现次数加权和）
    """
    # 预处理：转换为大写并验证DNA字符
    s1 = s1.upper()
    s2 = s2.upper()

    # 检查k的合法性
    if k <= 0 or k > len(s1) or k > len(s2):
        return 0
    
    # 生成子串频次字典
    def count_substrings(s: str) -> dict:
        counts = {}
        for i in range(len(s) - k + 1):
            substr = s[i:i+k]
            counts[substr] = counts.get(substr, 0) + 1
        return counts
    
    count1 = count_substrings(s1)
    count2 = count_substrings(s2)
    
    # 计算点积
    kernel_value = 0
    for substr, cnt1 in count1.items():
        cnt2 = count2.get(substr, 0)
        kernel_value += cnt1 * cnt2
    return kernel_value

#svm
def compute_kernel_matrix(X_train, kernel_func, k):
    """
    计算核矩阵 (训练集内积 或 训练集-测试集间核矩阵)
    
    参数:
        X_train (list): 训练集序列
        X_test (list): 测试集序列 (可选)
        kernel_func: substring_kernel函数
        k: 子串长度参数
    
    返回:
        K_train (numpy数组): 训练集核矩阵 [n_train, n_train]
        K_test (numpy数组): 测试集核矩阵 [n_test, n_train] (如果提供X_test)
    """
    n_train = len(X_train)
    K_train = np.zeros((n_train, n_train))
    for i in range(n_train):
        for j in range(i, n_train):
            val = kernel_func(X_train[i], X_train[j], k)
            K_train[i, j] = K_train[j, i] = val
    
    return K_train

class KernelSVM:
    def __init__(self, C=1.0, max_iters=1000, tol=1e-3):
        self.C = C          # 正则化参数
        self.max_iters = max_iters
        self.tol = tol      # 停止阈值
        self.alpha = None   # 对偶变量
        self.b = 0.0        # 偏置项
        self.X_train = None # 训练序列缓存
    
    def fit(self, K, y, X_train):
        """
        训练核SVM
        参数:
            K: 训练核矩阵 [n_samples, n_samples]
            y: 标签数组 (+1/-1格式)
            X_train: 训练序列列表（用于后续预测）
        """
        self.X_train = X_train
        self.y_train = y
        n_samples = K.shape[0]
        self.alpha = np.zeros(n_samples)
        y = y.astype(np.float32)
        
        # 简化的SMO优化算法
        for _ in range(self.max_iters):
            alpha_prev = np.copy(self.alpha)
            
            for i in range(n_samples):
                # 计算误差
                Ei = (self.alpha * y).dot(K[i]) + self.b - y[i]
                
                # 违反KKT条件时更新
                if (y[i]*Ei < -self.tol and self.alpha[i] < self.C) or \
                   (y[i]*Ei > self.tol and self.alpha[i] > 0):
                    
                    j = np.random.choice(list(range(n_samples)))
                    Ej = (self.alpha * y).dot(K[j]) + self.b - y[j]
                    
                    # 更新alpha_i和alpha_j
                    eta = K[i,i] + K[j,j] - 2*K[i,j]
                    if eta == 0:
                        continue
                        
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    
                    L, H = self._compute_L_H(alpha_i_old, alpha_j_old, y[i], y[j])
                    
                    self.alpha[j] = alpha_j_old + (y[j]*(Ei - Ej))/eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    self.alpha[i] = alpha_i_old + y[i]*y[j]*(alpha_j_old - self.alpha[j])
                    
                    # 更新偏置b
                    b1 = self.b - Ei - y[i]*(self.alpha[i]-alpha_i_old)*K[i,i] - y[j]*(self.alpha[j]-alpha_j_old)*K[i,j]
                    b2 = self.b - Ej - y[i]*(self.alpha[i]-alpha_i_old)*K[i,j] - y[j]*(self.alpha[j]-alpha_j_old)*K[j,j]
                    self.b = (b1 + b2)/2
                    
            # 检查收敛
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
        K_test = self._compute_test_kernel(valid_sequences, k)
        scores = (self.alpha * self.y_train).dot(K_test.T) + self.b
        pred_labels = np.where(scores >= 0, 1, 0)
        true_predictions = sum(p == r for p, r in zip(pred_labels, valid_labels))  # 计算正确预测数
        return  true_predictions / len(valid_labels)

    def predict_file(self, file_path: str, k: int):
        """
        预测多个测试文件并合并结果到单个CSV
        
        参数:
            test_files: 测试文件路径列表（如["xtest.csv", "xtest1.csv", "xtest2.csv"]）
            output_file: 输出文件路径（ytest.csv）
            k: 子串长度参数
        """
        try:
            # 加载单个测试文件
            df_test = pd.read_csv(file_path)
            if {'Id', 'seq'} - set(df_test.columns):
                raise ValueError(f"文件 {file_path} 缺少ID或seq列")
            
            # 执行预测
            test_sequences = df_test['seq'].tolist()
            test_ids = df_test['Id'].tolist()
            
            K_test = self._compute_test_kernel(test_sequences, k)
            scores = (self.alpha * self.y_train).dot(K_test.T) + self.b
            pred_labels = np.where(scores >= 0, 1, 0)
            
            # 暂存结果
            retsult = pd.DataFrame({
                'Id': test_ids,
                'Bound': pred_labels
            })
            
            print(f"已完成 {file_path} 的预测")
            
        except Exception as e:
            print(f"文件 {file_path} 处理失败: {str(e)}")

        return retsult
    
    def _compute_test_kernel(self, test_sequences: list, k: int) -> np.ndarray:
        """ 计算测试集核矩阵 """
        n_test = len(test_sequences)
        n_train = len(self.X_train)
        
        K_test = np.zeros((n_test, n_train))
        for i in range(n_test):
            for j in range(n_train):
                # 添加DNA有效性检查
                K_test[i,j] = substring_kernel(test_sequences[i], self.X_train[j], k)
        return K_test
    
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

def split_data(sequences, labels, split_ratio=0.2, seed=42):
    # 20% label 0 and 20% label 1 in validation set
    np.random.seed(seed)  # 保证结果可复现

    # 分别存储 label=0 和 label=1 的索引
    indices_0 = [i for i, label in enumerate(labels) if label == 0]
    indices_1 = [i for i, label in enumerate(labels) if label == 1]

    # 打乱索引
    np.random.shuffle(indices_0)
    np.random.shuffle(indices_1)

    # 计算划分点
    split_0 = int(len(indices_0) * (1 - split_ratio))
    split_1 = int(len(indices_1) * (1 - split_ratio))

    # 获取训练集和验证集索引
    train_indices = indices_0[:split_0] + indices_1[:split_1]
    valid_indices = indices_0[split_0:] + indices_1[split_1:]

    # 由于索引合并后顺序可能是打乱的，我们再打乱一次确保混合训练
    np.random.shuffle(train_indices)
    np.random.shuffle(valid_indices)

    # 根据索引提取数据
    train_seq = sequences[np.array(train_indices)]
    valid_seq = sequences[np.array(valid_indices)]
    train_label = labels[np.array(train_indices)]
    valid_label = labels[np.array(valid_indices)]
    return train_seq, valid_seq, train_label, valid_label

# 数据准备（示例）
t0 = time.time()
seq_files =["data/Xtr0.csv","data/Xtr1.csv","data/Xtr2.csv"]
label_files = ["data/Ytr0.csv","data/Ytr1.csv","data/Ytr2.csv"]
seq_test_files = ["data/Xte0.csv","data/Xte1.csv","data/Xte2.csv"]
y_test_file = "data/Ytrk.csv"
# 步骤1：计算核矩阵
k = 3  # substring长度参数

outputs=[]
for i in range(len(seq_files)):
    seq_file = seq_files[i]
    label_file = label_files[i]
    sequences_train, labels_train = load_data(seq_file, label_file)
    train_seq, valid_seq, train_label, valid_label =  split_data(sequences_train, labels_train)

    K_train = compute_kernel_matrix(train_seq, substring_kernel, k)

    # training 
    # TODO: OPTUNA
    # find best hyperparametre for C, max iters

    svm = KernelSVM(C=5.0, max_iters=10000)
    svm.fit(K_train, train_label, train_seq)

    # validation
    valid_precision = svm.predict(valid_seq, valid_label, k=k)
    print(f"precision of kernel {i} : {valid_precision}")

    # prediction
    seq_test_file = seq_test_files[i]
    output = svm.predict_file(
        file_path=seq_test_file,
        k=k
    )
    outputs.append(output)

# 合并所有结果

final_df = pd.concat(outputs, ignore_index=True)
final_df.to_csv(y_test_file, index=False)
print(f"合并后的预测结果已保存至 {y_test_file}")
t1 = time.time()
print('done in {:.3f} seconds'.format(t1 - t0))

