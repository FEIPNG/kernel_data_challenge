import numpy as np
import pandas as pd

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

    def predict_multiple_files(self, test_files: list, output_file: str, k: int):
        """
        预测多个测试文件并合并结果到单个CSV
        
        参数:
            test_files: 测试文件路径列表（如["xtest.csv", "xtest1.csv", "xtest2.csv"]）
            output_file: 输出文件路径（ytest.csv）
            k: 子串长度参数
        """
        all_results = []
        
        for file_path in test_files:
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
                all_results.append(pd.DataFrame({
                    'Id': test_ids,
                    'Bound': pred_labels
                }))
                
                print(f"已完成 {file_path} 的预测")
                
            except Exception as e:
                print(f"文件 {file_path} 处理失败: {str(e)}")
                continue
        
        # 合并所有结果
        final_df = pd.concat(all_results, ignore_index=True)
    
        # 保存结果
        final_df.to_csv(output_file, index=False)
        print(f"合并后的预测结果已保存至 {output_file}")

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
    
def load_multiple_data(seq_files: list, label_files: list) -> tuple:
    """
    加载多组CSV文件并合并数据
    
    参数:
        seq_files: 多个x.csv文件路径列表（如["x.csv", "x1.csv"]）
        label_files: 多个y.csv文件路径列表（如["y.csv", "y1.csv"]）
    
    返回:
        (sequences, labels): 合并后的序列列表和标签数组
    """
    # 检查文件数量匹配
    if len(seq_files) != len(label_files):
        raise ValueError("seq_files和label_files的数量必须相同")
    
    # 合并所有数据
    merged_dfs = []
    for seq_file, label_file in zip(seq_files, label_files):
        # 读取单个文件对
        df_seq = pd.read_csv(seq_file)
        df_label = pd.read_csv(label_file)
        
        # 合并单个文件对
        merged = pd.merge(df_seq, df_label, on='Id', how='inner')
        if merged.empty:
            print(f"警告: {seq_file}和{label_file}中没有匹配的ID")
            continue
            
        merged_dfs.append(merged)
    
    # 合并所有数据
    full_df = pd.concat(merged_dfs, ignore_index=True)

    # 提取数据
    sequences = full_df['seq'].tolist()
    labels = np.where(full_df['Bound'] == 1, 1, -1)
    
    return sequences, labels

# 数据准备（示例）

seq_files =["data/Xtr0.csv","data/Xtr1.csv","data/Xtr2.csv"]
label_files = ["data/Ytr0.csv","data/Ytr1.csv","data/Ytr2.csv"]
seq_test_files = ["data/Xte0.csv","data/Xte1.csv","data/Xte2.csv"]
y_test_file = "data/Ytrk.csv"
sequences_train, labels_train = load_multiple_data(seq_files, label_files)
# 步骤1：计算核矩阵
k = 3  # substring长度参数
K_train = compute_kernel_matrix(sequences_train, substring_kernel, k)

# 步骤2：训练核SVM
svm = KernelSVM(C=1.0, max_iters=1000)
svm.fit(K_train, labels_train, sequences_train)

# 步骤3：预测
svm.predict_multiple_files(
    test_files=seq_test_files,
    output_file=y_test_file,
    k=k
)