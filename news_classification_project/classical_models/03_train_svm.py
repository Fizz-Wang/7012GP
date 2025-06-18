# =================================================================================================
# 【中文总体说明】
#
# 大家好，这个Python文件的核心任务是：针对文件中【超参数配置】部分指定的“一组”参数，
# 完整地执行一次模型的训练、评估和保存。
#
# 你们的主要工作流程如下：
# 1. 填写模型信息：在【关键任务 A】部分，填写你们负责的具体模型名称和类。
# 2. 修改参数：在【关键任务 B】部分，手动设定你这一次想要实验的参数值。
# 3. 运行脚本：保存文件，然后运行一次本脚本。
# 4. 查看结果：脚本运行完毕后，会生成一个模型文件(.joblib)和一个报告文件(.txt)。
# 5. 重复实验：要测试另一组参数，只需回到本文件，再次修改【超参数配置】部分的数值，
#              然后再次运行脚本。每次运行都会生成新的结果文件。
# 6. 分析决策：当你们进行了足够多的实验后，通过比较所有生成的.txt报告来找出最佳模型。
#
# =================================================================================================
# FILE: 03_train_svm.py
# TEAM-MEMBER: Shibo
# =================================================================================================

# --- Step 1: Import Necessary Libraries ---
# Ensure you have all these libraries installed from your requirements.txt.
import os
import datetime
import pickle
import joblib
import numpy as np
from pathlib import Path
from sklearn.svm import SVC  # TODO: [关键任务 A.1] 从sklearn导入你负责的模型类
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

# --- Step 2: [CRITICAL TASK A] Define Model Configuration ---
# TODO: [关键任务 A.2] 填写你负责的模型名称，使用小写和下划线
MODEL_NAME = 'svm'

# --- Step 3: [CRITICAL TASK B] Configure Hyperparameters for This Specific Run ---
# 为你的模型定义一组参数。每次实验，你都需要手动修改这里的数值。
# 注意：对于大数据集，linear核函数比rbf核函数训练速度快很多
params_for_this_run = {
    'C': 1.0,            # 提高C值改善性能 (从0.1调回1.0)
    'kernel': 'linear',  # Kernel type: 'linear' or 'rbf' (linear更快)
    'probability': True,  # 启用概率估计以计算AUC分数
    'max_iter': 2000,    # 增加迭代次数确保收敛 (从500增加到2000)
    'class_weight': 'balanced'  # 处理类别不平衡问题
}

# 性能优化配置 - 为提高准确率调整
USE_DATA_SAMPLING = True    # 启用数据采样加速训练
SAMPLE_SIZE = 15000        # 增加采样大小提高性能 (从5000增加到15000)
REDUCE_FEATURES = True     # 启用特征降维
MAX_FEATURES = 20000       # 增加特征数量提高性能 (从10000增加到20000)
USE_STANDARDIZATION = True # 启用数据标准化

# --- Step 4: Setup Paths ---
# This part uses MODEL_NAME to create the correct folder for the results.
try:
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate one level up to get to the project root ('news_classification_project')
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    # Define the path to the main 'results' folder within the project root
    RESULTS_PATH = Path(os.path.join(project_root, 'results'))
    MODEL_RESULTS_PATH = RESULTS_PATH / MODEL_NAME
    MODEL_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"Error creating directories: {e}")
    exit(1)

# --- Step 5: Load the Preprocessed Data ---
# This step loads the data that was prepared by the preprocessing script.
try:
    print("Loading preprocessed data...")
    data_path = RESULTS_PATH / 'processed_data.pkl'
    with open(data_path, 'rb') as f:
        processed_data = pickle.load(f)
    X_train_tfidf = processed_data['X_train_tfidf']  # training features
    X_test_tfidf  = processed_data['X_test_tfidf']   # test features
    y_train       = processed_data['y_train']        # training labels
    y_test        = processed_data['y_test']         # test labels
    
    print(f"Data loading completed - Training set: {X_train_tfidf.shape}, test set: {X_test_tfidf.shape}")
    
    # 特征降维优化
    if REDUCE_FEATURES and X_train_tfidf.shape[1] > MAX_FEATURES:
        from sklearn.feature_selection import SelectKBest, chi2
        selector = SelectKBest(chi2, k=MAX_FEATURES)
        X_train_tfidf = selector.fit_transform(X_train_tfidf, y_train)
        X_test_tfidf = selector.transform(X_test_tfidf)
        print(f"feature dimension reduction: {X_train_tfidf.shape[1]}dimension")
    
    # 数据采样优化
    if USE_DATA_SAMPLING and X_train_tfidf.shape[0] > SAMPLE_SIZE:
        from sklearn.utils import resample
        X_train_tfidf, y_train = resample(X_train_tfidf, y_train, 
                                         n_samples=SAMPLE_SIZE, 
                                         random_state=42, 
                                         stratify=y_train)
        print(f"data sampling: {X_train_tfidf.shape[0]}sample")
    
    # 数据标准化优化
    if USE_STANDARDIZATION:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(with_mean=False)  # 稀疏矩阵不能计算均值
        X_train_tfidf = scaler.fit_transform(X_train_tfidf)
        X_test_tfidf = scaler.transform(X_test_tfidf)
        print(f"Data standardization has been completed.")
        
except FileNotFoundError:
    print(f"Error: 'processed_data.pkl' not found at {data_path}. Please run the preprocessing script first.")
    exit(1)
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# --- Step 6: Train, Evaluate, and Save the Model ---
# This code will execute ONCE using the parameters you defined in Step 3.
print(f"\nstart SVM training - parameter: C={params_for_this_run['C']}, kernel={params_for_this_run['kernel']}, probability={params_for_this_run['probability']}")

# 6a: Instantiate the model
model = SVC(**params_for_this_run, random_state=42)

# 6a: Train the model
print("Training model...")

import time
start_time = time.time()
model.fit(X_train_tfidf, y_train)
end_time = time.time()
training_time = end_time - start_time

print(f"Model training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

# 6b: Evaluate the model on the test set
print("Evaluating the model...")
y_pred = model.predict(X_test_tfidf)
# 6b: Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
# 6b: Compute AUC if probability is enabled
auc_score = None
if params_for_this_run.get('probability', False):
    y_pred_proba = model.predict_proba(X_test_tfidf)
    auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='weighted')
    print(f"AUC Score: {auc_score:.4f}")
else:
    print("AUC  calculation has been skipped. (probability=False)")
# 6b: Generate classification report and confusion matrix
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nClassification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)

# 6c: Generate filenames and save the results
param_string = '_'.join(f"{key}-{value}" for key, value in params_for_this_run.items())
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
base_filename = f"{MODEL_NAME}_{param_string}_{timestamp}"
model_path = MODEL_RESULTS_PATH / f"{base_filename}.joblib"
report_path = MODEL_RESULTS_PATH / f"{base_filename}.txt"
# 6c: Save the trained model
joblib.dump(model, model_path)
print(f"The model has been saved: {model_path.name}")
# 6c: Save the evaluation report
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(f"Parameters: {params_for_this_run}\n")
    f.write(f"Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)\n")
    f.write(f"Training Set Size: {X_train_tfidf.shape}\n")
    f.write(f"Test Set Size: {X_test_tfidf.shape}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    if auc_score is not None:
        f.write(f"AUC Score (weighted ovo): {auc_score:.4f}\n")
    else:
        f.write(f"AUC Score (weighted ovo): N/A (probability disabled)\n")
    f.write("\nClassification Report:\n")
    f.write(report + "\n")
    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix) + "\n")
print(f"The assessment report has been saved: {report_path.name}")

print("\nTraining Finished.")
