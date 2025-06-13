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
# FILE: [Your Script Filename e.g., 02_train_logistic.py]
#
# TEAM-MEMBER: [Your Name Here]
#
# =================================================================================================


# --- Step 1: Import Necessary Libraries ---
# Ensure you have all these libraries installed from your requirements.txt.
"""
import os
import datetime
import pickle
import joblib
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

# TODO: [关键任务 A.1] 从sklearn导入你负责的模型类
# 例如: from sklearn.linear_model import LogisticRegression
# 例如: from sklearn.svm import SVC
# 例如: from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import [YOUR_MODEL_CLASS_HERE]
"""


# --- Step 2: [CRITICAL TASK A] Define Model Configuration ---
# ========================================================================================
# !! 你的任务: 填写下面的模型名称 !!
# ========================================================================================
"""
# TODO: [关键任务 A.2] 将此变量更改为你负责的模型的小写名称
# 这个名字会用在文件夹和文件名中，请使用下划线命名法 (e.g., 'logistic_regression').
MODEL_NAME = '[your_model_name_e.g.,_logistic_regression]'
"""


# --- Step 3: [CRITICAL TASK B] Configure Hyperparameters for This Specific Run ---
# ========================================================================================
# !! 你的任务: 手动修改下面的参数字典 !!
#
# 为你的模型定义一组参数。每次实验，你都需要手动修改这里的数值，然后保存并运行脚本。
# 下面提供了一些常见模型的参数示例，请只保留和你模型相关的部分，并进行修改。
# ========================================================================================
"""
# params_for_this_run = {
#     # --- 示例: Logistic Regression ---
#     # 'C': 10.0,
#     # 'penalty': 'l2',
#     # 'solver': 'liblinear',
#     # 'max_iter': 1000,
#
#     # --- 示例: Support Vector Machine (SVM) ---
#     # 'C': 1.0,
#     # 'kernel': 'linear',
#     # 'probability': True, # Set to True to enable predict_proba for AUC
#
#     # --- 示例: Decision Tree ---
#     # 'criterion': 'gini',
#     # 'max_depth': 20,
#     # 'min_samples_leaf': 5,
# }
"""


# --- Step 4: Setup Paths ---
# This part uses the MODEL_NAME you defined in Step 2 to create the correct folder for the results.
"""
RESULTS_PATH = '../../results/'
MODEL_RESULTS_PATH = os.path.join(RESULTS_PATH, MODEL_NAME)

# Create the results directory for this model if it doesn't exist.
# os.makedirs(MODEL_RESULTS_PATH, exist_ok=True)
"""


# --- Step 5: Load the Preprocessed Data ---
# This step loads the data that was prepared by the preprocessing script.
"""
print("Loading preprocessed data from 'processed_data.pkl'...")
data_path = os.path.join(RESULTS_PATH, 'processed_data.pkl')
with open(data_path, 'rb') as f:
    processed_data = pickle.load(f)

X_train_tfidf = processed_data['X_train_tfidf']
X_test_tfidf = processed_data['X_test_tfidf']
y_train = processed_data['y_train']
y_test = processed_data['y_test']
tfidf_vectorizer = processed_data['tfidf_vectorizer']

print("Data loaded successfully.")
"""


# --- Step 6: Train, Evaluate, and Save the Model ---
# This code will execute ONCE using the parameters you defined in Step 3.
"""
print(f"\n{'='*25}\nStarting Experiment with Parameters: {params_for_this_run}\n{'='*25}")

# --- 6a: Instantiate and Train the Model ---
# TODO: [关键任务 A.3] 用你的模型类实例化模型
# 'random_state=42' ensures reproducibility.
# model = [YOUR_MODEL_CLASS_HERE](**params_for_this_run, random_state=42)

# Train the model on the training data.
# print("Training the model...")
# model.fit(X_train_tfidf, y_train)
# print("Model training complete.")

# --- 6b: Evaluate the Model on the Test Set ---
# print("Evaluating the model...")
# y_pred = model.predict(X_test_tfidf)
# y_pred_proba = model.predict_proba(X_test_tfidf)

# Calculate all required performance metrics.
# accuracy = accuracy_score(y_test, y_pred)
# auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='weighted')
# full_report_str = classification_report(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)

# print(f"Accuracy: {accuracy:.4f}")
# print(f"AUC Score: {auc_score:.4f}")
# print("Classification Report:\n", full_report_str)

# --- 6c: Generate Filenames and Save the Results ---
# Create a clean string from the parameters dictionary for the filename.
# param_string = '_'.join([f"{key}-{value}" for key, value in params_for_this_run.items()])
# Get the current timestamp to ensure the filename is always unique.
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Combine all parts into a unique base filename.
# base_filename = f"{MODEL_NAME}_{param_string}_{timestamp}"

# Define the full paths for the model (.joblib) and the report (.txt).
# model_path = os.path.join(MODEL_RESULTS_PATH, f"{base_filename}.joblib")
# report_path = os.path.join(MODEL_RESULTS_PATH, f"{base_filename}.txt")

# Save the trained model object.
# joblib.dump(model, model_path)
# print(f"Model saved to: {model_path}")

# Prepare the detailed text report.
# report_string = f"""
# ===============================================================
# EVALUATION REPORT: {MODEL_NAME.upper()}
# FILENAME: {base_filename}.txt
# ===============================================================
# Timestamp: {timestamp}
#
# --- Hyperparameters Used ---
# {params_for_this_run}
#
# --- Performance Metrics ---
# Accuracy: {accuracy:.4f}
# AUC Score (Weighted, One-vs-One): {auc_score:.4f}
#
# --- Classification Report ---
# {full_report_str}
#
# --- Confusion Matrix ---
# {str(conf_matrix)}
# ===============================================================
# """

# Write the report string to its dedicated .txt file.
# with open(report_path, 'w') as f:
#     f.write(report_string)
# print(f"Evaluation report saved to: {report_path}")

print("\nExperiment run has finished successfully.")

# --- Step 7: Final Analysis (Your Manual Task) ---
# After running this script multiple times with different parameters:
# 1. Go to the 'news_classification_project/results/[your_model_name]/' folder.
# 2. You will see a list of model (.joblib) and report (.txt) files.
# 3. The filenames clearly state which parameters were used for each run.
# 4. Open the '.txt' reports and compare their performance metrics.
# 5. In your final project report, state which set of hyperparameters was chosen
#    as the "best" and justify your decision with the data from these reports.
#
# if __name__ == '__main__':
#     # The logic to run the experiment would be placed here.
#     # This template is for guidance, so the main execution block is left for you to implement.
#     print("Framework script loaded. Please complete the [CRITICAL TASK] sections and then uncomment the code in Step 6 to run an experiment.")