# =================================================================================================
#
#                                       GENERAL INSTRUCTIONS
#
# This script trains, evaluates, and saves a Logistic Regression model based on the
# hyperparameters defined in "CRITICAL TASK B".
# To run another experiment, simply change the parameter values and re-run the script.
#
# =================================================================================================
# FILE: 02_train_logistic.py
#
# TEAM-MEMBER: [Your Name Here]
#
# =================================================================================================


# --- Step 1: Import Necessary Libraries ---
# Ensure you have all these libraries installed from your requirements.txt.
import os
import datetime
import pickle
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

# --- Step 2: Define Model Configuration ---
# This name will be used for creating folders and filenames.
MODEL_NAME = 'logistic_regression'

# --- Step 3: [CRITICAL TASK] Configure Hyperparameters for This Specific Run ---
# ========================================================================================
# !! 你的主要任务: 手动修改下面的参数字典来进行实验 !!
#
# 每次实验前，请修改这里的数值，然后保存并运行脚本。
# ========================================================================================
params_for_this_run = {
    # --- Logistic Regression Parameters ---

    # 'C': 正则化强度的倒数。这是一个非常重要的参数！
    #      - 较小的值 (如 0.01, 0.1) 表示更强的正则化，可以帮助防止模型过拟合，但可能导致模型过于简单。
    #      - 较大的值 (如 10.0, 100.0) 表示更弱的正则化，模型会更努力地拟合训练数据，但有过拟合的风险。
    #      - 建议尝试的值: [0.1, 1.0, 10.0, 50.0, 100.0]
    'C': 10.0,

    # 'penalty': 正则化的类型。
    #      - 'l2': 最常用的正则化项（Ridge），会使模型权重趋向于较小的值，但不会变为0。
    #      - 'l1': 也可以尝试（Lasso），它会产生稀疏解，即让一些不重要的特征的权重变为0，有特征选择的效果。
    #      - 注意: 不同的 'solver' 支持不同的 'penalty'。
    'penalty': 'l2',

    # 'solver': 用于优化的算法。
    #      - 'liblinear': 对于中小型数据集来说是一个很好的选择，同时支持 'l1' 和 'l2' 正则化。
    #      - 'saga': 对于大型数据集更有效，也支持 'l1' 和 'l2'。
    #      - 'lbfgs': 是默认的求解器之一，但只支持 'l2' 正则化。
    'solver': 'liblinear',

    # 'max_iter': 算法收敛的最大迭代次数。
    #      - 如果你看到关于 "convergence" 的警告，可以尝试增加这个值，比如 2000 或 5000。
    'max_iter': 1000,
}


def main():
    """Main function to run the training and evaluation process."""
    # --- Step 4: Setup Paths ---
    # This section sets up paths dynamically to ensure the script runs correctly
    # regardless of where it is executed from.
    try:
        # Get the absolute path of the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate one level up to get to the project root ('news_classification_project')
        project_root = os.path.abspath(os.path.join(script_dir, '..'))

        # Define the path to the main 'results' folder within the project root
        results_path = os.path.join(project_root, 'results')

        # Define the path for this specific model's results
        model_results_path = os.path.join(results_path, MODEL_NAME)

        # Create the results directory for this model if it doesn't exist.
        os.makedirs(model_results_path, exist_ok=True)
    except Exception as e:
        print(f"Error creating directories: {e}")
        return

    # --- Step 5: Load the Preprocessed Data ---
    # This step loads the data that was prepared by the preprocessing script.
    try:
        print("Loading preprocessed data from 'processed_data.pkl'...")
        data_path = os.path.join(results_path, 'processed_data.pkl')
        with open(data_path, 'rb') as f:
            processed_data = pickle.load(f)

        X_train_tfidf = processed_data['X_train_tfidf']
        X_test_tfidf = processed_data['X_test_tfidf']
        y_train = processed_data['y_train']
        y_test = processed_data['y_test']

        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: 'processed_data.pkl' not found at {data_path}. Please run the preprocessing script first.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- Step 6: Train, Evaluate, and Save the Model ---
    # This code will execute ONCE using the parameters you defined in Step 3.
    print(f"\n{'=' * 25}\nStarting Experiment with Parameters: {params_for_this_run}\n{'=' * 25}")

    # --- 6a: Instantiate and Train the Model ---
    # 'random_state=42' ensures reproducibility.
    model = LogisticRegression(**params_for_this_run, random_state=42)

    # Train the model on the training data.
    print("Training the model...")
    model.fit(X_train_tfidf, y_train)
    print("Model training complete.")

    # --- 6b: Evaluate the Model on the Test Set ---
    print("Evaluating the model...")
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)

    # Calculate all required performance metrics.
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='weighted')
    full_report_str = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print("Classification Report:\n", full_report_str)

    # --- 6c: Generate Filenames and Save the Results ---
    # Create a clean string from the parameters dictionary for the filename.
    param_string = '_'.join([f"{key}-{value}" for key, value in params_for_this_run.items()])
    # Get the current timestamp to ensure the filename is always unique.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Combine all parts into a unique base filename.
    base_filename = f"{MODEL_NAME}_{param_string}_{timestamp}"

    # Define the full paths for the model (.joblib) and the report (.txt).
    model_path = os.path.join(model_results_path, f"{base_filename}.joblib")
    report_path = os.path.join(model_results_path, f"{base_filename}.txt")

    # Save the trained model object.
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # Prepare the detailed text report.
    report_string = f"""
===============================================================
EVALUATION REPORT: {MODEL_NAME.upper()}
FILENAME: {base_filename}.txt
===============================================================
Timestamp: {timestamp}

--- Hyperparameters Used ---
{params_for_this_run}

--- Performance Metrics ---
Accuracy: {accuracy:.4f}
AUC Score (Weighted, One-vs-One): {auc_score:.4f}

--- Classification Report ---
{full_report_str}

--- Confusion Matrix ---
{str(conf_matrix)}
===============================================================
"""

    # Write the report string to its dedicated .txt file.
    with open(report_path, 'w') as f:
        f.write(report_string)
    print(f"Evaluation report saved to: {report_path}")

    print("\nExperiment run has finished successfully.")


if __name__ == '__main__':
    main()
