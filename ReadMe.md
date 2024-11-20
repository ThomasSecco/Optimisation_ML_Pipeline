# 🚀 Optimisation d’un Pipeline de Machine Learning 🎯

Welcome to **Optimisation d’un Pipeline de Machine Learning**, a comprehensive project showcasing the end-to-end process of building, optimizing, and validating a machine learning pipeline! 🧠💻

---

## 🗂️ Project Overview

In this project, we use the **Wine Quality Dataset** 🍷 to:
1. **Understand the data** through exploratory data analysis (EDA). 📊
2. **Preprocess the dataset** for machine learning. 🛠️
3. **Build a robust pipeline** using `scikit-learn`. ⚙️
4. **Optimize hyperparameters** to maximize performance. 🔍
5. **Evaluate the final model** on a test dataset. ✅

---

## 📁 Repository Structure

```plaintext
Optimisation_ML_Pipeline/
├── data/                    # Dataset and processed files
│   ├── winequality-red.csv  # Raw dataset
│   ├── processed_data.csv   # Cleaned and preprocessed data
├── notebooks/               # Jupyter notebooks
│   ├── EDA.ipynb            # Exploratory Data Analysis
│   ├── Pipeline_Optimization.ipynb # Pipeline and tuning
├── src/                     # Source code
│   ├── data_preprocessing.py # Data preprocessing script
│   ├── model_pipeline.py     # Model pipeline construction
│   ├── hyperparameter_tuning.py # Hyperparameter optimization
│   ├── cross_validation.py   # Cross-validation script
├── results/                 # Results and outputs
│   ├── model_performance.json # Final model performance
├── tests/                   # Unit tests
│   ├── test_data_preprocessing.py
│   ├── test_model_pipeline.py
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
```

## 📊 Dataset
The Wine Quality Dataset 🍷 is sourced from the UCI Machine Learning Repository(https://archive.ics.uci.edu/dataset/186/wine+quality). It contains chemical analysis results of red wine samples, along with a quality rating (0–10). The dataset includes:

- 11 Features: e.g., acidity, sugar, pH, alcohol.
- 1 Target Variable: Wine quality (integer score).

## 🚀 How to Get Started
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ThomasSecco/Optimisation_ML_Pipeline.git
cd Optimisation_ML_Pipeline
```

### 2️⃣ Install Dependencies
Make sure you have Python 3.8+ installed, then run:

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Preprocessing Script
Prepare the data for machine learning:

```bash
python src/data_preprocessing.py
```

### 4️⃣ Run the Model Pipeline
Train and evaluate the machine learning pipeline:

```bash
python src/model_pipeline.py
```


### 5️⃣ Optimize Hyperparameters
Boost performance with hyperparameter tuning:

```bash
python src/hyperparameter_tuning.py
```

## 🧠 Key Features
- End-to-End Workflow: From data exploration to final evaluation.
- Optimized Pipeline: Uses GridSearchCV for hyperparameter tuning. 🔍
- Automated Cross-Validation: Ensures robust model evaluation. ✅
- Well-Structured Codebase: Modular and easy to extend. 🛠️
- Beautiful Visualizations: Insightful plots for EDA. 📈



## 💡 Learnings
Through this project, we:

- Understood the importance of proper data preprocessing. 🧹
- Learned to construct and optimize machine learning pipelines. 🔧
- Practiced systematic evaluation using cross-validation. 🎯

## 🤝 Contributions
Contributions are welcome! Feel free to fork this repository, raise issues, or submit pull requests. 🙌