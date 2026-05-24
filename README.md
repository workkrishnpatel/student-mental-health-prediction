
# Mental Health Risk Prediction System

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![Library](https://img.shields.io/badge/Library-CatBoost-FF4655.svg?style=flat-square)](https://catboost.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning healthcare analytics pipeline and risk prediction interface designed to evaluate behavioral patterns and identify high-risk clinical depression tendencies among students using structured demographic, academic, and lifestyle vectors.


## Table of Contents

* [Problem Statement](#problem-statement)
* [Features](#features)
* [Technologies Used](#technologies-used)
* [Dataset Information](#dataset-information)
* [Workflow](#workflow)
* [Machine Learning Tournament](#machine-learning-tournament)
* [Model Evaluation](#model-evaluation)
* [Streamlit App Explanation](#streamlit-app-explanation)
* [Installation Steps](#installation-steps)
* [Usage Steps](#usage-steps)
* [Future Improvements](#future-improvements)
* [Conclusion](#conclusion)
* [Author Info](#author-info)


## Problem Statement

Mental health issues among university students often go unobserved due to social stigmas, limited evaluation capacity, and the complex interaction of academic, social, and economic pressures. Identifying early risk patterns before they escalate is crucial for timely institutional support.

This project addresses this issue by translating multi-dimensional lifestyle data into objective risk probabilities. Rather than relying on rigid diagnostic thresholds, the machine learning system benchmarks ensemble classification algorithms to establish an actionable risk index, offering student wellness programs an efficient data-driven preliminary screening mechanism.


## Features

* **Structured Preprocessing Pipeline:** Integrates systematic column normalization, lowercase feature standardizations, and robust categorical mapping pipelines.
* **Exploratory Visual Diagnostics:** Generates clean, dark-themed correlation heatmaps and trend analysis charts tracking cross-dependencies between financial anxiety, academic workload, and sleep cycles.
* **Multi-Model Tournament:** Implements an algorithmic benchmarking leaderboard comparing advanced gradient-boosted decision trees against non-linear baseline models.
* **Feature Importance Analysis:** Utilizes model-level information gain matrices to isolate and highlight primary behavioral risk drivers.
* **Interactive Screening Console:** Deploys a user-friendly Streamlit web interface that calculates immediate depression risk scores and predictive confidence intervals based on real-time parameters.


## Technologies Used

* **Language:** Python 3.13
* **Data Processing:** Pandas, NumPy
* **Machine Learning Tournament:** CatBoost, LightGBM, XGBoost, Scikit-Learn
* **Visualizations:** Matplotlib, Seaborn
* **Model Preservation:** Joblib
* **Deployment System:** Streamlit


## Dataset Information

The predictive modeling layer utilizes the structured **Student Depression Dataset** containing thousands of validated student profile rows.

### Core Features Evaluated

| Feature Name | Data Type | Description / Domain |
| :--- | :--- | :--- |
| `gender` | Categorical | Gender identification of the student |
| `age` | Integer | Chronological age of the student |
| `academic_pressure` | Integer / Ordinal | Subjective rating scale for workload pressure |
| `cgpa` | Float | Cumulative Grade Point Average metric |
| `study_satisfaction` | Integer / Ordinal | Subjective ranking of academic path fulfillment |
| `sleep_duration` | Categorical | Average duration of nightly rest |
| `dietary_habits` | Categorical | Standard classification of nutritional regularity |
| `suicidal_thoughts` | Categorical | Binary indicator of severe psychological distress history |
| `work_study_hours` | Integer | Total weekly hours spent balanced between jobs and study blocks |
| `financial_stress` | Integer / Ordinal | Rated index of economic stress factors |
| `family_history` | Categorical | Documented hereditary history of clinical depression issues |

### Target Mapping Boundaries

The system solves a supervised **Binary Classification** task mapped to the `depression` column string:
* `0` $\rightarrow$ **Low Depression Risk** (Stable behavioral indicators)
* `1` $\rightarrow$ **High Depression Risk** (High-priority distress signals detected)

> **Data Reference:** Clean repository records can be retrieved from the official [Kaggle Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset).


## Workflow

1. **Ingestion & Text Cleaning:** Standardizing column strings into a unified lowercase layout to ensure robust feature routing.
2. **Feature Isolation:** Dropping non-contributing index markers to focus the model exclusively on valid academic and psychological metrics.
3. **Categorical Encoding:** Converting structural text properties into machine-readable numeric formats via specialized encoders.
4. **Data Splitting:** Separating records into isolated training subsets and independent testing vectors using Scikit-Learn's `train_test_split()`.
5. **Multi-Model Execution:** Running the training data across multiple classification architectures under uniform constraints.
6. **Feature Weight Inspection:** Extracting internal tree splits from the champion classifier to identify critical risk variables.
7. **Artifact Export:** Packaging the processing pipeline (`preprocessor.pkl`) and the best performing model weights (`best_model.pkl`) using Joblib serialization.


## Machine Learning Tournament

To guarantee optimal prediction limits, this project rejects singular model assumptions and instead runs a rigorous classification benchmarking tournament:



                  [ Transformed Data Splits ]
                               ↓
    ┌───────────────┬──────────┼───────────────┬──────────────┐
    ↓               ↓          ↓               ↓              ↓



[ CatBoost ]    [ LightGBM ] [ XGBoost ] [ Random Forest ] [ Decision Tree ]
│               │          │               │              │
└───────────────┴──────────┼───────────────┴──────────────┘
↓
[ Metric Leaderboard Ranking ]
↓
★ Champion Selection: CatBoost (84.55%)


### Final Model Leaderboard

| Tournament Rank | Algorithmic Pipeline | Test Accuracy Score | Status |
| :---: | :--- | :---: | :--- |
| **1st** | **CatBoost Classifier** | **84.55%** | **Selected Champion** |
| 2nd | LightGBM Classifier | 84.12% | Contender |
| 3rd | XGBoost Classifier | 83.55% | Contender |
| 4th | Random Forest Classifier | 83.39% | Baseline Ensemble |
| 5th | Decision Tree Classifier | 76.00% | Structural Baseline |


## Model Evaluation

System validation relies on multiple metrics to evaluate the classification boundaries of the champion CatBoost engine:

* **Accuracy Verification:** Evaluates global classification success across out-of-sample data points.
* **Classification Reports:** Monitors precision, recall, and F1-score balances to prevent high false-negative mistakes.
* **Confusion Matrices:** Provides a visual heatmap of correct designations against false positive and false negative distribution rates.
* **Feature Importance Vectoring:** Ranks feature weights to understand the underlying choices driving the network's predictions.

### Primary Risk Drivers Identified
1. `suicidal_thoughts` (Highest predictive weight)
2. `academic_pressure`
3. `financial_stress`
4. `age`


## Streamlit App Explanation

The web portal turns complex ensemble probabilities into an accessible diagnostic dashboard for educational administrators and wellness counselors:

* **Interactive User Input Form:** Collects categorical variables and sliding scale metrics (CGPA, financial stress, sleep metrics) through a clean UI layout.
* **Real-Time Prediction Engine:** Processes input parameters through `preprocessor.pkl` and evaluates them using `best_model.pkl` instantly.
* **Dynamic Risk & Confidence Output:** Displays prominent status alerts marking the student profile as either **High Depression Risk** or **Low Depression Risk**, supplemented by an explicit model confidence percentage.


## Installation Steps

1. Clone the project repository to your local architecture:
```bash
git clone [https://github.com/workkrishnpatel/mental-health-risk-prediction.git](https://github.com/workkrishnpatel/mental-health-risk-prediction.git)
cd mental-health-risk-prediction

```

2. Establish an isolated Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows terminals use: venv\Scripts\activate

```

3. Populate required dependency libraries within your local runtime:

```bash
pip install -r requirements.txt

```



## Usage Steps

### Running the Model Training Pipeline

To inspect data preparation steps, execute dark-themed visual EDA code blocks, and evaluate the classification tournament results, run the Jupyter environment file:

```bash
jupyter notebook 4_ml_model_training.ipynb

```

### Initializing the Streamlit Deployment App

To launch the interactive local prediction server and open the web portal layout, execute the core app script:

```bash
streamlit run app.py

```



## Future Improvements

* **Hyperparameter Optimization:** Integrate extensive Optuna search spaces to fine-tune CatBoost regularization parameters and depth limits.
* **SHAP Interpretability Maps:** Implement SHAP (Shapley Additive exPlanations) visualizations within the Streamlit UI to show users exactly how individual inputs alter their specific risk trajectory.
* **Multi-Class Risk Tiering:** Expand the target framework from simple binary labels into granular operational categories: Minimal, Moderate, Severe Risk profiles.



## Conclusion

This project presents a robust, data-driven approach to preliminary mental health screening within educational settings. By leveraging advanced gradient boosting via CatBoost and managing complex lifestyle interactions, the system successfully extracts clear risk indicators from student survey data. While not a replacement for formal clinical diagnoses, the pipeline provides a scalable screening solution to help wellness programs identify vulnerable students early.


## Author Info

* **Author:** Krishn Patel
* **GitHub Profile:** [workkrishnpatel](https://www.google.com/search?q=https://github.com/workkrishnpatel)
* **Email:** work.krishnpatel@gmail.com

