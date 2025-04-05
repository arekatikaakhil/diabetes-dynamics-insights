# Diabetes Dynamics: A Data Analytics Approach

📊 A data-driven project analyzing the behavioral, clinical, and socio-demographic factors influencing diabetes risk, using advanced big data and machine learning tools.

---

## 🧠 Project Summary

This project explores and models diabetes risk factors to uncover insights that support more effective and equitable healthcare interventions. By leveraging a large health indicators dataset and advanced ML algorithms, we aim to:

- Identify high-impact predictors of diabetes
- Analyze risk variations across demographic groups
- Support targeted public health strategies

---

## 👨‍⚕️ Research Questions

1. Which health and lifestyle indicators have the strongest predictive power for identifying diabetes risk?
2. How do socio-demographic factors (income, education, gender, age) influence diabetes prevalence?
3. How can machine learning improve diabetes risk prediction and healthcare planning?

---

## 🛠️ Tools & Technologies

| Tool | Description |
|------|-------------|
| **Languages** | Python, SQL |
| **Big Data Platform** | Databricks |
| **Frameworks** | PySpark, Spark MLlib |
| **Database** | NoSQL (Databricks DBFS) |
| **ML Algorithms** | Logistic Regression, Random Forest, Decision Tree, SVM |

---

## 📁 Project Structure
# Diabetes Dynamics: A Data Analytics Approach

📊 A data-driven project analyzing the behavioral, clinical, and socio-demographic factors influencing diabetes risk, using advanced big data and machine learning tools.

---

## 🧠 Project Summary

This project explores and models diabetes risk factors to uncover insights that support more effective and equitable healthcare interventions. By leveraging a large health indicators dataset and advanced ML algorithms, we aim to:

- Identify high-impact predictors of diabetes
- Analyze risk variations across demographic groups
- Support targeted public health strategies

---

## 👨‍⚕️ Research Questions

1. Which health and lifestyle indicators have the strongest predictive power for identifying diabetes risk?
2. How do socio-demographic factors (income, education, gender, age) influence diabetes prevalence?
3. How can machine learning improve diabetes risk prediction and healthcare planning?

---

## 🛠️ Tools & Technologies

| Tool | Description |
|------|-------------|
| **Languages** | Python, SQL |
| **Big Data Platform** | Databricks |
| **Frameworks** | PySpark, Spark MLlib |
| **Database** | NoSQL (Databricks DBFS) |
| **ML Algorithms** | Logistic Regression, Random Forest, Decision Tree, SVM |

---

You're right! For GitHub Markdown, code blocks don't render file trees well unless we use proper indentation inside **a fenced code block with ` ``` `**. Here's the correct Markdown version that will display the **project structure properly in GitHub preview**:

---

## 📁 Project Structure
```markdown
bash
Copy
Edit
diabetes-dynamics-insights/
├── README.md                   # Project overview and usage instructions  
├── requirements.txt            # Python dependencies  
├── .gitignore                  # Files and folders to be ignored by Git  
│  
├── assets/                     # Data quality visuals and system architecture diagrams  
│   ├── Accuracy.png  
│   ├── Completeness.png  
│   ├── Relevance.png  
│   ├── Validity.png  
│   ├── System Architecture.jpg  
│   └── Diabetes Logo.jpg  
│  
├── data/                       # Raw and cleaned datasets  
│   ├── diabetes_health_indicators.csv  
│   ├── cleaned_diabetes_dataset.csv  
│   └── Dataset.md              # Dataset source and explanation  
│  
├── images/                     # Exploratory data analysis (EDA) visualizations  
│   ├── Age by Diabetes.png  
│   ├── Bar Plot of Diabetes Prevalence.png  
│   ├── Correlation btw high bp_chol & Diabetes.png  
│   ├── Diabetes vs Heart Disease.png  
│   ├── Education by diabetes.png  
│   ├── HighBP by Diabetes.png  
│   ├── HighChol by Diabetes.png  
│   ├── Income Level by Diabetes.png  
│   └── PhysActivity by Diabetes.png  
│  
├── notebooks/                  # Databricks notebooks used in the project  
│   ├── AIT614_TEAM6_PROJECT_DATA_CLEANING.ipynb  
│   ├── AIT614_TEAM6_PROJECT_EDA.ipynb  
│   └── AIT614_TEAM6_PROJECT_RESEARCH_QUESTION_ANALYSIS_AND_ML_MODEL_ANALYSIS.ipynb  
│  
├── html_exports/               # HTML exports of notebooks for easy viewing  
│   ├── AIT614_TEAM6_PROJECT_DATA_CLEANING.html  
│   ├── AIT614_TEAM6_PROJECT_EDA.html  
│   └── AIT614_TEAM6_PROJECT_RESEARCH_QUESTION_ANALYSIS_AND_ML_MODEL_ANALYSIS.html  
│  
├── scripts/                    # Python scripts for data cleaning and analysis  
│   ├── Data Cleaning and preprocessing.py  
│   ├── Data Quality Assessment.py  
│   ├── Exploratory Data Analysis.py  
│   └── Data Quality Assessment.md  
│  
├── docs/                       # Final presentation and documentation  
│   └── AIT614-Sec001_Team6_Final.ppt  
│  
└── misc/                       # Temporary or reference files  
    ├── source code.txt  
    ├── .DS_Store  
    └── ~$AIT614-Sec001_Team6_Final.ppt  
```


# Diabetes Dynamics: A Data Analytics Approach

📊 A data-driven project analyzing the behavioral, clinical, and socio-demographic factors influencing diabetes risk, using advanced big data and machine learning tools.

---

## 🧠 Project Summary

This project explores and models diabetes risk factors to uncover insights that support more effective and equitable healthcare interventions. By leveraging a large health indicators dataset and advanced ML algorithms, we aim to:

- Identify high-impact predictors of diabetes
- Analyze risk variations across demographic groups
- Support targeted public health strategies

---

## 👨‍⚕️ Research Questions

1. Which health and lifestyle indicators have the strongest predictive power for identifying diabetes risk?
2. How do socio-demographic factors (income, education, gender, age) influence diabetes prevalence?
3. How can machine learning improve diabetes risk prediction and healthcare planning?

---

## 🛠️ Tools & Technologies

| Tool | Description |
|------|-------------|
| **Languages** | Python, SQL |
| **Big Data Platform** | Databricks |
| **Frameworks** | PySpark, Spark MLlib |
| **Database** | NoSQL (Databricks DBFS) |
| **ML Algorithms** | Logistic Regression, Random Forest, Decision Tree, SVM |

---

## 📁 Project Structure










---

## 🔍 Key Features

- 📊 **Correlation heatmaps** of features vs diabetes likelihood  
- 🧠 **Machine learning models** with performance comparison (Accuracy, F1, AUC)
- 📈 **Visualization of demographic influences** on diabetes (education, income, gender)
- 🌐 **Deployment-ready** PySpark ML pipelines

---

## 📊 Dataset

**Source**:  
[Diabetes Health Indicators – Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

> The dataset includes over 50,000 observations and 20+ features including:
> - Health metrics: BMI, Blood Pressure, Cholesterol
> - Lifestyle: Smoking, Diet, Physical Activity
> - Demographics: Age, Income, Education, Sex

---

## 📈 Model Results

| Model               | Accuracy | AUC Score | Notes                            |
|--------------------|----------|-----------|----------------------------------|
| Logistic Regression | 84.83%   | 78.17%    | Balanced baseline                |
| Random Forest       | 84.17%   | 80.13%    | Highest overall F1 score         |
| Decision Tree       | 82.92%   | 74.89%    | More interpretable, less stable  |
| SVM                 | 83.25%   | 77.45%    | Strong performer with tuning     |

---

## 📘 Final Presentation

🧾 [Download Final Presentation (PDF)](presentation/AIT614-Sec001_Team6_Final.pdf)

---

## ✍️ Authors

- Akhil Arekatika  
- Raghu Manjunatha  
- Ritesh Somashekar  
- Instructor: Dr. Eddy Zhang  
- Course: AIT 614 – Big Data Essentials, George Mason University

---

## 🔮 Future Enhancements

- Real-time data from wearable devices
- Integration with electronic health records (EHRs)
- Interpretable ML models for clinicians
- Framework adaptation for other chronic diseases

---

## 📜 License

This project is licensed under the MIT License – feel free to use and contribute with attribution.

---

## 📬 Contact

For questions or collaboration opportunities, reach out:

**Akhil Arekatika**  
📧 [aarekati@gmu.edu](mailto:aarekati@gmu.edu)  
🎓 George Mason University

