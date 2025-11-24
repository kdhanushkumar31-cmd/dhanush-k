# 🚢 Task 2: Exploratory Data Analysis (EDA) of the Titanic Dataset

## Project Overview

This project completes **Task 2: Exploratory Data Analysis (EDA)** for the AI/ML internship. The goal was to conduct a thorough initial analysis of the Titanic passenger data using Python to understand the dataset's structure, identify key statistical properties, and uncover influential patterns related to survival.

## 🛠️ Tools & Libraries Used

* **Python 3.x**
* **Pandas:** For data loading, manipulation, and statistical summaries.
* **Seaborn & Matplotlib:** For data visualization (Histograms, Boxplots, Correlation Heatmap).
* **Jupyter Notebook:** The primary environment for analysis.
* **Git & GitHub:** For version control and final project submission.

## 📊 Key Findings and Inferences (Hints 4 & 5)

Based on the statistical analysis and visualizations (correlation heatmap, bar plots of survival rates), the following significant patterns were identified:

1.  **Gender is the Primary Indicator:** A massive disparity was observed in survival rates based on the `Sex` column. **Females** had a substantially higher probability of survival, supporting the "women and children first" protocol.
2.  **Socio-Economic Status:** `Pclass` (Passenger Class) is strongly correlated with survival. **First-class passengers** (Pclass=1) had the highest survival rate, while third-class passengers (Pclass=3) had the lowest, indicating wealth and location on the ship played a critical role.
3.  **Age Distribution:** The age distribution shows a higher count of young adults (20-40 years), and the **survival rate for children** (under 10) was notably high.
4.  **Missing Data:** Significant missing values were observed in the **`Age`** and **`Cabin`** columns. The high volume of missing `Cabin` data makes it difficult to use directly, while the missing `Age` data would require careful imputation before model training.

## 📁 Repository Contents

* **`Titanic_EDA.ipynb`:** The complete Jupyter Notebook containing all the Python code, visualizations, and detailed cell-by-cell output.
* **`Titanic-Dataset.csv`:** The raw dataset used for this analysis.

***

**Prepared by:** DHANUSH K
