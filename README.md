# 📘 Student Performance Evaluation with Evidently AI

## 👋 Introduction

This document summarizes my familiarisation task using the Evidently AI framework to monitor student model performance. The aim was to detect any data drift, assess prediction accuracy, and explore student feedback using descriptor-based analysis. I used Evidently's modern metrics and visualization tools to create an end-to-end report that would support machine learning quality assurance in production-like setups.

## 📂 Data Used

Two synthetic datasets were created to simulate a real-world scenario:

- `data/students_reference_dataset.csv`: Represents clean historical data.
- `data/students_current_dataset.csv`: Simulates new incoming data that may differ in pattern.

Each dataset includes:

- `student_id`, `age`, `study_hours_per_week`, `previous_gpa`, `course_difficulty`, `actual_grade`, `predicted_grade`, `satisfaction_level`, `feedback_text`

The task is to evaluate whether the current data behaves similarly to reference data, and whether the model continues to perform well.

## ⚙️ Methodology & Code Summary

The evaluation is implemented using the `evidently` library in Python.

### Key Steps:

1. Load the datasets using Pandas.
2. Define schema with `DataDefinition`, specifying:
   - Numerical columns (e.g., `age`, `study_hours_per_week`)
   - Categorical columns (e.g., `course_difficulty`)
   - Text column: `feedback_text`
3. Regression setup with `actual_grade` as target and `predicted_grade` as prediction
4. Attach descriptors to the feedback text:
   - **Sentiment** – measures polarity from -1 to +1
   - **TextLength** – total characters
   - **Contains** – flags specific words to detect issues or praise
5. Generate the report using these Evidently presets:
   - `DataSummaryPreset`
   - `DataDriftPreset`
   - `TextEvals`
   - `MeanError`
6. Save HTML report for visualization.

## 📊 Understanding the Metrics and Drift Scores

### ✅ Regression Metrics

Used to evaluate prediction performance:

- **Mean Error**: Measures average difference between predicted and actual grades. Ideal value is close to 0.
- **MAE** (Mean Absolute Error): Absolute average difference; robust against direction.
- **RMSE** (Root Mean Squared Error): Penalizes larger errors more than MAE.

### 🔄 Drift Score (KS-Test)

- Kolmogorov-Smirnov Test (KS-test) is used for continuous numerical features.
- Measures the difference between distributions in reference and current datasets.
- A **p-value < 0.05** indicates that the feature has changed statistically (drift detected).

### 📊 Chi-Square p_value (Categorical Drift)

- Chi-Square Test is used for categorical features.
- It compares the frequency distribution of categories between the reference and current datasets.
- A **p-value < 0.05** means there’s a statistically significant shift in category frequencies (drift detected).
- Best used when each category has a sufficient number of samples.

### 📈 Z-test p_value (Performance Drift)

- Z-test is typically used for comparing proportions, such as accuracy, error rates, or prediction distributions.
- It checks whether the difference between two proportions (like success rates in classification) is statistically significant.
- A **p-value < 0.05** suggests that the model's performance or output distribution has changed (performance drift detected).
- Often applied in evaluating binary classification outputs or performance over time.

## ❗ Test Section

Auto-generated pass/fail tests are included to:

- Check data quality (e.g., missing values, zero variance)
- Identify significant drifts
- Detect text issues via rule-based logic (e.g., negative wording)

## 📈 Findings from the Initial Experiment

### 🟩 Stable Features

- `study_hours_per_week`, `actual_grade`, `predicted_grade`, `satisfaction_level`, and `course_difficulty` showed no drift.
- Feedback sentiment remained stable (mean ≈ 0.68).

### ⚠️ Drifted Features (3/12 columns, 25%)

- **age**  
  p = 0.036 → Drift detected.

- **previous_gpa**  
  p = 0.0002 → Strong drift. Current students show lower GPA.

- **TextLength**  
  Significantly higher variance; longer feedback responses in the current dataset.

## 📝 Text Descriptor Observations

- Positive highlights like "excellent" and "rewarding" appear more often.
- Some feedbacks flagged as unclear ("confusing", "repetitive") but were within expected limits.

## 📋 Concepts Learned and Documented

### 📚 New Concepts

- Data Drift vs. Concept Drift
- KS-test and p-value interpretation
- Descriptors in text-based evaluation
- Regression error metrics and how to interpret them

**📄 Report Saved**: `reports/student_full_evaluation_report.html`
