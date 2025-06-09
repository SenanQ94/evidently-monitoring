# 📊 Report Walkthrough: Evidently AI Monitoring

This section explains, step by step, how the Evidently AI report was generated and what each component means. The report evaluates model performance, data drift, and feedback sentiment using a synthetic dataset of student performance.

---

## 1. 📁 Input Datasets

For this project, I created two demo datasets of students with the following columns:

- `student_id`
- `age`
- `study_hours_per_week`
- `previous_gpa`
- `course_difficulty`
- `actual_grade`
- `predicted_grade`
- `satisfaction_level`
- `feedback_text`

**Files used:**

- `student_reference_dataset.csv`: Simulated "reference" data
- `student_current_dataset.csv`: Simulated "current" production data

Each row represents a student’s academic profile, satisfaction, and written feedback.

---

## 2. ⚙️ Dataset Definition

We explicitly define the column roles using `DataDefinition`, including:

- `regression`: Actual vs. predicted grade
- `numerical_columns`: Input metrics
- `text_columns`: Free-text feedback

**Descriptors used:**

- `Sentiment`: Overall polarity of the feedback text
- `TextLength`: Number of characters per response
- `Contains`: Looks for denial or negative indicators

---

## 3. 📄 Report Configuration

The report uses these Evidently presets:

- `DataSummaryPreset`: Summary statistics for each column
- `DataDriftPreset`: Statistical comparison between reference and current datasets
- `TextEvals`: Summarizes feedback descriptors

**Regression metrics (optional):**

- `MeanError`
- `MeanAbsoluteError`
- `RootMeanSquaredError`

---

## 4. 📊 Key Report Sections

### ➤ Metadata Section

| Metric            | Current | Reference |
| ----------------- | ------- | --------- |
| id column         | None    | None      |
| target column     | None    | None      |
| prediction column | None    | None      |
| date column       | None    | None      |

❗ _This means we didn’t assign `id`, `target`, or `prediction` roles explicitly in this section of the report. These should be assigned in `DataDefinition()` using regression or classification._

---

### ➤ Data Summary

Provides count, min, max, mean, and standard deviation for each numerical feature.

Useful for:

- Checking for missing values
- Detecting unexpected shifts in distribution

---

### ➤ Data Drift

**Statistical tests used:**

- Numerical: KS-test (Kolmogorov-Smirnov)
- Categorical: Chi-square test

You’ll see a **"drift score"** and whether a column has statistically changed.  
**Drift is flagged when `p-value < 0.05`.**

---

### ➤ Regression Metrics

These are only shown if `Regression(target=..., prediction=...)` is defined.

- **Mean Error**: Average difference between prediction and true grade
- **MAE**, **RMSE**: Standard regression metrics

---

### ➤ Text Descriptors

Evaluates the `feedback_text` column:

- **Sentiment**: Value from -1 (negative) to +1 (positive)
- **TextLength**: Number of characters
- **Contains**: Flags presence of phrases like "sorry", "difficulties"

---

### ➤ Tests Section

Auto-generated pass/fail conditions based on:

- Drift thresholds
- Data quality (missing values, zero variance, etc.)
- Text-based rules (e.g. _"no feedback should contain denial terms"_)

---

## 5. 💡 How to Read the Report

Each tab in the HTML file represents a different evaluation:

- **Data Summary** → Feature-wise distribution stats
- **Data Drift** → Whether features statistically changed
- **Text Evals** → Sentiment & length of student feedback
- **Test Results** → Automated rule-based validations

---

## ✅ Next Steps

- Confirm if we should repeat this with real datasets
- Add MLflow integration for storing experiment results
- Expand feedback descriptors using LLM-based sentiment (optional)
