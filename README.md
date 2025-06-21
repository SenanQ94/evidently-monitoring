# Student Performance Evaluation with Evidently AI

## Introduction

This document summarizes my familiarisation task using the Evidently AI framework to monitor 
<mark data-comment="what is a student model? what is performance?">student model performance</mark>.
The aim was to detect any data drift, assess prediction accuracy, and explore
<mark data-comment="what is student feedback?">student feedback</mark>
using
<mark data-comment="please define that somewhere">descriptor-based analysis</mark>.
I used Evidently's
<mark data-comment="what does modern mean? no need for marketing buzzwords.">modern</mark>
metrics and visualization tools to create an
<mark data-comment="what does end-to-end mean? either explain or remove it.">end-to-end report</mark>
that would support machine learning quality assurance in production-like setups.

## Data Used

Two synthetic datasets were created to simulate a real-world scenario:

- `data/students_reference_dataset.csv`: Represents
<mark data-comment="what does clean mean?">clean</mark>
historical data.
- `data/students_current_dataset.csv`: Simulates
<mark data-comment="could you explain the purpose of new data? and give an example where that data might be coming from?">new incoming data</mark>
that may differ in
<Mark data-comment="when you say pattern, what exactly are you referring to?">pattern</Mark>.

Each dataset includes:

- `student_id`, `age`, `study_hours_per_week`, `previous_gpa`, `course_difficulty`, `actual_grade`, `predicted_grade`, `satisfaction_level`, `feedback_text`

The task is to evaluate whether the
<mark data-comment="you need to define what current data is">current data</mark>
behaves similarly to
<mark data-comment="you need to define what reference data is">reference data</mark>,
and whether the
<mark data-comment="you need to introduce and explain the model somewhere">model</mark>
continues to perform well.

## Methodology & Code Summary

The evaluation is implemented using the
<mark data-comment="a bit more context would be useful">`evidently` library in Python</mark>.

### Key Steps:

1. Load the datasets using Pandas.
2. Define <mark data-comment="this was never mentioned before, more context needed">schema with `DataDefinition`</mark>, specifying:
   - Numerical columns (e.g., `age`, `study_hours_per_week`)
   - Categorical columns (e.g., `course_difficulty`)
   - Text column: `feedback_text`
3. Regression setup with `actual_grade` as target and `predicted_grade` as prediction
4. <mark data-comment="this seems important thus needs better explanation">Attach descriptors to the feedback text</mark>:
   - **Sentiment** – measures polarity from -1 to +1
   - **TextLength** – total characters
   - **Contains** – flags specific words to detect issues or praise
5. Generate the report using these <mark data-comment="what are presets?">Evidently presets</mark>:
   - `DataSummaryPreset`
   - `DataDriftPreset`
   - `TextEvals`
   - `MeanError`
6. Save HTML report for visualization.

## Understanding the Metrics and Drift Scores

### Regression Metrics

<mark>Chatgpt: Could you provide me an overview of regressive metrics? Answer in markdown.</mark>

Used to evaluate the performance of a regression model
<mark data-comment="btw. this character 'indicates' that chatgpt was involved since there is no key on your keyboard for it :-)">—</mark>
how well the predicted values match the actual values.

#### Mean Error (ME)

- Measures the **average** difference between predicted and actual values.
- Can be **positive or negative**, depending on whether predictions are over or under the actual values.
- Useful for detecting **bias**: if consistently overestimating or underestimating.
- **Ideal value**: Close to **0** (no consistent bias).

> Example:  
> Predictions: [80, 85, 90]  
> Actual: [82, 83, 91]  
> ME = ((80-82) + (85-83) + (90-91)) / 3 = (-2 + 2 - 1) / 3 = **-0.33**

#### Mean Absolute Error (MAE)

- The **average of the absolute differences** between predictions and actual values.
- Unlike Mean Error, it doesn’t cancel out over- and under-predictions.
- **Always positive**.
- Gives a clear idea of the average size of the errors, regardless of direction.
- **Ideal value**: **0** (perfect predictions).

> Example:  
> MAE = (|80-82| + |85-83| + |90-91|) / 3 = (2 + 2 + 1) / 3 = **1.67**

#### Root Mean Squared Error (RMSE)

- The **square root of the average of squared differences** between predicted and actual values.
- Squaring the errors **penalizes larger errors** more than smaller ones.
- More sensitive to **outliers** than MAE.
- **Ideal value**: **0** (no error).

> Example:  
> RMSE = √[( (80-82)² + (85-83)² + (90-91)² ) / 3]  
> = √[(4 + 4 + 1) / 3] = √3 = **~1.73**

---

### When to use what?

| Metric         | Good For                     | Sensitive To             | Notes                                 |
| -------------- | ---------------------------- | ------------------------ | ------------------------------------- |
| **Mean Error** | Detecting bias               | Direction of errors      | Can cancel out over/under-predictions |
| **MAE**        | Measuring average error size | Evenly treats all errors | Easier to interpret                   |
| **RMSE**       | Highlighting large mistakes  | Outliers                 | Penalizes large errors more           |

### Drift Score (KS-Test)

- Kolmogorov-Smirnov Test (KS-test) is used for <mark data-comment="did you use it? example?">continuous numerical features</mark>.
- Measures the difference between distributions in reference and current datasets.
- A **p-value < 0.05** indicates that the feature has changed statistically (drift detected).

### Chi-Square p_value (Categorical Drift)

- Chi-Square Test is used for <mark data-comment="did you use it? example?">categorical features</mark>.
- It compares the frequency distribution of categories between the reference and current datasets.
- A **p-value < 0.05** means there’s a statistically significant shift in category frequencies (drift detected).
- Best used when each category has a sufficient number of samples.

### Z-test p_value (Performance Drift)

- Z-test is typically used for <mark data-comment="did you use it? exmple?">comparing proportions</mark>, such as accuracy, error rates, or prediction distributions.
- It checks whether the difference between two proportions (like success rates in classification) is statistically significant.
- A **p-value < 0.05** suggests that the model's performance or output distribution has changed (performance drift detected).
- Often applied in evaluating binary classification outputs or performance over time.

## Test Section

Auto-generated pass/fail tests are included to:

- Check data quality (e.g., missing values, zero variance)
- Identify significant drifts
- Detect text issues via rule-based logic (e.g., negative wording)

## Findings from the <mark data-comment="initial experiment was never described before">Initial Experiment<mark>

### Stable Features

- `study_hours_per_week`, `actual_grade`, `predicted_grade`, `satisfaction_level`, and `course_difficulty` showed no drift.
- <mark data-comment="needs better explanation, also: define 'stable'?">Feedback sentiment remained stable (mean ≈ 0.68)</mark>.

### Drifted Features (3/12 columns, 25%)

- **age**  
  <mark data-comment="how does it work? what is the meaning of p?">p = 0.036 → Drift detected</mark>.

- **previous_gpa**  
  p = 0.0002 → Strong drift. <mark data-comment="what does current mean?">Current students</mark> show lower GPA.

- **TextLength**  
  Significantly higher variance; longer feedback responses in the current dataset.

## Text Descriptor Observations

- Positive highlights like "excellent" and "rewarding" appear <mark data-comment="compared to what?">more often</mark>.
- <mark data-comment="what does that mean?">Some feedbacks flagged as unclear ("confusing", "repetitive")</mark> but were within <mark data-comment="whar are the limits?">expected limits</mark>.

## Concepts Learned and Documented

### New Concepts

- <mark data-comment="explain that using your experiment">Data Drift vs. Concept Drift</mark>
- <mark data-comment="explain that using your experiment">KS-test and p-value interpretation</mark>
- <mark data-comment="explain that using your experiment">Descriptors in text-based evaluation</mark>
- <mark data-comment="explain that using your experiment">Regression error metrics and how to interpret them</mark>

**Report Saved**: `reports/student_full_evaluation_report.html`