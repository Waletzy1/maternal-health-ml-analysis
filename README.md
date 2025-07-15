# Maternal Health Risk Prediction Using Machine Learning

This project applies machine learning techniques to analyze maternal health factors and predict associated health risks. Using real-world medical datasets, the objective is to support early diagnosis and improve treatment outcomes for pregnant individuals.

## Overview

The analysis is divided into three main tasks:
- **Clustering**: Segment pregnant individuals by health profiles
- **Regression**: Investigate how maternal health factors affect blood sugar levels
- **Classification**: Predict maternal health risk levels (low, medium, high)

## Datasets

- **Sources**: 
  - UCI Machine Learning Repository
  - Mendeley Data
- **Features Include**: Age, BMI, Blood Sugar, Blood Pressure, Heart Rate, Body Temperature, Health Risk Level

## Machine Learning Methods

| Task          | Algorithm Used       | Purpose                                   |
|---------------|----------------------|-------------------------------------------|
| Clustering    | K-means, Hierarchical| Segment health profiles                   |
| Regression    | Polynomial Regression| Predict maternal blood sugar              |
| Classification| Random Forest        | Classify risk levels                      |

## Key Findings

- Polynomial regression outperformed linear models with R² ≈ 0.64
- Random Forest achieved 84.65% classification accuracy
- Blood sugar and systolic BP were key predictors of risk level
- Clusters revealed health profiles needing different interventions

## Technologies Used

- R Programming
- ggplot2, caret, randomForest
- Data preprocessing, model training, validation, and visualization

## Visual Outputs

- Confusion Matrix
- Variable Importance Plots
- Cluster Dendrograms
- Best Fit Line (Regression)

## Business Value

This project demonstrates the potential of machine learning in maternal healthcare, enabling early risk detection, better monitoring, and improved intervention strategies—especially valuable in low-resource settings.

## License

This project is for academic use only. All datasets are publicly available and cited in the report.

## Author

**Seow Xin Yong**  
BSc (Hons) Data Science & Business Analytics  
University of London @ SIM  
[LinkedIn](https://www.linkedin.com/in/seow-xin-yong/)
