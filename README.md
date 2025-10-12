# House Price Prediction (EDA + k-NN Regression)

This project explores the Kaggle Housing Prices dataset to understand numeric feature distributions, handle outliers, check data quality, and predict house prices using a k-NN Regressor.

## Files

app.py : Streamlit app for interactive house price prediction.

housing.csv : Dataset used in the app.

requirements.txt : Python dependencies.

README.md : Project overview and instructions.

## Environment and Dependencies

Install the required Python packages:

pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib

## Notebook / App Workflow

Load dataset and inspect structure:

df.info()
df.describe()


Check missing values:

Confirm completeness with df.isnull().sum() and missingno.bar(df).

Visualize distributions:

Histogram/KDE plots for numeric features (area, bedrooms, bathrooms, stories, parking).

Boxplots to detect outliers.

Outlier handling:

Cap outliers using IQR method to reduce extreme values.

Correlation analysis:

Correlation heatmap to detect multicollinearity between numeric features.

Feature scaling:

Scale numeric features using StandardScaler for better k-NN performance.

Train k-NN Regressor:

5-fold cross-validation to calculate CV RMSE.

Train on full training data.

Evaluate on test set with Test RMSE and R² Score.

Prediction interface:

Streamlit app allows user input for numeric features to predict house price interactively.

## Visualizations

Distribution plots:

Histograms and KDE plots for numeric features to assess skewness and scale.

Correlation heatmap:

Detect multicollinearity among numeric features.

Boxplots:

Identify and visualize outliers in each numeric feature.

Predicted vs Actual Price Plot:

Scatter plot comparing model predictions against actual house prices in the test set.

## Key Findings

Dataset has no missing values.

Some numeric features are right-skewed (e.g., area, bedrooms), which can impact predictions; scaling mitigates this.

Moderate correlation exists between features, but no extreme multicollinearity after scaling.

Outlier capping reduces extreme price/area values, improving model stability.

## Evaluation Metrics
Metric	Value	Description
CV RMSE	varies (k=5)	Average error across training folds using 5-fold cross-validation.
Test RMSE	varies	Average prediction error on unseen test data.
Test R² Score	varies	Proportion of variance in house prices explained by the model.
How to Run

Ensure housing.csv is in the project root.

Run the Streamlit app:

streamlit run app.py


Enter numeric house details (area, bedrooms, bathrooms, stories, parking) in the app.

Click Predict Price to get the predicted house price.

## Notes

The app currently uses only numeric features.

Outliers are capped to improve predictions.

k-NN model uses 5 neighbors and distance weighting for better accuracy.

Predicted price is displayed interactively in the Streamlit app.

## Next Steps

Tune k in k-NN or try weighted distances to improve accuracy.

Include categorical features (e.g., mainroad, furnishingstatus) to enhance predictions.

Perform feature engineering (e.g., area per story, bedrooms * bathrooms).

Compare k-NN performance with other regression models like Linear Regression or Random Forest.