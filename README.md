# Air Quality & Pollution Trend Analysis

Streamlit app for Big Data Analytics Lab Practicals 1–7.

## Included practicals
- Dashboard and dataset preview
- Practical 2: data cleaning, descriptive statistics, EDA charts
- Practical 3: probability distributions and sampling techniques
- Practical 4: Central Limit Theorem
- Lab 5: confidence interval, one-sample Z-test and t-test
- Lab 6A: chi-square test of independence
- Lab 6B/C: covariance, Pearson/Spearman correlation and significance tests
- Practical 6: two-way ANOVA
- Lab 7: regression and model evaluation

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Upload `app.py`, `requirements.txt`, and `aqi.csv` to a GitHub repository.
2. Go to https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select the repository and branch, and set main file path to `app.py`.
4. Deploy and share the public URL with your instructor.

The app expects an `aqi_value` column. It uses `date`, `state`, `air_quality_status`, `prominent_pollutants`, and `number_of_monitoring_stations` where available.

## Statistical interpretation notes
- For a Z-test, population standard deviation should be known; the app asks for a sigma value and labels this assumption.
- Large datasets can make small effects statistically significant. Interpret effect sizes, assumptions, and data dependence as well as p-values.
- Two-way ANOVA requires sufficient observations across factor combinations and reasonable model assumptions.
- Regression performance depends on whether the available predictors genuinely explain AQI.
