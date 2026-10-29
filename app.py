import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency, ttest_1samp, norm, t as student_t, pearsonr, spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Air Quality & Pollution Trend Analysis", page_icon="🌫️", layout="wide")
st.title("🌫️ Air Quality & Pollution Trend Analysis")
st.caption("Big Data Analytics Lab Project | Practicals 1–7")

@st.cache_data
def load_data(file):
    # If the user uploads a CSV, use that file.
    if file is not None:
        return pd.read_csv(file)

    # Default dataset in the GitHub repository is gzip-compressed.
    try:
        return pd.read_csv("aqi.csv.gz", compression="gzip")
    except FileNotFoundError:
        # Fallback in case an ordinary CSV is present.
        return pd.read_csv("aqi.csv")

uploaded = st.sidebar.file_uploader(
    "Upload aqi.csv (optional)",
    type=["csv"]
)

try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"Could not load the dataset. Upload a CSV in the sidebar or add aqi.csv.gz beside app.py. Details: {e}")
    st.stop()

df = df.copy()
if "aqi_value" not in df.columns:
    st.error("The dataset must contain an 'aqi_value' column.")
    st.stop()

df["aqi_value"] = pd.to_numeric(df["aqi_value"], errors="coerce")
if "number_of_monitoring_stations" in df.columns:
    df["number_of_monitoring_stations"] = pd.to_numeric(df["number_of_monitoring_stations"], errors="coerce")
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

page_options = {
    "Dashboard": "Dashboard",
    "Data Cleaning & Exploratory Data Analysis": "Practical 2: Cleaning & EDA",
    "Probability Distributions & Sampling": "Practical 3: Probability & Sampling",
    "Central Limit Theorem": "Practical 4: Central Limit Theorem",
    "Confidence Intervals & Hypothesis Testing (Z-test and t-test)": "Lab 5: Confidence Intervals & Z/t Tests",
    "Chi-Square Test": "Lab 6A: Chi-Square Test",
    "Covariance & Correlation Analysis": "Lab 6B/C: Covariance & Correlation",
    "Two-Way ANOVA": "Practical 6: Two-Way ANOVA",
    "Regression Analysis": "Lab 7: Regression",
    "Regression Model Comparison": "Regression Model Comparison"
}
selected_topic = st.sidebar.radio("Navigate topics", list(page_options.keys()))
page = page_options[selected_topic]


topic_info = {
    "Dashboard": (
        "Purpose",
        "Gives a quick overview of the AQI dataset, including record count, average and maximum AQI, missing values, and duplicate rows.",
        "The summary values are calculated directly from the loaded dataset. Average AQI = sum of valid AQI values divided by the number of valid values."
    ),
    "Data Cleaning & Exploratory Data Analysis": (
        "Purpose",
        "Finds missing values and duplicates, summarizes numerical data, and uses charts to understand distributions, trends, outliers, and category frequencies.",
        "Mean = sum(x)/n; median = middle sorted value; mode = most frequent value; variance measures spread around the mean; standard deviation = square root of variance; quartiles divide sorted data into four parts."
    ),
    "Probability Distributions & Sampling": (
        "Purpose",
        "Studies how AQI values are distributed and how samples can represent the larger dataset.",
        "Simple random sampling selects records randomly; systematic sampling selects records at a regular interval; stratified sampling selects records from each category. A sampling distribution is formed by repeatedly calculating sample statistics."
    ),
    "Central Limit Theorem": (
        "Purpose",
        "Shows how the distribution of sample means behaves when many samples of the same size are drawn from a population.",
        "Draw repeated samples, calculate each sample mean, and plot those means. The standard error is approximately s/√n; larger samples generally produce less variation in the sample means."
    ),
    "Confidence Intervals & Hypothesis Testing (Z-test and t-test)": (
        "Purpose",
        "Estimates a plausible range for the population mean AQI and tests a claim about that mean.",
        "A confidence interval is estimate ± critical value × standard error. A one-sample t-test uses t = (sample mean − hypothesized mean)/(sample standard deviation/√n). A Z-test uses a known population standard deviation. Compare the p-value with α to make a decision."
    ),
    "Chi-Square Test": (
        "Purpose",
        "Checks whether two categorical variables, such as state and air-quality status, appear to be associated.",
        "Build a contingency table of observed counts. The chi-square statistic is Σ((observed − expected)²/expected). A small p-value (below α) provides evidence of an association; check expected cell counts before interpreting the result."
    ),
    "Covariance & Correlation Analysis": (
        "Purpose",
        "Measures the direction and strength of the relationship between two numerical variables.",
        "Covariance indicates whether variables tend to move together. Pearson correlation measures linear association from −1 to +1; Spearman correlation measures monotonic association using ranks. Their p-values test whether the observed association is statistically significant."
    ),
    "Two-Way ANOVA": (
        "Purpose",
        "Compares mean AQI across groups defined by two categorical factors and tests their individual and interaction effects.",
        "The ANOVA model separates variation associated with Factor A, Factor B, their interaction, and residual error. It uses F-statistics and p-values to assess effects. Results rely on suitable independence, residual, and variance assumptions."
    ),
    "Regression Analysis": (
        "Purpose",
        "Fits a model to predict AQI from selected numerical predictor variables and evaluates prediction quality.",
        "Linear regression estimates an intercept and coefficients that minimize squared prediction errors. R² describes explained variation; MAE is average absolute error; RMSE penalizes larger errors more strongly. Test-set metrics estimate performance on unseen records."
    ),
    "Regression Model Comparison": (
        "Purpose",
        "Compares different regression algorithms on the same AQI dataset and test records to see which predicts AQI more accurately.",
        "Simple linear regression uses one predictor; multiple linear regression uses several predictors; polynomial regression adds squared and interaction terms; random forest averages predictions from many decision trees. Compare R² (higher is generally better) and MAE/RMSE (lower is better) on the same held-out test set."
    )
}

info_title, info_purpose, info_method = topic_info[selected_topic]
with st.expander("About this topic: purpose and calculation", expanded=False):
    st.markdown(f"**What it is used for**  \n{info_purpose}")
    st.markdown(f"**How it is calculated (short)**  \n{info_method}")
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit, Pandas, NumPy, SciPy, Statsmodels and Scikit-learn")

def numeric_data(data, col):
    return pd.to_numeric(data[col], errors="coerce").dropna()

def as_text(series):
    return series.fillna("Missing").astype(str)

if page == "Dashboard":
    st.subheader("Project Overview")
    a, b, c, d = st.columns(4)
    a.metric("Total records", f"{len(df):,}")
    b.metric("Dataset columns", len(df.columns))
    c.metric("Average AQI", f"{df['aqi_value'].mean():.2f}")
    d.metric("Maximum AQI", f"{df['aqi_value'].max():.0f}")
    st.markdown("### Dataset preview")
    st.dataframe(df.head(100), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.markdown("**Missing values by column**")
        st.dataframe(df.isna().sum().rename("Missing values"), use_container_width=True)
    with right:
        st.markdown("**Dataset quality**")
        st.write(f"Duplicate rows: **{df.duplicated().sum():,}**")
        st.write(f"Rows: **{len(df):,}** · Columns: **{len(df.columns)}**")
    st.download_button("Download dataset CSV", df.to_csv(index=False).encode("utf-8"), "aqi_download.csv", "text/csv")

elif page == "Practical 2: Cleaning & EDA":
    st.subheader("Practical 2 — Data Cleaning, Descriptive Statistics & EDA")
    st.write("Missing values by column")
    st.dataframe(df.isna().sum().rename("Missing values"))
    st.write(f"Duplicate rows: **{df.duplicated().sum():,}**")
    clean = df.drop_duplicates().copy()
    numeric_cols = clean.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.warning("No numerical columns found.")
        st.stop()
    selected = st.selectbox("Choose a numerical variable", numeric_cols,
                            index=numeric_cols.index("aqi_value") if "aqi_value" in numeric_cols else 0)
    series = clean[selected].dropna()
    if len(series):
        mode = series.mode()
        measures = pd.DataFrame({
            "Measure": ["Count", "Mean", "Median", "Mode", "Variance", "Standard deviation", "Minimum", "Q1 (25%)", "Q2 (50%)", "Q3 (75%)", "Maximum"],
            "Value": [len(series), series.mean(), series.median(), mode.iloc[0] if len(mode) else np.nan,
                      series.var(), series.std(), series.min(), series.quantile(.25), series.quantile(.50),
                      series.quantile(.75), series.max()]
        })
        st.dataframe(measures, use_container_width=True)
        st.write("Summary statistics")
        st.dataframe(series.describe().to_frame("Value"))
    st.markdown("### Visualizations")
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots()
        ax.hist(series, bins=30)
        ax.set(title=f"Histogram of {selected}", xlabel=selected, ylabel="Frequency")
        st.pyplot(fig)
    with right:
        fig, ax = plt.subplots()
        ax.boxplot(series)
        ax.set(title=f"Box plot of {selected}", ylabel=selected)
        st.pyplot(fig)
    categorical_cols = clean.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        cat = st.selectbox("Category for bar and pie charts", categorical_cols,
                           index=categorical_cols.index("state") if "state" in categorical_cols else 0)
        counts = as_text(clean[cat]).value_counts().head(15)
        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots(figsize=(8, 4))
            counts.plot(kind="bar", ax=ax)
            ax.set(title=f"Top categories in {cat}", xlabel=cat, ylabel="Count")
            ax.tick_params(axis="x", rotation=60)
            st.pyplot(fig)
        with right:
            fig, ax = plt.subplots()
            counts.head(8).plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set(title=f"Category share: {cat}", ylabel="")
            st.pyplot(fig)
    if "number_of_monitoring_stations" in clean.columns:
        pair = clean[["number_of_monitoring_stations", "aqi_value"]].dropna()
        fig, ax = plt.subplots()
        ax.scatter(pair["number_of_monitoring_stations"], pair["aqi_value"], alpha=.25)
        ax.set(title="Monitoring Stations vs AQI", xlabel="Monitoring stations", ylabel="AQI")
        st.pyplot(fig)
    if "date" in clean.columns:
        trend = clean.dropna(subset=["date", "aqi_value"]).groupby("date")["aqi_value"].mean().sort_index()
        if len(trend):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(trend.index, trend.values)
            ax.set(title="Average AQI Trend Over Time", xlabel="Date", ylabel="Average AQI")
            ax.tick_params(axis="x", rotation=35)
            st.pyplot(fig)
    st.download_button("Download cleaned CSV", clean.to_csv(index=False).encode("utf-8"), "aqi_cleaned.csv", "text/csv")

elif page == "Practical 3: Probability & Sampling":
    st.subheader("Practical 3 — Probability Distributions & Random Sampling")
    population = numeric_data(df, "aqi_value")
    st.write(f"Valid AQI observations: **{len(population):,}**")
    st.write(f"Mean: **{population.mean():.2f}** · Standard deviation: **{population.std():.2f}**")
    fig, ax = plt.subplots()
    ax.hist(population, bins=30, density=True)
    ax.set(title="Empirical AQI Distribution", xlabel="AQI", ylabel="Density")
    st.pyplot(fig)
    if population.std() > 0:
        x = np.linspace(population.min(), population.max(), 250)
        fig, ax = plt.subplots()
        ax.hist(population, bins=30, density=True, alpha=.5, label="Observed AQI")
        ax.plot(x, norm.pdf(x, population.mean(), population.std()), label="Fitted normal curve")
        ax.set(title="AQI and Fitted Normal Distribution", xlabel="AQI", ylabel="Density")
        ax.legend()
        st.pyplot(fig)
    n = st.slider("Sample size", 2, min(500, len(df)), min(30, len(df)))
    method = st.selectbox("Sampling technique", ["Simple random", "Systematic", "Stratified by air-quality status"])
    seed = st.number_input("Random seed", 0, 99999, 42)
    valid = df.dropna(subset=["aqi_value"]).copy()
    if method == "Simple random":
        sample = valid.sample(n=min(n, len(valid)), random_state=int(seed))
    elif method == "Systematic":
        k = max(1, len(valid) // n)
        start = int(seed) % k
        sample = valid.iloc[start::k].head(n)
    elif "air_quality_status" in valid.columns:
        parts = []
        for _, group in valid.groupby("air_quality_status", dropna=False):
            count = max(1, round(n * len(group) / len(valid)))
            parts.append(group.sample(n=min(count, len(group)), random_state=int(seed)))
        sample = pd.concat(parts).head(n)
    else:
        st.info("No air_quality_status column found; using simple random sampling.")
        sample = valid.sample(n=min(n, len(valid)), random_state=int(seed))
    st.metric("Selected sample mean AQI", f"{sample['aqi_value'].mean():.2f}")
    st.dataframe(sample, use_container_width=True)
    st.download_button("Download selected sample", sample.to_csv(index=False).encode("utf-8"), "aqi_sample.csv", "text/csv")
    reps = st.slider("Repeated samples", 100, 2000, 500, step=100)
    means = [population.sample(n=min(n, len(population)), replace=True).mean() for _ in range(reps)]
    fig, ax = plt.subplots()
    ax.hist(means, bins=30)
    ax.set(title="Sampling Distribution of Mean AQI", xlabel="Sample mean AQI", ylabel="Frequency")
    st.pyplot(fig)
    st.write(f"Mean of sample means: **{np.mean(means):.2f}**")
    st.write(f"Standard deviation of sample means: **{np.std(means):.2f}**")

elif page == "Practical 4: Central Limit Theorem":
    st.subheader("Practical 4 — Central Limit Theorem")
    population = numeric_data(df, "aqi_value")
    a, b = st.columns(2)
    a.metric("Population mean AQI", f"{population.mean():.2f}")
    b.metric("Population standard deviation", f"{population.std():.2f}")
    n = st.slider("Sample size (n)", 2, min(500, len(population)), min(30, len(population)), key="clt_n")
    reps = st.slider("Number of repeated samples", 100, 3000, 1000, step=100)
    seed = st.number_input("CLT random seed", 0, 99999, 42)
    rng = np.random.default_rng(int(seed))
    values = population.to_numpy()
    means = np.array([rng.choice(values, size=n, replace=True).mean() for _ in range(reps)])
    fig, ax = plt.subplots()
    ax.hist(means, bins=30, density=True, alpha=.7, label="Sample means")
    if means.std() > 0:
        x = np.linspace(means.min(), means.max(), 250)
        ax.plot(x, norm.pdf(x, means.mean(), means.std()), label="Fitted normal curve")
    ax.set(title=f"CLT Sampling Distribution (n={n})", xlabel="Sample mean AQI", ylabel="Density")
    ax.legend()
    st.pyplot(fig)
    st.dataframe(pd.DataFrame({"Measure": ["Population mean", "Mean of sample means", "Theoretical standard error", "Observed standard error"],
                               "Value": [population.mean(), means.mean(), population.std(ddof=1)/np.sqrt(n), means.std(ddof=1)]}),
                 use_container_width=True)
    st.download_button("Download sample means", pd.DataFrame({"sample_mean_aqi": means}).to_csv(index=False).encode("utf-8"), "clt_sample_means.csv", "text/csv")

elif page == "Lab 5: Confidence Intervals & Z/t Tests":
    st.subheader("Lab 5 — Confidence Intervals and One-Sample Hypothesis Tests")
    values = numeric_data(df, "aqi_value").to_numpy()
    if len(values) < 2:
        st.error("Not enough valid AQI observations.")
        st.stop()
    alpha = st.select_slider("Significance level (alpha)", options=[0.10, 0.05, 0.01], value=0.05)
    confidence = 1 - alpha
    n = len(values)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    se = sd / np.sqrt(n)
    critical = student_t.ppf(1-alpha/2, df=n-1)
    st.markdown("### Confidence interval for mean AQI")
    st.write(f"{confidence:.0%} confidence interval using t distribution")
    st.code(f"Mean = {mean:.4f}\nStandard error = {se:.4f}\nCI = ({mean-critical*se:.4f}, {mean+critical*se:.4f})")
    st.markdown("### One-sample t-test")
    reference = st.number_input("Hypothesized mean AQI (μ₀)", value=100.0)
    alternative = st.selectbox("Alternative hypothesis", ["two-sided", "greater", "less"])
    t_stat, p_t = ttest_1samp(values, popmean=reference, alternative=alternative)
    st.write("H₀: population mean AQI equals the reference value.")
    st.write(f"t statistic: **{t_stat:.4f}** · p-value: **{p_t:.6g}**")
    st.write("Decision: **Reject H₀**" if p_t < alpha else "Decision: **Fail to reject H₀**")
    st.markdown("### One-sample Z-test")
    st.info("A Z-test requires a known population standard deviation. Enter a value justified by an authoritative source; the default is only a demonstration.")
    sigma = st.number_input("Known population standard deviation (σ)", min_value=0.0001, value=max(sd, 0.0001), format="%.4f")
    z_stat = (mean-reference)/(sigma/np.sqrt(n))
    if alternative == "two-sided":
        p_z = 2*norm.sf(abs(z_stat))
    elif alternative == "greater":
        p_z = norm.sf(z_stat)
    else:
        p_z = norm.cdf(z_stat)
    st.write(f"Z statistic: **{z_stat:.4f}** · p-value: **{p_z:.6g}**")
    st.write("Decision: **Reject H₀**" if p_z < alpha else "Decision: **Fail to reject H₀**")
    st.caption("Records across locations and dates may not be independent. Interpret results with the data collection design in mind.")

elif page == "Lab 6A: Chi-Square Test":
    st.subheader("Lab 6A — Chi-Square Test of Independence")
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if len(categorical) < 2:
        st.error("At least two categorical columns are required.")
        st.stop()
    a_default = categorical.index("state") if "state" in categorical else 0
    b_default = categorical.index("air_quality_status") if "air_quality_status" in categorical else min(1, len(categorical)-1)
    c1, c2 = st.columns(2)
    with c1:
        col_a = st.selectbox("First categorical variable", categorical, index=a_default)
    with c2:
        col_b = st.selectbox("Second categorical variable", categorical, index=b_default)
    max_categories = st.slider("Maximum categories per variable", 3, 30, 12)
    pair = df[[col_a, col_b]].dropna()
    keep_a = pair[col_a].value_counts().head(max_categories).index
    keep_b = pair[col_b].value_counts().head(max_categories).index
    pair = pair[pair[col_a].isin(keep_a) & pair[col_b].isin(keep_b)]
    contingency = pd.crosstab(pair[col_a], pair[col_b])
    st.write("Contingency table (most frequent categories)")
    st.dataframe(contingency, use_container_width=True)
    if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        st.write(f"Chi-square statistic: **{chi2:.4f}**")
        st.write(f"Degrees of freedom: **{dof}**")
        st.write(f"p-value: **{p_value:.6g}**")
        alpha = st.select_slider("Significance level", options=[0.10, 0.05, 0.01], value=0.05, key="chi_alpha")
        st.write("Decision: **Evidence of association**" if p_value < alpha else "Decision: **Insufficient evidence of association**")
        if (expected < 5).any():
            st.warning("Some expected cell counts are below 5. Consider combining categories or using a more suitable method.")
    else:
        st.warning("Choose variables that produce at least a 2×2 contingency table.")

elif page == "Lab 6B/C: Covariance & Correlation":
    st.subheader("Lab 6 Part B/C — Covariance and Correlation Hypothesis Tests")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("At least two numerical columns are required.")
        st.stop()
    x_default = numeric_cols.index("aqi_value") if "aqi_value" in numeric_cols else 0
    y_default = numeric_cols.index("number_of_monitoring_stations") if "number_of_monitoring_stations" in numeric_cols else min(1, len(numeric_cols)-1)
    c1, c2 = st.columns(2)
    with c1:
        x_col = st.selectbox("Variable X", numeric_cols, index=x_default)
    with c2:
        y_col = st.selectbox("Variable Y", numeric_cols, index=y_default)
    pair = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if x_col == y_col:
        st.warning("Choose two different variables.")
    elif len(pair) < 3:
        st.warning("At least three complete pairs are required.")
    else:
        st.metric("Sample covariance", f"{pair[x_col].cov(pair[y_col]):.5f}")
        pearson_r, pearson_p = pearsonr(pair[x_col], pair[y_col])
        spearman_r, spearman_p = spearmanr(pair[x_col], pair[y_col])
        c1, c2 = st.columns(2)
        c1.metric("Pearson correlation (r)", f"{pearson_r:.5f}")
        c1.write(f"p-value: {pearson_p:.6g}")
        c2.metric("Spearman correlation (ρ)", f"{spearman_r:.5f}")
        c2.write(f"p-value: {spearman_p:.6g}")
        alpha = st.select_slider("Significance level", options=[0.10, 0.05, 0.01], value=0.05, key="corr_alpha")
        st.write("Pearson: **statistically significant**" if pearson_p < alpha else "Pearson: **not statistically significant**")
        st.write("Spearman: **statistically significant**" if spearman_p < alpha else "Spearman: **not statistically significant**")
        fig, ax = plt.subplots()
        ax.scatter(pair[x_col], pair[y_col], alpha=.3)
        ax.set(title=f"{x_col} vs {y_col}", xlabel=x_col, ylabel=y_col)
        st.pyplot(fig)
        st.caption("Pearson measures linear association; Spearman measures monotonic rank association. Correlation does not imply causation.")

elif page == "Practical 6: Two-Way ANOVA":
    st.subheader("Practical 6 — Two-Way ANOVA")
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if len(categorical) < 2:
        st.error("Two categorical factors are required.")
        st.stop()
    default_a = categorical.index("air_quality_status") if "air_quality_status" in categorical else 0
    default_b = categorical.index("prominent_pollutants") if "prominent_pollutants" in categorical else min(1, len(categorical)-1)
    c1, c2 = st.columns(2)
    with c1:
        factor_a = st.selectbox("Factor A", categorical, index=default_a)
    with c2:
        factor_b = st.selectbox("Factor B", categorical, index=default_b)
    max_levels = st.slider("Maximum levels per factor", 2, 12, 6)
    anova_df = df[["aqi_value", factor_a, factor_b]].dropna().copy()
    for col in [factor_a, factor_b]:
        top = anova_df[col].value_counts().head(max_levels).index
        anova_df = anova_df[anova_df[col].isin(top)]
    if factor_a == factor_b:
        st.warning("Select two different categorical factors.")
    elif anova_df[factor_a].nunique() < 2 or anova_df[factor_b].nunique() < 2:
        st.warning("Each factor needs at least two levels.")
    else:
        try:
            formula = f"aqi_value ~ C(Q('{factor_a}')) * C(Q('{factor_b}'))"
            model = smf.ols(formula, data=anova_df).fit()
            table = sm.stats.anova_lm(model, typ=2)
            st.dataframe(table, use_container_width=True)
            alpha = st.select_slider("Significance level", options=[0.10, 0.05, 0.01], value=0.05, key="anova_alpha")
            p_columns = [c for c in table.columns if c.lower().startswith("pr(>f)")]
            if p_columns:
                for term, row in table.iterrows():
                    if term != "Residual" and pd.notna(row[p_columns[0]]):
                        label = "statistically significant" if row[p_columns[0]] < alpha else "not statistically significant"
                        st.write(f"**{term}:** {label}")
            st.caption("Check independence, residual normality, variance homogeneity, and adequate observations in factor combinations.")
        except Exception as exc:
            st.error(f"Could not fit ANOVA for these factors. Try different factors or fewer levels. Details: {exc}")

elif page == "Lab 7: Regression":
    st.subheader("Lab 7 — Regression Analysis")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    predictors = [c for c in numeric_cols if c != "aqi_value"]
    if not predictors:
        st.error("No numeric predictor columns are available besides AQI.")
        st.stop()
    selected_predictors = st.multiselect("Select predictor variable(s)", predictors, default=predictors[:1])
    if not selected_predictors:
        st.warning("Select at least one predictor.")
        st.stop()
    reg_df = df[["aqi_value"] + selected_predictors].apply(pd.to_numeric, errors="coerce").dropna()
    if len(reg_df) < 10:
        st.warning("Not enough complete rows for regression.")
        st.stop()
    test_fraction = st.slider("Test set proportion", 0.1, 0.4, 0.2, step=0.05)
    random_state = st.number_input("Train/test split seed", 0, 99999, 42)
    X = reg_df[selected_predictors]
    y = reg_df["aqi_value"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=int(random_state))
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    c1, c2, c3 = st.columns(3)
    c1.metric("R² score", f"{r2_score(y_test, predictions):.4f}")
    c2.metric("MAE", f"{mean_absolute_error(y_test, predictions):.4f}")
    c3.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, predictions)):.4f}")
    st.write(f"Intercept: **{model.intercept_:.4f}**")
    st.dataframe(pd.DataFrame({"Predictor": selected_predictors, "Coefficient": model.coef_}), use_container_width=True)
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, alpha=.35)
    low = min(float(y_test.min()), float(np.min(predictions)))
    high = max(float(y_test.max()), float(np.max(predictions)))
    ax.plot([low, high], [low, high])
    ax.set(title="Actual vs Predicted AQI", xlabel="Actual AQI", ylabel="Predicted AQI")
    st.pyplot(fig)
    results = pd.DataFrame({"Actual AQI": y_test.to_numpy(), "Predicted AQI": predictions, "Residual": y_test.to_numpy()-predictions})
    st.dataframe(results.head(100), use_container_width=True)
    st.download_button("Download test predictions", results.to_csv(index=False).encode("utf-8"), "aqi_regression_predictions.csv", "text/csv")
    st.caption("Choose scientifically meaningful predictors. A low R² can be a valid finding if available variables explain little of the AQI variation.")


elif page == "Regression Model Comparison":
    st.subheader("Regression Model Comparison")
    st.write("Compare simple linear, multiple linear, polynomial, and random forest regression using the same train/test split.")
    st.info("This comparison uses numerical predictors available in the dataset. AQI is the target. Model scores are calculated live from your loaded data; they are not preset values.")

    # Keep only numeric predictors with enough valid, varying observations.
    # Empty columns such as `note` may be inferred as numeric by pandas but
    # cannot be used for modelling and would otherwise reduce complete rows to zero.
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    predictors = []
    for col in numeric_cols:
        if col == "aqi_value":
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) >= 20 and values.nunique() > 1:
            predictors.append(col)

    if not predictors:
        st.error("No usable numerical predictor columns are available besides AQI. Check the dataset for numeric columns with valid values.")
        st.stop()
    excluded_numeric = [c for c in numeric_cols if c != "aqi_value" and c not in predictors]
    if excluded_numeric:
        st.caption("Ignored empty or unusable numeric columns: " + ", ".join(excluded_numeric))

    simple_predictor = st.selectbox("Predictor for Simple Linear Regression", predictors, index=0)
    multi_predictors = st.multiselect(
        "Predictors for Multiple Linear / Polynomial / Random Forest models",
        predictors,
        default=predictors
    )
    if not multi_predictors:
        st.warning("Select at least one predictor for the multi-feature models.")
        st.stop()

    test_fraction = st.slider("Test set proportion", 0.1, 0.4, 0.2, step=0.05, key="compare_test_fraction")
    random_state = st.number_input("Random seed", min_value=0, max_value=99999, value=42, key="compare_seed")

    # Use the same target rows and split for fair model comparison.
    needed = list(dict.fromkeys(["aqi_value", simple_predictor] + multi_predictors))
    compare_df = df[needed].apply(pd.to_numeric, errors="coerce").dropna(subset=["aqi_value"]).copy()
    compare_df = compare_df.replace([np.inf, -np.inf], np.nan)
    compare_df = compare_df.dropna(subset=[simple_predictor] + multi_predictors)

    if len(compare_df) < 20:
        st.warning(f"Only {len(compare_df)} complete rows are available. At least 20 are recommended for a meaningful comparison.")
        st.stop()
    if compare_df["aqi_value"].nunique() < 2:
        st.warning("AQI must contain more than one distinct value.")
        st.stop()

    y = compare_df["aqi_value"]
    X_simple = compare_df[[simple_predictor]]
    X_multi = compare_df[multi_predictors]
    indices = np.arange(len(compare_df))
    train_idx, test_idx = train_test_split(indices, test_size=test_fraction, random_state=int(random_state))
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model_specs = [
        ("Simple Linear Regression", make_pipeline(SimpleImputer(strategy="median"), LinearRegression()), X_simple),
        ("Multiple Linear Regression", make_pipeline(SimpleImputer(strategy="median"), LinearRegression()), X_multi),
        ("Polynomial Regression (degree 2)", make_pipeline(SimpleImputer(strategy="median"), PolynomialFeatures(degree=2, include_bias=False), LinearRegression()), X_multi),
        ("Random Forest Regression", make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_estimators=150, random_state=int(random_state), min_samples_leaf=2, n_jobs=-1)), X_multi)
    ]

    rows = []
    fitted = {}
    predictions_by_model = {}
    for name, estimator, features in model_specs:
        X_train = features.iloc[train_idx]
        X_test = features.iloc[test_idx]
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_test)
        fitted[name] = estimator
        predictions_by_model[name] = pred
        rows.append({
            "Model": name,
            "R² (higher is better)": r2_score(y_test, pred),
            "MAE (lower is better)": mean_absolute_error(y_test, pred),
            "RMSE (lower is better)": np.sqrt(mean_squared_error(y_test, pred))
        })

    scores = pd.DataFrame(rows).sort_values("RMSE (lower is better)").reset_index(drop=True)
    st.markdown("### Performance comparison")
    st.dataframe(scores.style.format({
        "R² (higher is better)": "{:.4f}",
        "MAE (lower is better)": "{:.4f}",
        "RMSE (lower is better)": "{:.4f}"
    }), use_container_width=True)
    best = scores.iloc[0]
    st.success(f"Lowest test RMSE in this run: {best['Model']} (RMSE = {best['RMSE (lower is better)']:.4f}). This does not guarantee the model will perform best on future data.")

    metric = st.selectbox("Metric to visualize", ["R² (higher is better)", "MAE (lower is better)", "RMSE (lower is better)"], key="compare_metric")
    chart_scores = scores.sort_values(metric, ascending=(metric != "R² (higher is better)"))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(chart_scores["Model"], chart_scores[metric])
    ax.set_title(f"Model comparison: {metric}")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    st.pyplot(fig)

    chosen_model = st.selectbox("Show actual vs predicted for", list(predictions_by_model.keys()), key="compare_pred_model")
    chosen_pred = predictions_by_model[chosen_model]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, chosen_pred, alpha=0.35)
    lower = min(float(y_test.min()), float(np.min(chosen_pred)))
    upper = max(float(y_test.max()), float(np.max(chosen_pred)))
    ax.plot([lower, upper], [lower, upper], linestyle="--")
    ax.set(title=f"Actual vs Predicted AQI — {chosen_model}", xlabel="Actual AQI", ylabel="Predicted AQI")
    fig.tight_layout()
    st.pyplot(fig)

    pred_export = pd.DataFrame({"Actual AQI": y_test.to_numpy()})
    for name, pred in predictions_by_model.items():
        pred_export[f"Predicted — {name}"] = pred
    st.download_button("Download model comparison results", scores.to_csv(index=False).encode("utf-8"), "aqi_model_comparison.csv", "text/csv")
    st.download_button("Download test predictions", pred_export.to_csv(index=False).encode("utf-8"), "aqi_model_predictions.csv", "text/csv")
    st.caption("All models use the same held-out test records for fair comparison. A high R² or low error should be interpreted alongside data quality, leakage risks, and whether the predictors are available at prediction time.")
