import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Bias Detector",
    page_icon="🏦",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0f1117; }
    [data-testid="stSidebar"] { background: #1a1d27; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #3a3d5c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    .metric-card h2 { color: #7c8cff; font-size: 2.2rem; margin: 0; }
    .metric-card p  { color: #a0a3b1; margin: 4px 0 0; font-size: 0.9rem; }
    .section-title {
        color: #e0e3ff;
        font-size: 1.4rem;
        font-weight: 700;
        border-left: 4px solid #7c8cff;
        padding-left: 12px;
        margin: 24px 0 16px;
    }
    .badge-approved {
        background: #1a3d2b; color: #4ade80;
        padding: 2px 10px; border-radius: 20px; font-size: 0.8rem;
    }
    .badge-rejected {
        background: #3d1a1a; color: #f87171;
        padding: 2px 10px; border-radius: 20px; font-size: 0.8rem;
    }
    h1 { color: #e0e3ff !important; }
    h2, h3 { color: #c8cbee !important; }
    p, li { color: #a0a3b1 !important; }
    [data-testid="stMetricValue"] { color: #7c8cff !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD & PROCESS DATA (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_and_process(file):
    df = pd.read_csv(file)

    # Fill missing values
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

    # Feature engineering
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
    df['ApplicantIncome_Log'] = np.log(df['ApplicantIncome'] + 1)
    df['LoanAmount_Log'] = np.log(df['LoanAmount'] + 1)
    df['Total_Income_Log'] = np.log(df['Total_Income'] + 1)

    # Bias copy (before encoding)
    bias_df = df.copy()
    bias_df['Loan_Status_Num'] = bias_df['Loan_Status'].map({'Y': 1, 'N': 0})
    bias_df['Income_Group'] = pd.qcut(bias_df['Total_Income'], q=3, labels=['Low', 'Medium', 'High'])

    # Encode for modelling
    model_df = df.copy()
    model_df.drop('Loan_ID', axis=1, inplace=True)
    model_df['Loan_Status'] = model_df['Loan_Status'].map({'Y': 1, 'N': 0})
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    model_df = pd.get_dummies(model_df, columns=categorical_cols, drop_first=True)

    X = model_df.drop('Loan_Status', axis=1)
    y = model_df['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

    # Logistic Regression
    log_model = LogisticRegression(max_iter=2000, class_weight='balanced')
    log_model.fit(X_train_sm, y_train_sm)
    y_pred_log = log_model.predict(X_test_scaled)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
    rf_model.fit(X_train_sm, y_train_sm)
    y_pred_rf = rf_model.predict(X_test_scaled)

    return df, bias_df, X, y_test, y_pred_log, y_pred_rf, log_model, rf_model, scaler, X_train_sm, X_test_scaled


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Loan Bias Detector")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Overview",
        "🔍 Exploratory Analysis",
        "🤖 Model Results",
        "⚖️ Bias Analysis",
        "🎯 Predict Loan",
    ])
    st.markdown("---")
    st.markdown("<p style='font-size:0.75rem;color:#666;'>ABA Final Project · Loan Bias Detection</p>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
CSV_PATH = r"D:\ABA\2511004 - Final project\train_LOAN BIAS PROJECT.csv"

try:
    (df, bias_df, X, y_test,
     y_pred_log, y_pred_rf,
     log_model, rf_model, scaler,
     X_train_sm, X_test_scaled) = load_and_process(CSV_PATH)
except FileNotFoundError:
    st.error(f"❌ CSV file not found at:\n`{CSV_PATH}`\n\nPlease make sure the file exists at that path.")
    st.stop()


# ─────────────────────────────────────────────
# HELPER: dark-themed figure
# ─────────────────────────────────────────────
def dark_fig(w=8, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('#1e2130')
    ax.set_facecolor('#1e2130')
    ax.tick_params(colors='#a0a3b1')
    ax.xaxis.label.set_color('#a0a3b1')
    ax.yaxis.label.set_color('#a0a3b1')
    ax.title.set_color('#e0e3ff')
    for spine in ax.spines.values():
        spine.set_edgecolor('#3a3d5c')
    return fig, ax


# ═══════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# 🏦 Loan Bias Detection Dashboard")
    st.markdown("A complete ML pipeline to detect loan approval bias across gender & income groups.")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    approved = (df['Loan_Status'] == 'Y').sum()
    rejected = (df['Loan_Status'] == 'N').sum()

    with col1:
        st.markdown(f"""<div class="metric-card"><h2>{len(df)}</h2><p>Total Applicants</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card"><h2 style="color:#4ade80">{approved}</h2><p>Approved</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card"><h2 style="color:#f87171">{rejected}</h2><p>Rejected</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card"><h2>{df.shape[1]}</h2><p>Features</p></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Raw Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown('<div class="section-title">Missing Values</div>', unsafe_allow_html=True)
    miss = df.isnull().sum().reset_index()
    miss.columns = ['Column', 'Missing Count']
    miss = miss[miss['Missing Count'] > 0]
    if miss.empty:
        st.success("✅ No missing values in dataset.")
    else:
        st.dataframe(miss, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 2 — EXPLORATORY ANALYSIS
# ═══════════════════════════════════════════════
elif page == "🔍 Exploratory Analysis":
    st.markdown("# 🔍 Exploratory Data Analysis")
    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Loan Status Distribution</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(5, 4)
        counts = df['Loan_Status'].value_counts()
        ax.pie(counts, labels=['Approved', 'Rejected'], autopct='%1.1f%%',
               colors=['#4ade80', '#f87171'], startangle=90,
               textprops={'color': '#e0e3ff'})
        ax.set_title("Loan Status", color='#e0e3ff')
        st.pyplot(fig)
        plt.close()

    with c2:
        st.markdown('<div class="section-title">Applicant Income Distribution</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(5, 4)
        ax.hist(df['ApplicantIncome'], bins=30, color='#7c8cff', edgecolor='#3a3d5c')
        ax.set_title("Applicant Income", color='#e0e3ff')
        ax.set_xlabel("Income")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close()

    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-title">Loan Amount Distribution</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(5, 4)
        ax.hist(df['LoanAmount'].dropna(), bins=30, color='#a78bfa', edgecolor='#3a3d5c')
        ax.set_title("Loan Amount", color='#e0e3ff')
        ax.set_xlabel("Amount (₹ Thousands)")
        st.pyplot(fig)
        plt.close()

    with c4:
        st.markdown('<div class="section-title">Approval by Education</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(5, 4)
        edu = df.groupby('Education')['Loan_Status'].value_counts(normalize=True).unstack()
        edu.plot(kind='bar', ax=ax, color=['#f87171', '#4ade80'], edgecolor='#3a3d5c')
        ax.set_title("Approval Rate by Education", color='#e0e3ff')
        ax.set_xlabel("")
        ax.set_xticklabels(edu.index, rotation=0)
        ax.legend(['Rejected', 'Approved'], facecolor='#1e2130', labelcolor='#a0a3b1')
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    fig, ax = dark_fig(10, 5)
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, ax=ax, cmap='coolwarm', annot=True, fmt='.2f',
                linewidths=0.5, linecolor='#0f1117',
                annot_kws={'color': '#e0e3ff', 'size': 8})
    ax.set_title("Feature Correlation Matrix", color='#e0e3ff')
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════
# PAGE 3 — MODEL RESULTS
# ═══════════════════════════════════════════════
elif page == "🤖 Model Results":
    st.markdown("# 🤖 Model Results")
    st.markdown("---")

    log_acc = accuracy_score(y_test, y_pred_log)
    rf_acc  = accuracy_score(y_test, y_pred_rf)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="metric-card"><h2>{log_acc:.1%}</h2><p>Logistic Regression Accuracy</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card"><h2>{rf_acc:.1%}</h2><p>Random Forest Accuracy</p></div>""", unsafe_allow_html=True)

    # Confusion matrices
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Logistic Regression — Confusion Matrix</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(5, 4)
        sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d',
                    cmap='Blues', ax=ax, linewidths=0.5)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix – LR", color='#e0e3ff')
        st.pyplot(fig); plt.close()

    with c2:
        st.markdown('<div class="section-title">Random Forest — Confusion Matrix</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(5, 4)
        sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d',
                    cmap='Greens', ax=ax, linewidths=0.5)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix – RF", color='#e0e3ff')
        st.pyplot(fig); plt.close()

    # Classification report as table
    st.markdown('<div class="section-title">Classification Reports</div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Logistic Regression", "Random Forest"])
    with tab1:
        report_log = classification_report(y_test, y_pred_log, output_dict=True)
        st.dataframe(pd.DataFrame(report_log).transpose().round(3), use_container_width=True)
    with tab2:
        report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
        st.dataframe(pd.DataFrame(report_rf).transpose().round(3), use_container_width=True)

    # Feature importance
    st.markdown('<div class="section-title">Top 10 Feature Importances — Random Forest</div>', unsafe_allow_html=True)
    fi = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_}).sort_values('Importance', ascending=False).head(10)
    fig, ax = dark_fig(9, 4)
    sns.barplot(data=fi, x='Importance', y='Feature', ax=ax, palette='Blues_r')
    ax.set_title("Feature Importance", color='#e0e3ff')
    st.pyplot(fig); plt.close()

    # LR Coefficients
    st.markdown('<div class="section-title">Logistic Regression Coefficients</div>', unsafe_allow_html=True)
    coeff_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': log_model.coef_[0]}).sort_values('Coefficient', ascending=False)
    top10 = pd.concat([coeff_df.head(5), coeff_df.tail(5)])
    fig, ax = dark_fig(9, 4)
    colors = ['#4ade80' if c > 0 else '#f87171' for c in top10['Coefficient']]
    ax.barh(top10['Feature'], top10['Coefficient'], color=colors)
    ax.axvline(0, color='#a0a3b1', linewidth=0.8)
    ax.set_title("LR Coefficients (Top & Bottom 5)", color='#e0e3ff')
    st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════
# PAGE 4 — BIAS ANALYSIS
# ═══════════════════════════════════════════════
elif page == "⚖️ Bias Analysis":
    st.markdown("# ⚖️ Bias Analysis")
    st.markdown("---")

    # Gender bias
    st.markdown('<div class="section-title">Approval Rate by Gender</div>', unsafe_allow_html=True)
    gender_bias = bias_df.groupby('Gender')['Loan_Status_Num'].mean().reset_index()
    gender_bias.columns = ['Gender', 'Approval Rate']

    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(gender_bias.style.format({'Approval Rate': '{:.1%}'}), use_container_width=True)
        # Disparate Impact Ratio
        gb = bias_df.groupby('Gender')['Loan_Status_Num'].mean()
        if 'Female' in gb.index and 'Male' in gb.index:
            dir_val = gb['Female'] / gb['Male']
            color = "#4ade80" if dir_val >= 0.8 else "#f87171"
            st.markdown(f"""<div class="metric-card"><h2 style="color:{color}">{dir_val:.3f}</h2>
            <p>Disparate Impact Ratio (Female/Male)<br>Threshold: 0.8</p></div>""", unsafe_allow_html=True)
            if dir_val < 0.8:
                st.error("⚠️ Bias may exist against Female applicants (DIR < 0.8)")
            else:
                st.success("✅ No strong gender bias based on DIR")

    with c2:
        fig, ax = dark_fig(6, 4)
        bars = ax.bar(gender_bias['Gender'], gender_bias['Approval Rate'],
                      color=['#a78bfa', '#7c8cff'], edgecolor='#3a3d5c')
        ax.set_ylim(0, 1)
        ax.set_title("Loan Approval Rate by Gender", color='#e0e3ff')
        ax.set_ylabel("Approval Rate")
        ax.axhline(0.8, color='#f87171', linestyle='--', linewidth=1, label='80% threshold')
        ax.legend(facecolor='#1e2130', labelcolor='#a0a3b1')
        for bar, val in zip(bars, gender_bias['Approval Rate']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.1%}', ha='center', va='bottom', color='#e0e3ff', fontsize=11)
        st.pyplot(fig); plt.close()

    # Gender crosstab
    st.markdown('<div class="section-title">Gender vs Loan Status Cross-Tab</div>', unsafe_allow_html=True)
    gender_ct = pd.crosstab(bias_df['Gender'], bias_df['Loan_Status'], margins=True)
    st.dataframe(gender_ct, use_container_width=True)

    st.markdown("---")

    # Income Group Bias
    st.markdown('<div class="section-title">Approval Rate by Income Group</div>', unsafe_allow_html=True)
    income_bias = bias_df.groupby('Income_Group')['Loan_Status_Num'].mean().reset_index()
    income_bias.columns = ['Income Group', 'Approval Rate']

    c3, c4 = st.columns([1, 2])
    with c3:
        st.dataframe(income_bias.style.format({'Approval Rate': '{:.1%}'}), use_container_width=True)

    with c4:
        fig, ax = dark_fig(6, 4)
        bars = ax.bar(income_bias['Income Group'], income_bias['Approval Rate'],
                      color=['#f87171', '#fbbf24', '#4ade80'], edgecolor='#3a3d5c')
        ax.set_ylim(0, 1)
        ax.set_title("Approval Rate by Income Group", color='#e0e3ff')
        ax.set_ylabel("Approval Rate")
        for bar, val in zip(bars, income_bias['Approval Rate']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.1%}', ha='center', va='bottom', color='#e0e3ff', fontsize=11)
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">Income Group vs Loan Status Cross-Tab</div>', unsafe_allow_html=True)
    income_ct = pd.crosstab(bias_df['Income_Group'], bias_df['Loan_Status'], margins=True)
    st.dataframe(income_ct, use_container_width=True)

    # Property Area
    st.markdown('<div class="section-title">Approval Rate by Property Area</div>', unsafe_allow_html=True)
    area_bias = bias_df.groupby('Property_Area')['Loan_Status_Num'].mean().reset_index()
    area_bias.columns = ['Property Area', 'Approval Rate']
    fig, ax = dark_fig(7, 3)
    bars = ax.bar(area_bias['Property Area'], area_bias['Approval Rate'],
                  color=['#7c8cff', '#a78bfa', '#60a5fa'], edgecolor='#3a3d5c')
    ax.set_ylim(0, 1)
    ax.set_title("Approval Rate by Property Area", color='#e0e3ff')
    for bar, val in zip(bars, area_bias['Approval Rate']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', va='bottom', color='#e0e3ff', fontsize=11)
    st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════
# PAGE 5 — PREDICT LOAN
# ═══════════════════════════════════════════════
elif page == "🎯 Predict Loan":
    st.markdown("# 🎯 Predict Loan Approval")
    st.markdown("Fill in the applicant details below to get a prediction from both models.")
    st.markdown("---")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])

        with col2:
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            credit_history = st.selectbox("Credit History", [1.0, 0.0])

        with col3:
            applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=5000, step=500)
            coapplicant_income = st.number_input("Co-applicant Income (₹)", min_value=0, value=0, step=500)
            loan_amount = st.number_input("Loan Amount (₹ Thousands)", min_value=0, value=100, step=10)
            loan_term = st.selectbox("Loan Term (months)", [360, 180, 480, 300, 240, 84, 120, 60, 36, 12])

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        # Build input row
        total_income = applicant_income + coapplicant_income
        emi = loan_amount / loan_term if loan_term > 0 else 0
        input_dict = {
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_term,
            'Credit_History': credit_history,
            'Total_Income': total_income,
            'EMI': emi,
            'ApplicantIncome_Log': np.log(applicant_income + 1),
            'CoapplicantIncome_Log': np.log(coapplicant_income + 1),
            'LoanAmount_Log': np.log(loan_amount + 1),
            'Total_Income_Log': np.log(total_income + 1),
            # One-hot encoded columns
            'Gender_Male': 1 if gender == 'Male' else 0,
            'Married_Yes': 1 if married == 'Yes' else 0,
            'Dependents_1': 1 if dependents == '1' else 0,
            'Dependents_2': 1 if dependents == '2' else 0,
            'Dependents_3+': 1 if dependents == '3+' else 0,
            'Education_Not Graduate': 1 if education == 'Not Graduate' else 0,
            'Self_Employed_Yes': 1 if self_employed == 'Yes' else 0,
            'Property_Area_Semiurban': 1 if property_area == 'Semiurban' else 0,
            'Property_Area_Urban': 1 if property_area == 'Urban' else 0,
        }

        input_df = pd.DataFrame([input_dict])

        # Align columns with training data
        for col in X.columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[X.columns]

        input_scaled = scaler.transform(input_df)

        pred_log = log_model.predict(input_scaled)[0]
        prob_log = log_model.predict_proba(input_scaled)[0]
        pred_rf  = rf_model.predict(input_scaled)[0]
        prob_rf  = rf_model.predict_proba(input_scaled)[0]

        st.markdown("---")
        st.markdown("### 🔮 Prediction Results")
        col1, col2 = st.columns(2)

        def result_card(model_name, pred, prob):
            label = "✅ APPROVED" if pred == 1 else "❌ REJECTED"
            color = "#4ade80" if pred == 1 else "#f87171"
            conf = prob[1] if pred == 1 else prob[0]
            return f"""<div class="metric-card">
                <h3 style="color:#a0a3b1;font-size:1rem">{model_name}</h3>
                <h2 style="color:{color};font-size:1.8rem">{label}</h2>
                <p>Confidence: <strong style="color:{color}">{conf:.1%}</strong></p>
            </div>"""

        with col1:
            st.markdown(result_card("Logistic Regression", pred_log, prob_log), unsafe_allow_html=True)
        with col2:
            st.markdown(result_card("Random Forest", pred_rf, prob_rf), unsafe_allow_html=True)

        # Probability bars
        st.markdown("#### Approval Probability Breakdown")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("LR — P(Approved)", f"{prob_log[1]:.1%}")
            st.progress(float(prob_log[1]))
        with c2:
            st.metric("RF — P(Approved)", f"{prob_rf[1]:.1%}")
            st.progress(float(prob_rf[1]))