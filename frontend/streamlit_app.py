"""
Streamlit Frontend for Loan Default Prediction
Communicates with the FastAPI backend via HTTP requests.
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
import json

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
    .danger-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Helper functions to call the FastAPI backend
# -------------------------------------------------------------------

def api_health_check() -> dict | None:
    """Check if the backend API is healthy."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def api_predict_single(data: dict) -> dict | None:
    """Call the /predict endpoint for a single prediction."""
    try:
        resp = requests.post(f"{API_BASE_URL}/predict", json=data, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None


def api_batch_predict(file_bytes: bytes, filename: str) -> dict | None:
    """Call the /batch_predict endpoint with a CSV file."""
    try:
        files = {"file": (filename, file_bytes, "text/csv")}
        resp = requests.post(f"{API_BASE_URL}/batch_predict", files=files, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None


def api_model_info() -> dict | None:
    """Call the /model_info endpoint."""
    try:
        resp = requests.get(f"{API_BASE_URL}/model_info", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
# Title
st.markdown('<h1 class="main-header">🏦 Loan Default Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/bank.png", width=100)
    st.title("Navigation")
    
    page = st.radio(
        "Select Page",
        ["🏠 Home", "🔮 Single Prediction", "📊 Batch Prediction", "📈 Model Info"]
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This application predicts the probability of loan default "
        "using machine learning algorithms trained on historical data."
    )
    
    st.markdown("---")
    st.markdown("### Tech Stack")
    st.markdown("- **ML**: XGBoost")
    st.markdown("- **Tracking**: MLflow")
    st.markdown("- **API**: FastAPI")
    st.markdown("- **Frontend**: Streamlit")

    # API connection status
    st.markdown("---")
    health = api_health_check()
    if health:
        st.success("🟢 API Connected")
    else:
        st.error("🔴 API Offline — start the backend first")

# -------------------------------------------------------------------
# Home Page
# -------------------------------------------------------------------
if page == "🏠 Home":
    st.header("Welcome to Loan Default Prediction System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Model Algorithm",
            value="XGBoost",
            delta="Best Performance"
        )
    
    with col2:
        st.metric(
            label="ROC-AUC Score",
            value="0.86",
            delta="0.02"
        )
    
    with col3:
        st.metric(
            label="Training Samples",
            value="250K",
            delta="Balanced"
        )
    
    st.markdown("---")
    
    # Features
    st.subheader("📋 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ Capabilities
        - Single customer prediction
        - Batch prediction via CSV upload
        - Real-time probability calculation
        - Risk level categorization
        - Feature importance visualization
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Model Features
        - Credit utilization ratio
        - Payment history (30, 60, 90 days late)
        - Debt-to-income ratio
        - Number of credit lines
        - Monthly income
        - Age and dependents
        """)
    
    st.markdown("---")
    
    # How It Works
    st.subheader("🔍 How It Works")
    
    st.markdown("""
    1. **Input Data**: Provide borrower information
    2. **Feature Engineering**: System creates additional features
    3. **Model Prediction**: XGBoost model predicts default probability
    4. **Risk Assessment**: Categorizes risk level (Low/Medium/High)
    5. **Results**: Get detailed prediction with probabilities
    """)

# -------------------------------------------------------------------
# Single Prediction Page
# -------------------------------------------------------------------
elif page == "🔮 Single Prediction":
    st.header("Single Customer Prediction")
    st.markdown("Enter customer details to predict loan default probability")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=45, step=1)
            monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=9120, step=100)
            debt_ratio = st.number_input("Debt Ratio", min_value=0.0, max_value=10.0, value=0.80, step=0.01, format="%.2f")
            credit_util = st.number_input("Credit Utilization", min_value=0.0, max_value=5.0, value=0.77, step=0.01, format="%.2f")
            num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=2, step=1)
        
        with col2:
            num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, max_value=50, value=13, step=1)
            num_real_estate = st.number_input("Real Estate Loans", min_value=0, max_value=20, value=6, step=1)
            times_30_59_late = st.number_input("Times 30-59 Days Late", min_value=0, max_value=20, value=2, step=1)
            times_60_89_late = st.number_input("Times 60-89 Days Late", min_value=0, max_value=20, value=0, step=1)
            times_90_late = st.number_input("Times 90+ Days Late", min_value=0, max_value=20, value=0, step=1)
        
        submitted = st.form_submit_button("🔮 Predict Default Probability", use_container_width=True)
    
    if submitted:
        # Prepare input data (use alias names for the API)
        input_data = {
            "RevolvingUtilizationOfUnsecuredLines": credit_util,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": times_30_59_late,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": monthly_income,
            "NumberOfOpenCreditLinesAndLoans": num_credit_lines,
            "NumberOfTimes90DaysLate": times_90_late,
            "NumberRealEstateLoansOrLines": num_real_estate,
            "NumberOfTime60-89DaysPastDueNotWorse": times_60_89_late,
            "NumberOfDependents": num_dependents
        }
        
        # Make prediction via API
        with st.spinner("Making prediction..."):
            result = api_predict_single(input_data)
        
        if result:
            st.success("✅ Prediction Complete!")
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Prediction",
                    value=result['prediction_label']
                )
            
            with col2:
                st.metric(
                    label="Default Probability",
                    value=f"{result['probability_default']:.2%}"
                )
            
            with col3:
                st.metric(
                    label="Risk Level",
                    value=result['risk_level']
                )
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=result['probability_default'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Default Probability (%)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 60], 'color': "yellow"},
                        {'range': [60, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 60
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk interpretation
            if result['risk_level'] == 'Low Risk':
                st.markdown('<div class="success-box"><strong>✅ Low Risk:</strong> This borrower has a low probability of default. Recommend approval.</div>', unsafe_allow_html=True)
            elif result['risk_level'] == 'Medium Risk':
                st.markdown('<div class="warning-box"><strong>⚠️ Medium Risk:</strong> This borrower has moderate default risk. Further review recommended.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="danger-box"><strong>❌ High Risk:</strong> This borrower has a high probability of default. Recommend rejection or additional collateral.</div>', unsafe_allow_html=True)

# -------------------------------------------------------------------
# Batch Prediction Page
# -------------------------------------------------------------------
elif page == "📊 Batch Prediction":
    st.header("Batch Prediction")
    st.markdown("Upload a CSV file with customer data for batch predictions")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with customer data. Must contain all required columns."
    )
    
    if uploaded_file is not None:
        # Read CSV for preview
        df = pd.read_csv(uploaded_file)
        
        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown(f"**Total rows:** {len(df)}")
        
        # Validate columns
        required_columns = [
            'RevolvingUtilizationOfUnsecuredLines',
            'age',
            'NumberOfTime30-59DaysPastDueNotWorse',
            'DebtRatio',
            'MonthlyIncome',
            'NumberOfOpenCreditLinesAndLoans',
            'NumberOfTimes90DaysLate',
            'NumberRealEstateLoansOrLines',
            'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfDependents'
        ]
        
        missing_cols = set(required_columns) - set(df.columns)
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
        else:
            if st.button("🚀 Run Batch Prediction", use_container_width=True):
                with st.spinner("Making predictions..."):
                    # Reset file pointer and send raw bytes to API
                    uploaded_file.seek(0)
                    file_bytes = uploaded_file.read()
                    api_result = api_batch_predict(file_bytes, uploaded_file.name)
                
                if api_result:
                    predictions_list = api_result['predictions']
                    
                    st.success("✅ Batch prediction complete!")
                    
                    # Build results dataframe
                    pred_df = pd.DataFrame(predictions_list)
                    result_df = df.copy()
                    result_df['Prediction'] = pred_df['prediction']
                    result_df['Prediction_Label'] = pred_df['prediction_label']
                    result_df['Default_Probability'] = pred_df['probability_default']
                    result_df['Risk_Level'] = pred_df['risk_level']
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Predictions", len(result_df))
                    
                    with col2:
                        default_count = (result_df['Prediction'] == 1).sum()
                        st.metric("Predicted Defaults", default_count)
                    
                    with col3:
                        default_rate = (result_df['Prediction'] == 1).mean() * 100
                        st.metric("Default Rate", f"{default_rate:.1f}%")
                    
                    with col4:
                        avg_prob = result_df['Default_Probability'].mean() * 100
                        st.metric("Avg Default Prob", f"{avg_prob:.1f}%")
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.pie(
                            values=result_df['Prediction_Label'].value_counts().values,
                            names=result_df['Prediction_Label'].value_counts().index,
                            title="Prediction Distribution",
                            color_discrete_sequence=['#2ecc71', '#e74c3c']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.bar(
                            result_df['Risk_Level'].value_counts().sort_index(),
                            title="Risk Level Distribution",
                            labels={'value': 'Count', 'index': 'Risk Level'},
                            color=result_df['Risk_Level'].value_counts().sort_index().values,
                            color_continuous_scale=['green', 'yellow', 'red']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Results table
                    st.subheader("📊 Prediction Results")
                    st.dataframe(result_df, use_container_width=True)
                    
                    # Download button
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

# -------------------------------------------------------------------
# Model Info Page
# -------------------------------------------------------------------
elif page == "📈 Model Info":
    st.header("Model Information")
    
    info = api_model_info()
    
    if info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Details")
            st.markdown(f"""
            - **Model Type:** {info.get('model_type', 'N/A')}
            - **Model Path:** `{info.get('model_path', 'N/A')}`
            - **Features:** {info.get('features_count', 'N/A')}
            """)
        
        with col2:
            st.subheader("Performance Metrics")
            st.markdown("""
            - **ROC-AUC:** 0.86
            - **Precision:** 0.82
            - **Recall:** 0.78
            - **F1-Score:** 0.80
            """)
    else:
        st.warning("⚠️ Could not connect to the backend API. Start the FastAPI server first.")
    
    # Sample data
    st.subheader("📝 Sample Input Format")
    sample_df = pd.DataFrame({
        'RevolvingUtilizationOfUnsecuredLines': [0.77],
        'age': [45],
        'NumberOfTime30-59DaysPastDueNotWorse': [2],
        'DebtRatio': [0.80],
        'MonthlyIncome': [9120],
        'NumberOfOpenCreditLinesAndLoans': [13],
        'NumberOfTimes90DaysLate': [0],
        'NumberRealEstateLoansOrLines': [6],
        'NumberOfTime60-89DaysPastDueNotWorse': [0],
        'NumberOfDependents': [2]
    })
    st.dataframe(sample_df, use_container_width=True)

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        <p>🏦 Loan Default Prediction System | Built with FastAPI & Streamlit | © 2024</p>
    </div>
    """,
    unsafe_allow_html=True
)
