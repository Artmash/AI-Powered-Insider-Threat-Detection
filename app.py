import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Insider Threat Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ AI-Powered Insider Threat Detection System")
st.markdown("Upload an employee's behavioral data or select a sample profile to detect potential insider threats.")

# Load model, scaler, and feature columns
@st.cache_resource
def load_artifacts():
    model = joblib.load('insider_threat_model.pkl')
    scaler = joblib.load('scaler.pkl')
    with open('feature_columns.json', 'r') as f:
        features = json.load(f)
    return model, scaler, features

model, scaler, feature_columns = load_artifacts()

# Define baseline normal values (median from training data)
baseline_values = {
    'employee_seniority_years': 5,
    'is_contractor': 0,
    'employee_classification': 2,
    'has_foreign_citizenship': 0,
    'has_criminal_record': 0,
    'has_medical_history': 0,
    'total_printed_pages': 15,
    'num_printed_pages_off_hours': 5,
    'total_files_burned': 2,
    'burned_from_other': 1,
    'is_abroad': 0,
    'trip_day_number': 0,
    'hostility_country_level': 1,
    'num_entries': 3,
    'num_unique_campus': 1,
    'late_exit_flag': 0,
    'entry_during_weekend': 0
}

# Define feature categories for risk breakdown
feature_categories = {
    'Access Patterns': ['num_entries', 'num_unique_campus', 'late_exit_flag', 'entry_during_weekend'],
    'Data Transfer': ['total_files_burned', 'burned_from_other'],
    'Printing Activity': ['total_printed_pages', 'num_printed_pages_off_hours'],
    'Employee Profile': ['employee_seniority_years', 'is_contractor', 'employee_classification'],
    'Risk Factors': ['has_foreign_citizenship', 'has_criminal_record', 'has_medical_history', 'is_abroad', 'hostility_country_level', 'trip_day_number']
}

# Define risk indicators for color-coded display
risk_indicators = [
    ('late_exit_flag', 'Late exit from building', 'Leaving after normal hours'),
    ('entry_during_weekend', 'Weekend entry', 'Accessing facility on non-working days'),
    ('total_files_burned', 'High file transfers', 'More than 30 files transferred'),
    ('num_printed_pages_off_hours', 'High off-hours printing', 'More than 50 pages printed after hours'),
    ('is_abroad', 'International travel', 'Employee is currently abroad'),
    ('has_criminal_record', 'Criminal record', 'Employee has prior criminal history'),
    ('has_foreign_citizenship', 'Foreign citizenship', 'Employee holds foreign citizenship'),
    ('late_exit_flag', 'Unusual timing', 'Building entry/exit at abnormal times')
]

# Sidebar for input
st.sidebar.header("Input Options")

# Define enhanced sample profiles with descriptions
sample_profiles = {
    "Normal Employee (Low Risk)": {
        "description": "Typical employee with regular work hours, minimal data transfers, no risk factors.",
        "values": {
            'employee_seniority_years': 5, 'is_contractor': 0, 'employee_classification': 2,
            'has_foreign_citizenship': 0, 'has_criminal_record': 0, 'has_medical_history': 0,
            'total_printed_pages': 10, 'num_printed_pages_off_hours': 0, 'total_files_burned': 0,
            'burned_from_other': 0, 'is_abroad': 0, 'trip_day_number': 0, 'hostility_country_level': 1,
            'num_entries': 2, 'num_unique_campus': 1, 'late_exit_flag': 0, 'entry_during_weekend': 0
        }
    },
    "Suspicious Employee (High Risk)": {
        "description": "Contractor with high data exfiltration, off-hours access, international travel.",
        "values": {
            'employee_seniority_years': 2, 'is_contractor': 1, 'employee_classification': 4,
            'has_foreign_citizenship': 1, 'has_criminal_record': 0, 'has_medical_history': 0,
            'total_printed_pages': 150, 'num_printed_pages_off_hours': 120, 'total_files_burned': 45,
            'burned_from_other': 30, 'is_abroad': 1, 'trip_day_number': 5, 'hostility_country_level': 3,
            'num_entries': 20, 'num_unique_campus': 5, 'late_exit_flag': 1, 'entry_during_weekend': 1
        }
    },
    "Disgruntled Employee (Medium Risk)": {
        "description": "Long-term employee with medical history, moderate data transfers, late exits.",
        "values": {
            'employee_seniority_years': 8, 'is_contractor': 0, 'employee_classification': 3,
            'has_foreign_citizenship': 0, 'has_criminal_record': 0, 'has_medical_history': 1,
            'total_printed_pages': 80, 'num_printed_pages_off_hours': 40, 'total_files_burned': 15,
            'burned_from_other': 5, 'is_abroad': 0, 'trip_day_number': 0, 'hostility_country_level': 2,
            'num_entries': 12, 'num_unique_campus': 3, 'late_exit_flag': 1, 'entry_during_weekend': 0
        }
    },
    "Privileged User Abusing Access (High Risk)": {
        "description": "Senior employee with high clearance accessing unusual resources.",
        "values": {
            'employee_seniority_years': 12, 'is_contractor': 0, 'employee_classification': 5,
            'has_foreign_citizenship': 0, 'has_criminal_record': 0, 'has_medical_history': 0,
            'total_printed_pages': 5, 'num_printed_pages_off_hours': 0, 'total_files_burned': 60,
            'burned_from_other': 45, 'is_abroad': 0, 'trip_day_number': 0, 'hostility_country_level': 1,
            'num_entries': 8, 'num_unique_campus': 2, 'late_exit_flag': 1, 'entry_during_weekend': 1
        }
    },
    "Careless Employee (Low Risk)": {
        "description": "Well-intentioned employee with poor security habits (printing, after-hours).",
        "values": {
            'employee_seniority_years': 3, 'is_contractor': 0, 'employee_classification': 1,
            'has_foreign_citizenship': 0, 'has_criminal_record': 0, 'has_medical_history': 0,
            'total_printed_pages': 45, 'num_printed_pages_off_hours': 35, 'total_files_burned': 3,
            'burned_from_other': 0, 'is_abroad': 0, 'trip_day_number': 0, 'hostility_country_level': 1,
            'num_entries': 10, 'num_unique_campus': 2, 'late_exit_flag': 1, 'entry_during_weekend': 0
        }
    }
}

# Option 1: Upload CSV
uploaded_file = st.sidebar.file_uploader("Upload a CSV file (single row with feature columns)", type=["csv"])

# Option 2: Select sample profile
selected_sample = st.sidebar.selectbox("Or select a sample employee profile", list(sample_profiles.keys()))

# Show sample description
if selected_sample in sample_profiles:
    st.sidebar.info(f"**Profile Info:** {sample_profiles[selected_sample]['description']}")

# Button to trigger prediction
if st.sidebar.button("🔍 Analyze Behavior", type="primary"):
    
    # Get input values
    if uploaded_file is not None:
        # Read uploaded CSV
        input_df = pd.read_csv(uploaded_file)
        # Check if required columns exist
        missing_cols = [col for col in feature_columns if col not in input_df.columns]
        if missing_cols:
            st.error(f"CSV missing required columns: {missing_cols}")
            st.stop()
        input_values = input_df[feature_columns].iloc[0].values
        st.success(f"Loaded data from: {uploaded_file.name}")
    else:
        # Use selected sample
        input_values = list(sample_profiles[selected_sample]['values'].values())
        st.success(f"Loaded sample profile: {selected_sample}")
    
    # Scale input
    input_scaled = scaler.transform([input_values])
    
    # Predict
    probability = model.predict_proba(input_scaled)[0][1]
    prediction = 1 if probability >= 0.6 else 0
    confidence =probability  
    
    # Get feature importance from model
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    # ========== MAIN DISPLAY AREA ==========
    
    # Row 1: Detection Result (Large, prominent)
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if prediction == 1:
            st.error("## 🚨 MALICIOUS ACTIVITY DETECTED")
            st.markdown("### ⚠️ Immediate investigation recommended")
        else:
            st.success("## ✅ NORMAL / BENIGN BEHAVIOUR")
            st.markdown("### No immediate action required")
    
    with col2:
        st.metric("Confidence Score", f"{confidence:.2%}", 
                  delta="High certainty" if confidence > 0.85 else "Moderate certainty" if confidence > 0.6 else "Low certainty")
    
    with col3:
        risk_level = "Critical" if confidence > 0.9 else "High" if confidence > 0.7 else "Medium" if confidence > 0.5 else "Low"
        st.metric("Risk Level", risk_level)
    
    st.markdown("---")
    
    # Row 2: Two columns for explanations
    left_col, right_col = st.columns(2)
    
    with left_col:
        st.subheader("📊 Key Risk Indicators")
        # Show top 5 most important features
        top5 = importance_df.head(5)
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['coral' if x > 0.15 else 'steelblue' for x in top5['Importance']]
        ax.barh(top5['Feature'], top5['Importance'], color=colors)
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 5 Factors Influencing This Decision')
        ax.invert_yaxis()
        st.pyplot(fig)
    
    with right_col:
        st.subheader("📝 Explanation for Manager")
        
        # Identify which top features were abnormal
        abnormal_features = []
        for _, row in top5.iterrows():
            feature = row['Feature']
            idx = feature_columns.index(feature)
            actual = input_values[idx]
            baseline = baseline_values.get(feature, 0)
            if feature in ['late_exit_flag', 'entry_during_weekend', 'is_abroad'] and actual == 1:
                abnormal_features.append(feature.replace('_', ' ').title())
            elif actual > baseline * 1.5:
                abnormal_features.append(feature.replace('_', ' ').title())
        
        if prediction == 1:
            st.markdown("**Why this was flagged as Malicious:**")
            if abnormal_features:
                st.markdown("The following unusual behaviors were detected:")
                for feat in abnormal_features[:5]:
                    st.markdown(f"- ⚠️ **{feat}**")
            st.markdown("\nThese behavioral patterns are consistent with known insider threat indicators such as data exfiltration, credential abuse, or unauthorized access.")
        else:
            st.markdown("**Why this is considered Normal:**")
            st.markdown("The employee's behavior falls within expected patterns for their role and seniority level.")
            if not abnormal_features:
                st.markdown("- No significant deviations from baseline behavior")
            else:
                st.markdown("- Minor deviations detected but below risk threshold")
        
        st.caption(f"Confidence: {confidence:.2%} | Model: {type(model).__name__}")
    
    st.markdown("---")
    
    # Row 3: Color-coded Risk Indicators
    st.subheader("⚠️ Risk Indicator Summary")
    
    risk_cols = st.columns(2)
    risk_count = 0
    
    # Define thresholds for numeric indicators
    risk_thresholds = {
        'total_files_burned': 30,
        'num_printed_pages_off_hours': 50,
        'total_printed_pages': 100,
        'trip_day_number': 3,
        'hostility_country_level': 2,
        'num_entries': 15,
        'num_unique_campus': 4
    }
    
    for i, (feature, label, description) in enumerate(risk_indicators[:8]):  # Limit to 8 indicators
        if feature in feature_columns:
            idx = feature_columns.index(feature)
            value = input_values[idx]
            
            # Determine if risky
            if feature in risk_thresholds:
                is_risky = value > risk_thresholds[feature]
            elif feature in ['late_exit_flag', 'entry_during_weekend', 'is_abroad', 'has_criminal_record', 'has_foreign_citizenship']:
                is_risky = value == 1
            else:
                is_risky = False
            
            if is_risky:
                risk_count += 1
            col_idx = i % 2
            with risk_cols[col_idx]:
                if is_risky:
                    st.markdown(f"🔴 **{label}** - {description}")
                else:
                    st.markdown(f"🟢 **{label}** - Normal")
    
    if risk_count == 0:
        st.info("✅ No high-risk indicators detected")
    elif risk_count <= 2:
        st.warning(f"⚠️ {risk_count} risk indicator(s) detected - Monitor closely")
    else:
        st.error(f"🚨 {risk_count} risk indicator(s) detected - High concern")
    
    st.markdown("---")
    
    # Row 4: Risk Score Breakdown by Category (Bar Chart)
    st.subheader("📈 Risk Score Breakdown by Category")
    
    # Calculate risk score per category (0-100 scale)
    category_scores = {}
    for cat, features in feature_categories.items():
        cat_sum = 0
        cat_count = 0
        for f in features:
            if f in feature_columns:
                idx = feature_columns.index(f)
                value = input_values[idx]
                # Normalize based on typical ranges
                if f in risk_thresholds:
                    normalized = min(value / risk_thresholds[f], 1.0)
                elif f in ['late_exit_flag', 'entry_during_weekend', 'is_abroad', 'has_criminal_record', 'has_foreign_citizenship']:
                    normalized = value  # 0 or 1
                elif f == 'employee_classification':
                    normalized = min(value / 5, 1.0)
                elif f == 'employee_seniority_years':
                    normalized = min(value / 15, 1.0)
                else:
                    normalized = min(value / 50, 1.0)
                cat_sum += normalized
                cat_count += 1
        category_scores[cat] = min((cat_sum / cat_count) * 100 if cat_count > 0 else 0, 100)
    
    # Display as horizontal bar chart
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    categories = list(category_scores.keys())
    scores = list(category_scores.values())
    colors_bar = ['red' if s > 70 else 'orange' if s > 40 else 'green' for s in scores]
    ax2.barh(categories, scores, color=colors_bar)
    ax2.set_xlabel('Risk Score (0-100)')
    ax2.set_title('Risk Profile by Category')
    ax2.set_xlim(0, 100)
    for i, (cat, score) in enumerate(category_scores.items()):
        ax2.text(score + 2, i, f'{score:.0f}', va='center')
    st.pyplot(fig2)
    
    # Row 5: Deviation Analysis
    st.subheader("🔍 Detailed Deviation Analysis")
    
    # Show which features deviated significantly from baseline
    deviations = []
    for i, feature in enumerate(feature_columns):
        actual = input_values[i]
        baseline = baseline_values.get(feature, 0)
        if feature in ['late_exit_flag', 'entry_during_weekend', 'is_abroad', 'has_criminal_record', 'has_foreign_citizenship']:
            if actual == 1 and baseline == 0:
                deviations.append((feature, actual, baseline, "Present (Risk Indicator)"))
        elif actual > baseline * 1.5 and baseline > 0:
            pct_increase = ((actual - baseline) / baseline) * 100
            deviations.append((feature, actual, baseline, f"{pct_increase:.0f}% above normal"))
        elif actual < baseline * 0.5 and baseline > 0:
            pct_decrease = ((baseline - actual) / baseline) * 100
            deviations.append((feature, actual, baseline, f"{pct_decrease:.0f}% below normal"))
    
    if deviations:
        st.markdown("**Behaviors that deviated from normal patterns:**")
        dev_df = pd.DataFrame(deviations, columns=['Feature', 'Actual', 'Baseline', 'Deviation'])
        dev_df['Feature'] = dev_df['Feature'].str.replace('_', ' ').str.title()
        st.dataframe(dev_df, use_container_width=True)
    else:
        st.info("No significant deviations from baseline behavior detected.")
    
    # Row 6: Raw Input Data (Collapsible)
    with st.expander("📄 View Raw Input Data"):
        input_dict = dict(zip(feature_columns, [float(x) if isinstance(x, (int, float)) else x for x in input_values]))
        st.json(input_dict)
    
    # Footer with recommendations
    st.markdown("---")
    if prediction == 1:
        st.warning("""
        **Recommended Actions:**
        1. Review employee's recent activity logs
        2. Verify all data access requests were authorized
        3. Contact employee's manager for context
        4. Consider temporary access restrictions if suspicious
        """)
    else:
        st.info("""
        **Recommendation:** No immediate action required. Continue routine monitoring.
        """)

# Sidebar footer with model info
st.sidebar.markdown("---")
st.sidebar.caption(f"**Model:** {type(model).__name__}")
st.sidebar.caption("**Features:** 17 behavioral indicators")
st.sidebar.caption("**Explainability:** Feature importance + Deviation analysis")