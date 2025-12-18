
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
import os

# Page config
st.set_page_config(
    page_title="Justice Gap Analyzer",
    page_icon="⚖️",
    layout="wide"
)

# Title
st.title("⚖️ Systemic Justice Gap Analyzer")
st.markdown("**Using FP-Growth Association Rule Mining with Lift-Based Filtering**")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Legal Aid Dataset", type=['csv'])
    
    # Parameters
    st.subheader("FP-Growth Parameters")
    min_support = st.slider("Minimum Support", 0.01, 0.1, 0.025, 0.005)
    min_confidence = st.slider("Minimum Confidence", 0.1, 0.9, 0.55, 0.05)
    min_lift = st.slider("Minimum Lift", 1.0, 3.0, 1.3, 0.1)
    
    # Load sample data
    use_sample = st.checkbox("Use Sample Data", value=True)

# Main content
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(df)} records")
elif use_sample:
    try:
        df = pd.read_csv("legal_aid_dataset.csv")
        st.success(f"✅ Loaded sample dataset with {len(df)} records")
    except:
        st.error("❌ Sample dataset not found. Please upload a file.")
        df = None
else:
    st.info("👈 Please upload a dataset or check 'Use Sample Data'")
    df = None

if df is not None:
    # Show dataset
    with st.expander("View Dataset"):
        st.dataframe(df.head(100))
    
    # Basic stats
    st.subheader("Dataset Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cases", f"{len(df):,}")
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        if 'Delay_Flag' in df.columns:
            delay_rate = df['Delay_Flag'].mean() * 100
            st.metric("Delay Rate", f"{delay_rate:.1f}%")
    with col4:
        if 'Case_Type' in df.columns:
            st.metric("Case Types", df['Case_Type'].nunique())
    
    # Simple analysis
    st.subheader("Basic Analysis")
    
    if 'Case_Type' in df.columns and 'Delay_Flag' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Delay by case type
            delay_by_case = df.groupby('Case_Type')['Delay_Flag'].mean().reset_index()
            delay_by_case = delay_by_case.sort_values('Delay_Flag', ascending=False)
            st.bar_chart(delay_by_case.set_index('Case_Type'))
            st.caption("Delay Rate by Case Type")
        
        with col2:
            # Case type distribution
            case_dist = df['Case_Type'].value_counts()
            st.dataframe(case_dist)
            st.caption("Case Type Distribution")
    
    # Try to import and run analyzer
    try:
        # Add current directory to path to find modules
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from preprocessor import LegalAidPreprocessor
        from analyzer import JusticeGapAnalyzer
        
        st.subheader("Advanced Analysis")
        
        if st.button("Run FP-Growth Analysis", type="primary"):
            with st.spinner("Processing data and mining association rules..."):
                # Preprocess
                preprocessor = LegalAidPreprocessor()
                df_processed, transactions = preprocessor.preprocess_data(df)
                
                # Analyze
                analyzer = JusticeGapAnalyzer(
                    min_support=min_support,
                    min_confidence=min_confidence,
                    min_lift=min_lift
                )
                
                rules_df = analyzer.fit(transactions)
                
                if rules_df is not None and len(rules_df) > 0:
                    st.success(f"✅ Found {len(rules_df)} association rules!")
                    
                    # Show top rules
                    st.subheader(f"Top {min(10, len(rules_df))} Association Rules")
                    
                    for i, (idx, rule) in enumerate(rules_df.head(10).iterrows()):
                        with st.expander(f"Rule {i+1}: Lift = {rule['lift']:.2f}"):
                            st.write(f"**IF:** {list(rule['antecedents'])}")
                            st.write(f"**THEN:** {list(rule['consequents'])}")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Support", f"{rule['support']:.3f}")
                            col2.metric("Confidence", f"{rule['confidence']:.2%}")
                            col3.metric("Lift", f"{rule['lift']:.2f}")
                else:
                    st.warning("⚠️ No rules found. Try adjusting parameters.")
    
    except Exception as e:
        st.error(f"Error loading analysis modules: {str(e)}")
        st.info("Make sure `preprocessor.py` and `analyzer.py` are in the same directory.")
        
        # Show file structure
        with st.expander("Debug: Check files in directory"):
            import os
            files = os.listdir('.')
            st.write("Files in current directory:", files)

# Footer
st.markdown("---")
st.markdown("**Systemic Justice Gap Identification using FP-Growth Association Rule Mining**")
st.markdown("*Created for public legal aid case analysis*")
