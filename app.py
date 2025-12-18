
import streamlit as st
import pandas as pd
import numpy as np

# Page config
st.set_page_config(
    page_title="Justice Gap Analyzer",
    page_icon="⚖️",
    layout="wide"
)

# Title
st.title("⚖️ Justice Gap Analyzer - Working Version")
st.markdown("**Using FP-Growth Association Rule Mining**")

# Try to load the dataset
try:
    df = pd.read_csv("legal_aid_dataset.csv")
    st.success(f"✅ Data loaded successfully! {len(df)} records found")
    
    # Show basic info
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10))
    
    # Basic statistics
    st.subheader("📈 Key Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cases", f"{len(df):,}")
    
    with col2:
        if 'Case_Type' in df.columns:
            st.metric("Case Types", df['Case_Type'].nunique())
    
    with col3:
        if 'Delay_Flag' in df.columns:
            delay_rate = df['Delay_Flag'].mean() * 100
            st.metric("Delay Rate", f"{delay_rate:.1f}%")
    
    with col4:
        if 'Gender' in df.columns:
            st.metric("Gender Ratio", f"{df['Gender'].value_counts().to_dict()}")
    
    # Simple analysis
    if 'Case_Type' in df.columns and 'Delay_Flag' in df.columns:
        st.subheader("🔍 Delay Analysis by Case Type")
        delay_by_type = df.groupby('Case_Type')['Delay_Flag'].mean().sort_values(ascending=False)
        
        # Convert to DataFrame for display
        delay_df = pd.DataFrame({
            'Case Type': delay_by_type.index,
            'Delay Rate (%)': (delay_by_type.values * 100).round(1)
        })
        
        st.dataframe(delay_df)
        
        # Simple bar chart
        st.bar_chart(delay_by_type)
    
    # Show column information
    with st.expander("📋 View All Columns"):
        st.write("Dataset columns:", df.columns.tolist())
        st.write("Data types:", df.dtypes)
        
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    
    # Create sample data if file doesn't exist
    st.info("Creating sample data for demonstration...")
    
    sample_data = {
        'Case_ID': range(1, 101),
        'Case_Type': ['Domestic_Violence'] * 40 + ['Property_Dispute'] * 30 + ['Criminal'] * 20 + ['Civil'] * 10,
        'Gender': ['Female'] * 60 + ['Male'] * 40,
        'Income_Bracket': ['Below_Poverty_Line'] * 40 + ['Low_Income'] * 30 + ['Middle_Income'] * 20 + ['High_Income'] * 10,
        'State': ['State_1'] * 25 + ['State_2'] * 25 + ['State_3'] * 25 + ['State_4'] * 25,
        'Delay_Flag': [1] * 72 + [0] * 28,  # 72% delay as per your paper
        'Outcome_Disparity': [1] * 35 + [0] * 65,
        'Hearing_Count': np.random.randint(1, 10, 100),
        'Resolution_Time': ['>24 mo'] * 72 + ['12-24 mo'] * 20 + ['6-12 mo'] * 8
    }
    
    df = pd.DataFrame(sample_data)
    st.success("✅ Sample data created for demonstration")
    
    # Show the sample data
    st.dataframe(df.head())

# Check installed packages
st.subheader("🛠️ System Check")
col1, col2, col3 = st.columns(3)

with col1:
    try:
        import sklearn
        st.success("scikit-learn ✓")
    except:
        st.error("scikit-learn ✗")

with col2:
    try:
        import mlxtend
        st.success("mlxtend ✓")
    except:
        st.error("mlxtend ✗")

with col3:
    try:
        import plotly
        st.success("plotly ✓")
    except:
        st.error("plotly ✗")

# Instructions
with st.expander("📖 How to Use This App"):
    st.markdown("""
    1. **View Data**: The dataset is automatically loaded and displayed
    2. **Analyze**: Basic statistics and delay analysis are shown
    3. **Next Steps**: Once this version works, we'll add:
       - FP-Growth algorithm
       - Association rule mining
       - Advanced visualizations
       - Justice gap detection
    """)

# Footer
st.markdown("---")
st.markdown("**Systemic Justice Gap Identification** • *Basic Working Version*")
