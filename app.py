
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Page configuration
st.set_page_config(
    page_title="Justice Gap Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin-bottom: 1rem;
    }
    .rule-card {
        background-color: #fff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">⚖️ Systemic Justice Gap Analyzer</h1>', unsafe_allow_html=True)
st.markdown("**Using FP-Growth Association Rule Mining with Lift-Based Filtering**")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Legal Aid Dataset", type=['csv', 'xlsx'])
    
    # Algorithm parameters
    st.subheader("FP-Growth Parameters")
    min_support = st.slider("Minimum Support", 0.01, 0.1, 0.025, 0.005)
    min_confidence = st.slider("Minimum Confidence", 0.1, 0.9, 0.55, 0.05)
    min_lift = st.slider("Minimum Lift", 1.0, 3.0, 1.3, 0.1)
    
    # Analysis filters
    st.subheader("Filters")
    gap_type_filter = st.multiselect(
        "Filter by Justice Gap Type",
        ["Demographic Disparity", "Regional Disparity", "Case-Type Bias", "Process-Stage Bottleneck"]
    )
    
    # Load sample data
    if st.button("Load Sample Data"):
        st.session_state.sample_loaded = True
    
    st.markdown("---")
    st.info("""
    **About:**
    This tool analyzes public legal aid cases to identify systemic justice gaps using association rule mining.
    """)

# Main content
if uploaded_file is not None or 'sample_loaded' in st.session_state:
    # Load data
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    else:
        # Load sample data
        df = pd.read_csv('legal_aid_dataset.csv')
    
    # Display dataset info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cases", f"{len(df):,}")
    with col2:
        st.metric("Case Types", df['Case_Type'].nunique())
    with col3:
        st.metric("States/Jurisdictions", df['State'].nunique())
    with col4:
        delay_rate = df['Delay_Flag'].mean() * 100
        st.metric("Delay Rate", f"{delay_rate:.1f}%")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔍 Rule Mining", 
        "⚖️ Justice Gaps", 
        "📈 Visualizations",
        "📋 Insights"
    ])
    
    with tab1:
        st.subheader("Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Case type distribution
            case_dist = df['Case_Type'].value_counts()
            fig1 = px.pie(values=case_dist.values, 
                         names=case_dist.index,
                         title="Case Type Distribution")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Delay by case type
            delay_by_case = df.groupby('Case_Type')['Delay_Flag'].mean().reset_index()
            fig2 = px.bar(delay_by_case, x='Case_Type', y='Delay_Flag',
                         title="Delay Rate by Case Type")
            fig2.update_yaxis(tickformat=".0%")
            st.plotly_chart(fig2, use_container_width=True)
        
        # Demographics overview
        st.subheader("Demographic Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender_delay = df.groupby('Gender')['Delay_Flag'].mean().reset_index()
            fig3 = px.bar(gender_delay, x='Gender', y='Delay_Flag',
                         title="Delay Rate by Gender")
            fig3.update_yaxis(tickformat=".0%")
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            income_delay = df.groupby('Income_Bracket')['Delay_Flag'].mean().reset_index()
            fig4 = px.bar(income_delay, x='Income_Bracket', y='Delay_Flag',
                         title="Delay Rate by Income")
            fig4.update_yaxis(tickformat=".0%")
            st.plotly_chart(fig4, use_container_width=True)
        
        with col3:
            vulnerability_delay = df.groupby('Vulnerability_Category')['Delay_Flag'].mean().reset_index()
            fig5 = px.bar(vulnerability_delay, x='Vulnerability_Category', y='Delay_Flag',
                         title="Delay Rate by Vulnerability")
            fig5.update_yaxis(tickformat=".0%")
            st.plotly_chart(fig5, use_container_width=True)
    
    with tab2:
        st.subheader("Association Rule Mining")
        
        # Run FP-Growth
        if st.button("Run FP-Growth Analysis", type="primary"):
            with st.spinner("Mining association rules..."):
                # Preprocess data
                from preprocessor import LegalAidPreprocessor
                preprocessor = LegalAidPreprocessor()
                df_processed, transactions = preprocessor.preprocess_data(df)
                
                # Run FP-Growth
                from analyzer import JusticeGapAnalyzer
                analyzer = JusticeGapAnalyzer(
                    min_support=min_support,
                    min_confidence=min_confidence,
                    min_lift=min_lift
                )
                
                rules_df = analyzer.fit(transactions)
                justice_gaps_df = analyzer.analyze_justice_gaps(df, rules_df)
                
                # Store in session state
                st.session_state.rules_df = rules_df
                st.session_state.justice_gaps_df = justice_gaps_df
                st.session_state.analyzer = analyzer
                
                st.success(f"Found {len(rules_df)} association rules!")
        
        # Display rules if available
        if 'rules_df' in st.session_state:
            rules_df = st.session_state.rules_df
            justice_gaps_df = st.session_state.justice_gaps_df
            
            # Filter rules if needed
            if gap_type_filter:
                justice_gaps_df = justice_gaps_df[justice_gaps_df['gap_type'].isin(gap_type_filter)]
                rule_ids = justice_gaps_df['rule_id'].tolist()
                rules_df = rules_df.loc[rule_ids]
            
            # Display top rules
            st.subheader(f"Top {min(10, len(rules_df))} Association Rules")
            
            for idx, rule in rules_df.head(10).iterrows():
                with st.expander(f"Rule {idx+1}: Lift = {rule['lift']:.2f}, Confidence = {rule['confidence']:.2%}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**IF (Antecedents):**")
                        antecedents = list(rule['antecedents'])
                        for ant in antecedents:
                            st.write(f"• {ant}")
                    
                    with col2:
                        st.markdown("**THEN (Consequents):**")
                        consequents = list(rule['consequents'])
                        for con in consequents:
                            st.write(f"• {con}")
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Support", f"{rule['support']:.3f}")
                    col2.metric("Confidence", f"{rule['confidence']:.2%}")
                    col3.metric("Lift", f"{rule['lift']:.2f}")
                    
                    # Find corresponding justice gap analysis
                    if idx in justice_gaps_df['rule_id'].values:
                        gap_info = justice_gaps_df[justice_gaps_df['rule_id'] == idx].iloc[0]
                        st.info(f"**Justice Gap Type:** {gap_info['gap_type']}")
                        st.info(f"**Policy Implications:** {gap_info['policy_implications']}")
    
    with tab3:
        st.subheader("Justice Gap Analysis")
        
        if 'justice_gaps_df' in st.session_state:
            justice_gaps_df = st.session_state.justice_gaps_df
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_gaps = len(justice_gaps_df)
                st.metric("Total Justice Gaps", total_gaps)
            
            with col2:
                avg_lift = justice_gaps_df['lift'].mean()
                st.metric("Average Lift", f"{avg_lift:.2f}")
            
            with col3:
                dem_gaps = len(justice_gaps_df[justice_gaps_df['gap_type'] == 'Demographic Disparity'])
                st.metric("Demographic Gaps", dem_gaps)
            
            with col4:
                reg_gaps = len(justice_gaps_df[justice_gaps_df['gap_type'] == 'Regional Disparity'])
                st.metric("Regional Gaps", reg_gaps)
            
            # Gap type distribution
            gap_dist = justice_gaps_df['gap_type'].value_counts()
            fig = px.bar(x=gap_dist.index, y=gap_dist.values,
                        title="Justice Gap Type Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.subheader("Detailed Justice Gap Analysis")
            display_cols = ['gap_type', 'antecedents', 'consequents', 'lift', 'policy_implications']
            st.dataframe(justice_gaps_df[display_cols], use_container_width=True)
    
    with tab4:
        st.subheader("Interactive Visualizations")
        
        if 'rules_df' in st.session_state:
            rules_df = st.session_state.rules_df
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Scatter plot: Support vs Confidence
                fig1 = px.scatter(rules_df, x='support', y='confidence',
                                 size='lift', color='lift',
                                 hover_data=['antecedents', 'consequents'],
                                 title="Support vs Confidence (Size = Lift)")
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Lift distribution
                fig2 = px.histogram(rules_df, x='lift', 
                                   title="Lift Distribution",
                                   nbins=20)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Heatmap of delay rates
            st.subheader("Delay Rate Heatmap")
            
            # Create cross-tabulation
            heatmap_data = pd.crosstab(df['Case_Type'], df['Gender'], 
                                       values=df['Delay_Flag'], 
                                       aggfunc='mean')
            
            fig3 = px.imshow(heatmap_data, 
                            labels=dict(x="Gender", y="Case Type", color="Delay Rate"),
                            title="Delay Rates by Case Type and Gender",
                            aspect="auto")
            fig3.update_xaxes(side="top")
            st.plotly_chart(fig3, use_container_width=True)
            
            # Network graph
            st.subheader("Association Rule Network")
            
            # Create simple network visualization
            if len(rules_df) > 0:
                G = nx.Graph()
                top_rules = rules_df.nlargest(15, 'lift')
                
                for idx, rule in top_rules.iterrows():
                    antecedents = list(rule['antecedents'])
                    consequents = list(rule['consequents'])
                    
                    # Add nodes and edges
                    for ant in antecedents:
                        if 'OUTCOME' not in str(ant):  # Don't add outcome as separate node for clarity
                            G.add_node(str(ant)[:30], type='antecedent', size=10)
                    
                    for con in consequents:
                        if 'OUTCOME' in str(con):
                            G.add_node('OUTCOME', type='consequent', size=15)
                            for ant in antecedents:
                                if 'OUTCOME' not in str(ant):
                                    G.add_edge(str(ant)[:30], 'OUTCOME', weight=rule['lift'])
                
                # Create plotly network
                pos = nx.spring_layout(G)
                
                edge_x = []
                edge_y = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=1, color='#888'),
                    hoverinfo='none',
                    mode='lines')
                
                node_x = []
                node_y = []
                node_text = []
                node_size = []
                node_color = []
                
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
                    node_size.append(G.nodes[node].get('size', 10) * 10)
                    node_color.append('red' if node == 'OUTCOME' else 'blue')
                
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=node_text,
                    textposition="top center",
                    hoverinfo='text',
                    marker=dict(
                        size=node_size,
                        color=node_color,
                        line=dict(width=2, color='DarkSlateGrey')
                    )
                )
                
                fig4 = go.Figure(data=[edge_trace, node_trace],
                                layout=go.Layout(
                                    title='Association Rule Network',
                                    showlegend=False,
                                    hovermode='closest',
                                    margin=dict(b=0, l=0, r=0, t=40),
                                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                                ))
                
                st.plotly_chart(fig4, use_container_width=True)
    
    with tab5:
        st.subheader("Policy Insights and Recommendations")
        
        if 'justice_gaps_df' in st.session_state:
            justice_gaps_df = st.session_state.justice_gaps_df
            
            # Generate insights
            insights = []
            
            # Demographic disparities
            dem_gaps = justice_gaps_df[justice_gaps_df['gap_type'] == 'Demographic Disparity']
            if len(dem_gaps) > 0:
                top_dem_gap = dem_gaps.nlargest(1, 'lift').iloc[0]
                insights.append({
                    'type': '⚠️ Demographic Disparity',
                    'finding': f"Strong association found: {top_dem_gap['antecedents']} → {top_dem_gap['consequents']}",
                    'recommendation': "Implement targeted interventions for affected demographic groups"
                })
            
            # Regional disparities
            reg_gaps = justice_gaps_df[justice_gaps_df['gap_type'] == 'Regional Disparity']
            if len(reg_gaps) > 0:
                top_reg_gap = reg_gaps.nlargest(1, 'lift').iloc[0]
                insights.append({
                    'type': '📍 Regional Disparity',
                    'finding': f"Geographic inequity detected: {top_reg_gap['antecedents']}",
                    'recommendation': "Allocate additional resources to underserved regions"
                })
            
            # Case-type biases
            case_gaps = justice_gaps_df[justice_gaps_df['gap_type'] == 'Case-Type Bias']
            if len(case_gaps) > 0:
                top_case_gap = case_gaps.nlargest(1, 'lift').iloc[0]
                insights.append({
                    'type': '📋 Case-Type Bias',
                    'finding': f"Specific case types show disparities: {top_case_gap['antecedents']}",
                    'recommendation': "Review procedures for identified case types"
                })
            
            # Display insights
            for insight in insights:
                with st.container():
                    st.markdown(f"### {insight['type']}")
                    st.markdown(f"**Finding:** {insight['finding']}")
                    st.markdown(f"**Recommendation:** {insight['recommendation']}")
                    st.markdown("---")
            
            # Overall recommendations
            st.subheader("Overall Recommendations")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                #### 🎯 Short-term Actions
                - Implement monitoring for top 3 justice gaps
                - Train staff on identified biases
                - Establish rapid response team
                """)
            
            with col2:
                st.markdown("""
                #### 📈 Medium-term Initiatives
                - Develop targeted intervention programs
                - Enhance data collection systems
                - Regular justice gap audits
                """)
            
            with col3:
                st.markdown("""
                #### 🏛️ Long-term Strategies
                - Policy reform based on evidence
                - Systemic procedural changes
                - Capacity building across regions
                """)
            
            # Export option
            st.subheader("Export Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Download Justice Gap Report"):
                    # Create report
                    report = f"""
                    SYSTEMIC JUSTICE GAP ANALYSIS REPORT
                    =====================================
                    
                    Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}
                    Total Cases Analyzed: {len(df):,}
                    Association Rules Found: {len(rules_df)}
                    Justice Gaps Identified: {len(justice_gaps_df)}
                    
                    TOP 5 JUSTICE GAPS:
                    --------------------
                    """
                    
                    for i, gap in justice_gaps_df.head(5).iterrows():
                        report += f"\n{i+1}. {gap['gap_type']}\n"
                        report += f"   Rule: IF {gap['antecedents']} THEN {gap['consequents']}\n"
                        report += f"   Lift: {gap['lift']:.2f}\n"
                        report += f"   Policy Implications: {gap['policy_implications']}\n"
                    
                    # Create download button
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name="justice_gap_report.txt",
                        mime="text/plain"
                    )
            
            with col2:
                if st.button("Download All Data"):
                    # Create CSV with all data
                    all_data = pd.concat([
                        rules_df,
                        justice_gaps_df.add_prefix('gap_')
                    ], axis=1)
                    
                    csv = all_data.to_csv(index=False)
                    
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="justice_gap_analysis_full.csv",
                        mime="text/csv"
                    )

else:
    # Welcome screen
    st.markdown("""
    ## Welcome to the Justice Gap Analyzer
    
    This application uses **FP-Growth Association Rule Mining** with **Lift-Based Filtering** to identify systemic justice gaps in public legal aid systems.
    
    ### How to use:
    1. **Upload your legal aid dataset** (CSV or Excel format) using the sidebar
    2. **Adjust the algorithm parameters** (support, confidence, lift thresholds)
    3. **Run the analysis** to discover association rules
    4. **Explore** the results through interactive visualizations
    
    ### Sample dataset structure should include:
    - Demographic features (Gender, Age, Income, Vulnerability status)
    - Case characteristics (Case Type, Jurisdiction)
    - Procedural features (Hearing count, Adjournments)
    - Outcome variables (Delay flag, Case outcome)
    
    ### Quick Start:
    Click **"Load Sample Data"** in the sidebar to try with example data.
    """)
    
    # Add feature diagram
    st.image("https://raw.githubusercontent.com/plotly/datasets/master/sample_image.png", 
             caption="Justice Gap Analysis Workflow", use_column_width=True)
