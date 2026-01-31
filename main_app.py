import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
from groq import Groq
import io

# Page configuration
st.set_page_config(
    page_title="Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [role="tab"] {
        font-size: 1.1em;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("📊 Data Analysis Dashboard with LLM Insights")

# Sidebar - File upload and Groq API key
st.sidebar.header("⚙️ Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset (CSV, Excel, JSON, Parquet)",
    type=["csv", "xlsx", "xls", "json", "parquet"]
)

# Groq API Key
groq_api_key = st.sidebar.text_input(
    "Enter your Groq API Key (optional for LLM insights)",
    type="password",
    help="Get your API key from https://console.groq.com"
)

# Initialize session state for data
if "df" not in st.session_state:
    st.session_state.df = None
if "df_cleaned" not in st.session_state:
    st.session_state.df_cleaned = None

# Load data
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            st.session_state.df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            st.session_state.df = pd.read_json(uploaded_file)
        elif uploaded_file.name.endswith('.parquet'):
            st.session_state.df = pd.read_parquet(uploaded_file)
        
        st.sidebar.success(f"✅ File loaded successfully!")
        st.sidebar.info(f"Shape: {st.session_state.df.shape}")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading file: {str(e)}")

# Main content
if st.session_state.df is not None:
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Data Preview",
        "🧹 Data Cleaning",
        "📈 Quantitative EDA",
        "📊 Qualitative EDA",
        "🤖 LLM Insights"
    ])
    
    # Tab 1: Data Preview
    with tab1:
        st.header("📋 Data Preview & Info")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Info")
            st.write(f"**Shape:** {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns")
            st.write(f"**Memory Usage:** {st.session_state.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            st.subheader("Data Types")
            dtype_df = st.session_state.df.dtypes.astype(str).reset_index()
            dtype_df.columns = ['Column', 'Type']
            st.dataframe(dtype_df, use_container_width=True)
        
        with col2:
            st.subheader("Missing Values")
            missing_df = pd.DataFrame({
                'Column': st.session_state.df.columns,
                'Missing': st.session_state.df.isnull().sum(),
                'Percentage': (st.session_state.df.isnull().sum() / len(st.session_state.df) * 100).round(2)
            })
            missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
            
            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ No missing values found!")
        
        st.subheader("First 10 Rows")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
    
    # Tab 2: Data Cleaning
    with tab2:
        st.header("🧹 Data Cleaning")
        
        # Initialize cleaned dataframe
        if st.session_state.df_cleaned is None:
            st.session_state.df_cleaned = st.session_state.df.copy()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Remove Duplicates"):
                initial_shape = st.session_state.df_cleaned.shape
                st.session_state.df_cleaned = st.session_state.df_cleaned.drop_duplicates()
                st.success(f"Removed {initial_shape[0] - st.session_state.df_cleaned.shape[0]} duplicate rows!")
                st.rerun()
        
        with col2:
            if st.button("✂️ Drop Columns with >50% Missing"):
                cols_to_drop = st.session_state.df_cleaned.columns[
                    st.session_state.df_cleaned.isnull().sum() / len(st.session_state.df_cleaned) > 0.5
                ]
                st.session_state.df_cleaned = st.session_state.df_cleaned.drop(columns=cols_to_drop)
                if len(cols_to_drop) > 0:
                    st.success(f"Dropped {len(cols_to_drop)} columns: {', '.join(cols_to_drop)}")
                    st.rerun()
                else:
                    st.info("No columns with >50% missing values found.")
        
        with col3:
            if st.button("🔄 Fill Numeric Missing (Mean)"):
                numeric_cols = st.session_state.df_cleaned.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if st.session_state.df_cleaned[col].isnull().sum() > 0:
                        st.session_state.df_cleaned[col].fillna(
                            st.session_state.df_cleaned[col].mean(), inplace=True
                        )
                st.success("Filled numeric missing values with mean!")
                st.rerun()
        
        # Manual column selection and operations
        st.subheader("Manual Cleaning Operations")
        
        cleaning_option = st.selectbox(
            "Select cleaning operation:",
            ["None", "Remove Outliers (IQR)", "Remove Columns", "Fill Missing Values"]
        )
        
        if cleaning_option == "Remove Outliers (IQR)":
            numeric_cols = st.session_state.df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                col_to_clean = st.selectbox("Select numeric column:", numeric_cols)
                if st.button("Remove Outliers"):
                    Q1 = st.session_state.df_cleaned[col_to_clean].quantile(0.25)
                    Q3 = st.session_state.df_cleaned[col_to_clean].quantile(0.75)
                    IQR = Q3 - Q1
                    initial_len = len(st.session_state.df_cleaned)
                    st.session_state.df_cleaned = st.session_state.df_cleaned[
                        (st.session_state.df_cleaned[col_to_clean] >= Q1 - 1.5 * IQR) &
                        (st.session_state.df_cleaned[col_to_clean] <= Q3 + 1.5 * IQR)
                    ]
                    st.success(f"Removed {initial_len - len(st.session_state.df_cleaned)} outliers!")
                    st.rerun()
        
        elif cleaning_option == "Remove Columns":
            cols_to_remove = st.multiselect("Select columns to remove:", st.session_state.df_cleaned.columns)
            if st.button("Remove Selected Columns"):
                st.session_state.df_cleaned = st.session_state.df_cleaned.drop(columns=cols_to_remove)
                st.success(f"Removed {len(cols_to_remove)} columns!")
                st.rerun()
        
        elif cleaning_option == "Fill Missing Values":
            cols_with_missing = st.session_state.df_cleaned.columns[
                st.session_state.df_cleaned.isnull().sum() > 0
            ].tolist()
            if cols_with_missing:
                col_to_fill = st.selectbox("Select column to fill:", cols_with_missing)
                fill_method = st.radio("Fill method:", ["Forward Fill", "Backward Fill", "Mean", "Median", "Drop"])
                if st.button("Apply Fill"):
                    if fill_method == "Forward Fill":
                        st.session_state.df_cleaned[col_to_fill].fillna(method='ffill', inplace=True)
                    elif fill_method == "Backward Fill":
                        st.session_state.df_cleaned[col_to_fill].fillna(method='bfill', inplace=True)
                    elif fill_method == "Mean":
                        st.session_state.df_cleaned[col_to_fill].fillna(
                            st.session_state.df_cleaned[col_to_fill].mean(), inplace=True
                        )
                    elif fill_method == "Median":
                        st.session_state.df_cleaned[col_to_fill].fillna(
                            st.session_state.df_cleaned[col_to_fill].median(), inplace=True
                        )
                    elif fill_method == "Drop":
                        st.session_state.df_cleaned = st.session_state.df_cleaned.dropna(subset=[col_to_fill])
                    st.success(f"Filled missing values using {fill_method}!")
                    st.rerun()
        
        st.subheader("Cleaned Data Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Current Shape:** {st.session_state.df_cleaned.shape}")
            st.write(f"**Missing Values:** {st.session_state.df_cleaned.isnull().sum().sum()}")
        with col2:
            st.write(f"**Duplicates:** {st.session_state.df_cleaned.duplicated().sum()}")
        
        st.dataframe(st.session_state.df_cleaned.head(), use_container_width=True)
    
    # Tab 3: Quantitative EDA
    with tab3:
        st.header("📈 Quantitative EDA")
        
        df_analysis = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df
        numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Summary Statistics
            st.subheader("Summary Statistics")
            st.dataframe(df_analysis[numeric_cols].describe().T, use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distribution of Numeric Columns")
                selected_col = st.selectbox("Select column for histogram:", numeric_cols, key="hist")
                fig = px.histogram(df_analysis, x=selected_col, nbins=30, 
                                  title=f"Distribution of {selected_col}",
                                  color_discrete_sequence=['#1f77b4'])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Box Plots")
                fig = go.Figure()
                for col in numeric_cols[:5]:  # Limit to 5 columns for clarity
                    fig.add_trace(go.Box(y=df_analysis[col], name=col))
                fig.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation Matrix
            st.subheader("Correlation Matrix")
            if len(numeric_cols) > 1:
                corr_matrix = df_analysis[numeric_cols].corr()
                fig = px.imshow(corr_matrix, 
                               text_auto=True,
                               color_continuous_scale='RdBu_r',
                               zmin=-1, zmax=1,
                               title="Correlation Matrix")
                fig.update_layout(height=600, width=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # High correlations
                st.subheader("High Correlations (> 0.7 or < -0.7)")
                high_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            high_corr.append({
                                'Variable 1': corr_matrix.columns[i],
                                'Variable 2': corr_matrix.columns[j],
                                'Correlation': round(corr_matrix.iloc[i, j], 3)
                            })
                
                if high_corr:
                    st.dataframe(pd.DataFrame(high_corr), use_container_width=True)
                else:
                    st.info("No high correlations found.")
            
            # Scatter plots for top correlations
            st.subheader("Scatter Plots")
            if len(numeric_cols) > 1:
                col1_scatter = st.selectbox("X-axis:", numeric_cols, key="x_scatter")
                col2_scatter = st.selectbox("Y-axis:", numeric_cols, key="y_scatter")
                fig = px.scatter(df_analysis, x=col1_scatter, y=col2_scatter,
                               title=f"{col1_scatter} vs {col2_scatter}",
                               trendline="ols", trendline_color_override="red")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No numeric columns found in the dataset.")
    
    # Tab 4: Qualitative EDA
    with tab4:
        st.header("📊 Qualitative EDA")
        
        df_analysis = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df
        categorical_cols = df_analysis.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            # Value Counts
            st.subheader("Categorical Distribution")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                selected_cat = st.selectbox("Select categorical column:", categorical_cols)
            
            with col2:
                chart_type = st.radio("Chart type:", ["Bar Chart", "Pie Chart", "Donut Chart"], horizontal=True)
            
            value_counts = df_analysis[selected_cat].value_counts().head(10)
            
            if chart_type == "Bar Chart":
                fig = px.bar(value_counts, 
                           title=f"Distribution of {selected_cat}",
                           labels={'index': selected_cat, 'value': 'Count'},
                           color_discrete_sequence=['#636EFA'])
            elif chart_type == "Pie Chart":
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f"Distribution of {selected_cat}")
            else:  # Donut Chart
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f"Distribution of {selected_cat}",
                           hole=0.3)
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Multiple categorical columns
            st.subheader("Multiple Categories View")
            cols_to_show = st.multiselect("Select columns to display:", categorical_cols, 
                                         default=categorical_cols[:3])
            
            for col in cols_to_show:
                st.write(f"**{col}**")
                val_counts = df_analysis[col].value_counts().head(10)
                fig = px.bar(val_counts, 
                           labels={'index': col, 'value': 'Count'},
                           height=300,
                           color_discrete_sequence=['#EF553B'])
                st.plotly_chart(fig, use_container_width=True)
            
            # Unique values summary
            st.subheader("Unique Values Summary")
            unique_summary = pd.DataFrame({
                'Column': categorical_cols,
                'Unique Values': [df_analysis[col].nunique() for col in categorical_cols],
                'Most Common': [df_analysis[col].mode()[0] if len(df_analysis[col].mode()) > 0 else 'N/A' 
                               for col in categorical_cols]
            })
            st.dataframe(unique_summary, use_container_width=True)
        else:
            st.warning("⚠️ No categorical columns found in the dataset.")
    
    # Tab 5: LLM Insights
    with tab5:
        st.header("🤖 LLM Insights with Groq")
        
        if not groq_api_key:
            st.warning("⚠️ Please provide your Groq API key in the sidebar to use LLM insights.")
            st.info("Get your API key from: https://console.groq.com")
        else:
            try:
                client = Groq(api_key=groq_api_key)
                
                df_analysis = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df
                
                # Prepare data summary for LLM
                data_summary = f"""
                Dataset Overview:
                - Shape: {df_analysis.shape}
                - Columns: {', '.join(df_analysis.columns)}
                - Data Types: {df_analysis.dtypes.to_dict()}
                - Missing Values: {df_analysis.isnull().sum().to_dict()}
                
                Statistical Summary:
                {df_analysis.describe().to_string()}
                
                First few rows:
                {df_analysis.head(10).to_string()}
                """
                
                # Query options
                query_type = st.radio("Select insight type:", 
                                     ["General Insights", "Anomalies & Patterns", 
                                      "Data Quality", "Business Recommendations",
                                      "Custom Query"], 
                                     horizontal=True)
                
                if query_type == "General Insights":
                    prompt = f"""Analyze this dataset and provide key insights about the data distribution, 
                    relationships, and important statistics:\n\n{data_summary}"""
                
                elif query_type == "Anomalies & Patterns":
                    prompt = f"""Identify any anomalies, patterns, outliers, or unusual trends in this dataset:
                    \n\n{data_summary}"""
                
                elif query_type == "Data Quality":
                    prompt = f"""Evaluate the data quality of this dataset. Comment on missing values, 
                    duplicates, inconsistencies, and provide recommendations for improvement:\n\n{data_summary}"""
                
                elif query_type == "Business Recommendations":
                    prompt = f"""Based on this dataset, provide business insights and actionable recommendations:
                    \n\n{data_summary}"""
                
                else:  # Custom Query
                    custom_query = st.text_area("Enter your custom query about the data:", height=100)
                    prompt = f"""{custom_query}\n\nDataset Context:\n{data_summary}"""
                
                # Generate insights button
                if st.button("🔍 Generate Insights", key="generate_insights"):
                    with st.spinner("🤖 Groq is analyzing your data..."):
                        try:
                            response = client.messages.create(
                                model="mixtral-8x7b-32768",
                                max_tokens=2000,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": prompt
                                    }
                                ]
                            )
                            
                            insights = response.content[0].text
                            
                            st.subheader("💡 Insights & Analysis")
                            st.markdown(insights)
                            
                            # Export insights
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("📋 Copy to Clipboard"):
                                    st.info("Insights copied! (Use system copy function)")
                            with col2:
                                if st.button("📄 Download as Text"):
                                    st.download_button(
                                        label="Download Insights",
                                        data=insights,
                                        file_name="data_insights.txt",
                                        mime="text/plain"
                                    )
                        
                        except Exception as e:
                            st.error(f"❌ Error generating insights: {str(e)}")
                
                # Suggestions section
                st.divider()
                st.subheader("📊 Quick Analysis Suggestions")
                
                numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df_analysis.select_dtypes(include=['object']).columns.tolist()
                
                if numeric_cols:
                    st.markdown(f"**Numeric Columns:** {', '.join(numeric_cols)}")
                if categorical_cols:
                    st.markdown(f"**Categorical Columns:** {', '.join(categorical_cols)}")
                
                st.markdown("💡 Try asking about correlations, distributions, or specific column relationships!")
            
            except Exception as e:
                st.error(f"❌ Groq API Error: {str(e)}")
                st.info("Make sure your API key is valid and you have sufficient credits.")
else:
    # Welcome message when no file is uploaded
    st.info("👈 Please upload a dataset from the sidebar to get started!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📋 Data Preview")
        st.markdown("View your dataset structure, info, and first rows")
    with col2:
        st.markdown("### 🧹 Data Cleaning")
        st.markdown("Clean your data with automated and manual operations")
    with col3:
        st.markdown("### 📈 Quantitative EDA")
        st.markdown("Statistical analysis and numeric visualizations")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 Qualitative EDA")
        st.markdown("Categorical analysis and distribution charts")
    with col2:
        st.markdown("### 🤖 LLM Insights")
        st.markdown("AI-powered insights from your data using Groq")
    with col3:
        st.markdown("### 🚀 Getting Started")
        st.markdown("1. Upload a file\n2. Clean if needed\n3. Explore & analyze\n4. Get AI insights")
