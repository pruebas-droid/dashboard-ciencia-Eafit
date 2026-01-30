import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Set page config
st.set_page_config(page_title="Smart Data Dashboard", layout="wide", initial_sidebar_state="expanded")

# Title
st.title("📊 Smart Data Dashboard")
st.markdown("Upload CSV, Excel, or other data files for automatic analysis and visualization")
st.markdown("---")

def is_quantitative(series):
    """Determine if a series is quantitative (numerical) or qualitative (categorical)"""
    if pd.api.types.is_numeric_dtype(series):
        return True
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    return False

def is_categorical(series):
    """Determine if a series is categorical"""
    return not is_quantitative(series)

def clean_data(df):
    """Clean data: handle NA, spaces, duplicates, etc."""
    df_clean = df.copy()
    
    # Replace spaces with NaN for string columns
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].replace(r'^\s*$', np.nan, regex=True)
            df_clean[col] = df_clean[col].str.strip() if df_clean[col].dtype == 'object' else df_clean[col]
    
    return df_clean

def load_data(file):
    """Load data from uploaded file"""
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file)
        elif file.name.endswith('.parquet'):
            return pd.read_parquet(file)
        else:
            st.error("Unsupported file format. Please use CSV, Excel, or Parquet.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Sidebar - File Upload
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls', 'parquet'])

if uploaded_file is not None:
    # Load data
    df = load_data(uploaded_file)
    
    if df is not None:
        st.sidebar.success("✅ File loaded successfully!")
        
        # ============================================
        # SECTION 1: DATA CLEANING
        # ============================================
        st.header("🧹 Section 1: Data Cleaning")
        
        # Show original data info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Data")
            st.metric("Rows", f"{df.shape[0]:,}")
            st.metric("Columns", f"{df.shape[1]}")
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
            st.metric("Duplicates", f"{df.duplicated().sum():,}")
        
        # Clean data
        df_clean = clean_data(df)
        
        # Show cleaning options
        st.subheader("🛠️ Cleaning Options")
        
        col1, col2, col3 = st.columns(3)
        
        remove_duplicates = col1.checkbox("Remove Duplicates", value=True)
        remove_na = col2.checkbox("Remove Rows with NA", value=False)
        fill_na_strategy = col3.selectbox("Fill NA Strategy", ["Keep", "Drop Column", "Forward Fill", "Backward Fill", "Mean (numeric only)"])
        
        # Apply selected cleaning
        if remove_duplicates:
            df_clean = df_clean.drop_duplicates()
        
        if fill_na_strategy == "Drop Column":
            df_clean = df_clean.dropna(axis=1, how='all')
        elif fill_na_strategy == "Forward Fill":
            df_clean = df_clean.fillna(method='ffill')
        elif fill_na_strategy == "Backward Fill":
            df_clean = df_clean.fillna(method='bfill')
        elif fill_na_strategy == "Mean (numeric only)":
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        
        if remove_na:
            df_clean = df_clean.dropna()
        
        # Show cleaned data info
        with col2:
            st.subheader("Cleaned Data")
            st.metric("Rows", f"{df_clean.shape[0]:,}")
            st.metric("Columns", f"{df_clean.shape[1]}")
            st.metric("Missing Values", f"{df_clean.isnull().sum().sum():,}")
            st.metric("Duplicates", f"{df_clean.duplicated().sum():,}")
        
        # Show data quality report
        st.subheader("📊 Data Quality Report")
        quality_data = []
        for col in df_clean.columns:
            quality_data.append({
                "Column": col,
                "Type": str(df_clean[col].dtype),
                "Non-Null %": f"{(1 - df_clean[col].isnull().sum() / len(df_clean)) * 100:.1f}%",
                "Unique Values": df_clean[col].nunique()
            })
        
        st.dataframe(pd.DataFrame(quality_data), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Display cleaned data sample
        st.subheader("📋 Cleaned Data Sample")
        st.dataframe(df_clean.head(10), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # ============================================
        # SECTION 2: EXPLORATORY DATA ANALYSIS (EDA)
        # ============================================
        st.header("📈 Section 2: Exploratory Data Analysis")
        
        # Separate columns by type
        quantitative_cols = [col for col in df_clean.columns if is_quantitative(df_clean[col])]
        qualitative_cols = [col for col in df_clean.columns if is_categorical(df_clean[col])]
        
        st.subheader("📊 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{df_clean.shape[0]:,}")
        with col2:
            st.metric("Total Columns", f"{df_clean.shape[1]}")
        with col3:
            st.metric("Numeric Columns", len(quantitative_cols))
        with col4:
            st.metric("Categorical Columns", len(qualitative_cols))
        
        st.markdown("---")
        
        # Sidebar filters for EDA
        st.sidebar.header("🔍 EDA Filters")
        
        filters = {}
        for col in qualitative_cols[:5]:
            unique_values = df_clean[col].unique()
            if len(unique_values) <= 50:
                selected = st.sidebar.multiselect(
                    f"Filter by {col}",
                    options=sorted([str(v) for v in unique_values]),
                    default=[str(v) for v in unique_values]
                )
                if selected:
                    filters[col] = selected
        
        # Apply filters
        filtered_df = df_clean.copy()
        for col, values in filters.items():
            filtered_df = filtered_df[filtered_df[col].isin(values)]
        
        # Update column lists after filtering
        quantitative_cols = [col for col in filtered_df.columns if is_quantitative(filtered_df[col])]
        qualitative_cols = [col for col in filtered_df.columns if is_categorical(filtered_df[col])]
        
        # ========== QUANTITATIVE ANALYSIS ==========
        if quantitative_cols:
            st.subheader("📊 Quantitative Variables Analysis")
            
            quant_col = st.selectbox("Select numeric column to analyze", quantitative_cols, key="quant_select")
            
            viz_types = st.multiselect(
                "Select visualizations",
                ["Histogram", "Box Plot", "Violin Plot", "Statistics"],
                default=["Histogram"],
                key="quant_viz"
            )
            
            col1, col2 = st.columns(2)
            
            if "Histogram" in viz_types:
                try:
                    fig = px.histogram(
                        filtered_df,
                        x=quant_col,
                        nbins=30,
                        title=f"Distribution of {quant_col}",
                        marginal="box",
                        color_discrete_sequence=["#1f77b4"]
                    )
                    col1.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    col1.error(f"Error creating histogram: {e}")
            
            if "Box Plot" in viz_types:
                try:
                    fig = px.box(filtered_df, y=quant_col, title=f"Box Plot of {quant_col}")
                    col2.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    col2.error(f"Error creating box plot: {e}")
            
            if "Violin Plot" in viz_types:
                try:
                    fig = px.violin(filtered_df, y=quant_col, title=f"Violin Plot of {quant_col}", box=True, points="outliers")
                    col1.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    col1.error(f"Error creating violin plot: {e}")
            
            if "Statistics" in viz_types:
                col2.subheader("📊 Statistical Summary")
                col2.dataframe(filtered_df[quant_col].describe(), use_container_width=True)
            
            st.markdown("---")
            
            # Correlation analysis
            if len(quantitative_cols) > 1:
                st.subheader("🔗 Correlation Analysis")
                
                corr_viz = st.selectbox(
                    "Select correlation visualization",
                    ["Correlation Heatmap", "Correlation Matrix", "Pair Plot (first 5 columns)"],
                    key="corr_viz"
                )
                
                if corr_viz == "Correlation Heatmap":
                    corr_matrix = filtered_df[quantitative_cols].corr()
                    fig = px.imshow(
                        corr_matrix,
                        labels=dict(color="Correlation"),
                        x=quantitative_cols,
                        y=quantitative_cols,
                        color_continuous_scale="RdBu_r",
                        zmin=-1,
                        zmax=1,
                        title="Correlation Heatmap"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif corr_viz == "Correlation Matrix":
                    st.dataframe(filtered_df[quantitative_cols].corr(), use_container_width=True)
                
                st.markdown("---")
        
        # ========== QUALITATIVE ANALYSIS ==========
        if qualitative_cols:
            st.subheader("🏷️ Categorical Variables Analysis")
            
            qual_col = st.selectbox("Select categorical column to analyze", qualitative_cols, key="qual_select")
            
            viz_types = st.multiselect(
                "Select visualizations",
                ["Bar Chart", "Pie Chart", "Value Counts"],
                default=["Bar Chart"],
                key="qual_viz"
            )
            
            col1, col2 = st.columns(2)
            
            if "Bar Chart" in viz_types:
                try:
                    value_counts = filtered_df[qual_col].value_counts().head(20)
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        labels={"x": qual_col, "y": "Count"},
                        title=f"Frequency of {qual_col}",
                        color=value_counts.values,
                        color_continuous_scale="Viridis"
                    )
                    col1.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    col1.error(f"Error creating bar chart: {e}")
            
            if "Pie Chart" in viz_types:
                try:
                    value_counts = filtered_df[qual_col].value_counts().head(10)
                    fig = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f"Distribution of {qual_col}"
                    )
                    col2.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    col2.error(f"Error creating pie chart: {e}")
            
            if "Value Counts" in viz_types:
                col1.subheader("📊 Value Counts")
                value_counts = filtered_df[qual_col].value_counts()
                col1.dataframe(value_counts, use_container_width=True)
            
            st.markdown("---")
        
        # ========== COMBINED ANALYSIS ==========
        if quantitative_cols and qualitative_cols:
            st.subheader("🔀 Combined Analysis (Numeric vs Categorical)")
            
            selected_quant = st.selectbox("Select numeric column", quantitative_cols, key="combined_quant")
            selected_qual = st.selectbox("Select categorical column", qualitative_cols, key="combined_qual")
            
            combined_viz = st.selectbox(
                "Select visualization type",
                ["Box Plot", "Violin Plot", "Scatter with Color", "Bar (Mean)"],
                key="combined_viz"
            )
            
            try:
                if combined_viz == "Box Plot":
                    fig = px.box(
                        filtered_df,
                        x=selected_qual,
                        y=selected_quant,
                        title=f"{selected_quant} by {selected_qual}",
                        color=selected_qual
                    )
                
                elif combined_viz == "Violin Plot":
                    fig = px.violin(
                        filtered_df,
                        x=selected_qual,
                        y=selected_quant,
                        title=f"{selected_quant} by {selected_qual}",
                        color=selected_qual,
                        box=True
                    )
                
                elif combined_viz == "Scatter with Color":
                    fig = px.scatter(
                        filtered_df,
                        x=filtered_df.index,
                        y=selected_quant,
                        color=selected_qual,
                        title=f"{selected_quant} colored by {selected_qual}",
                        hover_data=[selected_quant, selected_qual]
                    )
                
                elif combined_viz == "Bar (Mean)":
                    mean_data = filtered_df.groupby(selected_qual)[selected_quant].mean().reset_index()
                    fig = px.bar(
                        mean_data,
                        x=selected_qual,
                        y=selected_quant,
                        title=f"Mean {selected_quant} by {selected_qual}",
                        color=selected_qual
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating visualization: {e}")
else:
    st.info("👆 Upload a CSV, Excel, or Parquet file to get started!")
