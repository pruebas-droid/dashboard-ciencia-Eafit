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

def plot_quantitative_column(df, col_name):
    """Create visualizations for quantitative data"""
    fig = px.histogram(
        df,
        x=col_name,
        nbins=30,
        title=f"📊 Distribution of {col_name}",
        labels={col_name: col_name},
        color_discrete_sequence=["#1f77b4"]
    )
    fig.update_layout(showlegend=False)
    return fig

def plot_qualitative_column(df, col_name):
    """Create visualizations for qualitative data"""
    value_counts = df[col_name].value_counts().head(15)
    fig = px.bar(
        x=value_counts.index,
        y=value_counts.values,
        title=f"📈 Frequency of {col_name}",
        labels={"x": col_name, "y": "Count"},
        color=value_counts.values,
        color_continuous_scale="Viridis"
    )
    fig.update_layout(showlegend=False)
    return fig

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
        
        # Display basic info
        st.subheader("📋 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", f"{df.shape[1]}")
        with col3:
            quantitative_cols = sum([is_quantitative(df[col]) for col in df.columns])
            st.metric("Quantitative", quantitative_cols)
        with col4:
            qualitative_cols = df.shape[1] - quantitative_cols
            st.metric("Qualitative", qualitative_cols)
        
        st.markdown("---")
        
        # Sidebar filters
        st.sidebar.header("🔧 Filters")
        
        # Get qualitative columns for filtering
        qualitative_columns = [col for col in df.columns if not is_quantitative(df[col])]
        
        filters = {}
        for col in qualitative_columns[:5]:  # Limit to 5 filters
            unique_values = df[col].unique()
            if len(unique_values) <= 50:  # Only show filter if reasonable number of values
                selected = st.sidebar.multiselect(
                    f"Filter by {col}",
                    options=unique_values,
                    default=unique_values
                )
                if selected:
                    filters[col] = selected
        
        # Apply filters
        filtered_df = df.copy()
        for col, values in filters.items():
            filtered_df = filtered_df[filtered_df[col].isin(values)]
        
        # Display data sample
        st.subheader("📊 Data Sample")
        st.dataframe(filtered_df.head(10), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Separate columns by type
        quantitative_cols = [col for col in filtered_df.columns if is_quantitative(filtered_df[col])]
        qualitative_cols = [col for col in filtered_df.columns if not is_quantitative(filtered_df[col])]
        
        # Quantitative Data Visualizations
        if quantitative_cols:
            st.subheader("📈 Quantitative Data Analysis")
            
            # Display histograms for quantitative columns
            for i, col in enumerate(quantitative_cols[:4]):  # Show first 4
                if i % 2 == 0:
                    cols = st.columns(2)
                try:
                    fig = plot_quantitative_column(filtered_df, col)
                    cols[i % 2].plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    cols[i % 2].warning(f"Could not plot {col}: {e}")
            
            st.markdown("---")
            
            # Correlation heatmap for multiple quantitative columns
            if len(quantitative_cols) > 1:
                st.subheader("🔗 Correlation Matrix")
                corr_matrix = filtered_df[quantitative_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlation"),
                    x=quantitative_cols,
                    y=quantitative_cols,
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    title="Correlation Between Numeric Columns"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")
            
            # Statistics
            st.subheader("📊 Statistics")
            st.dataframe(filtered_df[quantitative_cols].describe(), use_container_width=True)
            st.markdown("---")
        
        # Qualitative Data Visualizations
        if qualitative_cols:
            st.subheader("🏷️ Qualitative Data Analysis")
            
            # Display bar charts for qualitative columns
            for i, col in enumerate(qualitative_cols[:4]):  # Show first 4
                if i % 2 == 0:
                    cols = st.columns(2)
                try:
                    fig = plot_qualitative_column(filtered_df, col)
                    cols[i % 2].plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    cols[i % 2].warning(f"Could not plot {col}: {e}")
            
            st.markdown("---")
        
        # Combined analysis if both types exist
        if quantitative_cols and qualitative_cols:
            st.subheader("🔀 Combined Analysis")
            
            selected_quant = st.selectbox("Select numeric column", quantitative_cols)
            selected_qual = st.selectbox("Select categorical column", qualitative_cols)
            
            if selected_quant and selected_qual:
                fig = px.box(
                    filtered_df,
                    x=selected_qual,
                    y=selected_quant,
                    title=f"{selected_quant} by {selected_qual}",
                    color=selected_qual
                )
                st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👆 Upload a CSV, Excel, or Parquet file to get started!")
