import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="Dashboard", layout="wide", initial_sidebar_state="expanded")

# Title and description
st.title("📊 Random Database Dashboard")
st.markdown("---")

# Generate random database
@st.cache_data
def generate_database():
    """Generate a random database with sales data"""
    np.random.seed(42)
    
    dates = [datetime.now() - timedelta(days=x) for x in range(90)]
    products = ["Product A", "Product B", "Product C", "Product D", "Product E"]
    regions = ["North", "South", "East", "West"]
    
    data = {
        "Date": np.repeat(dates, 20),
        "Product": np.tile(np.repeat(products, 4), 90),
        "Region": np.tile(regions, 450),
        "Sales": np.random.randint(100, 5000, 1800),
        "Quantity": np.random.randint(1, 100, 1800),
        "Customer_ID": np.random.randint(1000, 9999, 1800)
    }
    
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Revenue"] = df["Sales"] * df["Quantity"]
    return df

# Load data
df = generate_database()

# Sidebar filters
st.sidebar.header("🔧 Filters")
selected_product = st.sidebar.multiselect("Select Products", options=df["Product"].unique(), default=df["Product"].unique())
selected_region = st.sidebar.multiselect("Select Regions", options=df["Region"].unique(), default=df["Region"].unique())

# Apply filters
filtered_df = df[(df["Product"].isin(selected_product)) & (df["Region"].isin(selected_region))]

# Key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Revenue", f"${filtered_df['Revenue'].sum():,.0f}")

with col2:
    st.metric("Total Sales", f"${filtered_df['Sales'].sum():,.0f}")

with col3:
    st.metric("Total Quantity", f"{filtered_df['Quantity'].sum():,.0f}")

with col4:
    st.metric("Unique Customers", f"{filtered_df['Customer_ID'].nunique()}")

st.markdown("---")

# Row 1: Charts
col1, col2 = st.columns(2)

with col1:
    # Sales by Product
    sales_by_product = filtered_df.groupby("Product")["Sales"].sum().sort_values(ascending=False)
    fig1 = px.bar(
        x=sales_by_product.index,
        y=sales_by_product.values,
        labels={"x": "Product", "y": "Sales"},
        title="📦 Sales by Product",
        color=sales_by_product.values,
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Revenue by Region
    revenue_by_region = filtered_df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)
    fig2 = px.pie(
        values=revenue_by_region.values,
        names=revenue_by_region.index,
        title="🗺️ Revenue by Region"
    )
    st.plotly_chart(fig2, use_container_width=True)

# Row 2: Time series
st.subheader("📈 Sales Trend Over Time")
daily_sales = filtered_df.groupby("Date").agg({"Sales": "sum", "Quantity": "sum", "Revenue": "sum"}).reset_index()
fig3 = px.line(
    daily_sales,
    x="Date",
    y=["Sales", "Revenue"],
    title="Sales and Revenue Trend",
    markers=True,
    labels={"value": "Amount ($)", "variable": "Metric"}
)
st.plotly_chart(fig3, use_container_width=True)

# Row 3: Data table
st.subheader("📋 Data Sample")
st.dataframe(
    filtered_df.head(10),
    use_container_width=True,
    hide_index=True
)

# Statistics
st.subheader("📊 Statistics")
col1, col2 = st.columns(2)

with col1:
    st.write("**Sales Statistics**")
    st.write(filtered_df["Sales"].describe())

with col2:
    st.write("**Quantity Statistics**")
    st.write(filtered_df["Quantity"].describe())

st.markdown("---")
st.caption("Dashboard generated with Streamlit | Data refreshes every session")

