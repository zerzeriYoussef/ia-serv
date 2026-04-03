import json
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.analysis.kpi_executor import KPIExecutor

import numpy as np

# Create dummy dataset based on the payload features
np.random.seed(42)
n_rows = 100
df = pd.DataFrame({
    'price_per_cup': np.random.uniform(2.0, 6.0, n_rows),
    'monthly_spend': np.random.uniform(20, 150, n_rows),
    'continent': np.random.choice(['North America', 'Europe', 'Asia', 'South America'], n_rows),
    'age': np.random.randint(18, 65, n_rows),
    'satisfaction_level': np.random.randint(1, 6, n_rows),
    'income_level': np.random.choice(['Low', 'Medium', 'High'], n_rows),
    'drink_category': np.random.choice(['Coffee', 'Tea', 'Matcha', 'Espresso'], n_rows),
    'drink_preference': np.random.choice(['Strong', 'Mild', 'Sweet'], n_rows),
    'gender': np.random.choice(['M', 'F', 'Other'], n_rows),
    'heart_rate': np.random.uniform(60, 100, n_rows),
    'favorite_drink': np.random.choice(['Latte', 'Green Tea', 'Americano'], n_rows)
})

# Provide the payload
dashboard_config = {
  "id": 2,
  "dataset_id": 37,
  "analysis_id": 73,
  "executive_summary_kpis": [
    {
      "kpi_name": "Total Revenue Potential from Price per Cup",
      "target_column": "price_per_cup",
      "pandas_function": ".sum()",
      "logic_hint": "df['price_per_cup'].sum()",
      "description": "Represents the aggregated value of price per cup across all transactions, indicating overall revenue potential.",
      "icon": "payments"
    },
    {
      "kpi_name": "Average Price per Cup",
      "target_column": "price_per_cup",
      "pandas_function": ".mean()",
      "logic_hint": "df['price_per_cup'].mean()",
      "description": "Provides insight into the typical price customers pay per cup, useful for pricing strategies and value assessment.",
      "icon": "attach_money"
    },
    {
      "kpi_name": "Total Customer Monthly Spend",
      "target_column": "monthly_spend",
      "pandas_function": ".sum()",
      "logic_hint": "df['monthly_spend'].sum()",
      "description": "Aggregated monthly spending by customers, highlighting total market engagement and customer lifetime value potential.",
      "icon": "shopping_cart"
    },
    {
      "kpi_name": "Leading Continent by Price per Cup",
      "target_column": "price_per_cup",
      "pandas_function": "groupby().sum().sort_values().head(1)",
      "logic_hint": "df.groupby('continent')['price_per_cup'].sum().sort_values(ascending=False).head(1)",
      "description": "Identifies the continent with the highest total price per cup, indicating key regional markets for strategic focus.",
      "icon": "public"
    },
    {
      "kpi_name": "Average Customer Age",
      "target_column": "age",
      "pandas_function": ".mean()",
      "logic_hint": "df['age'].mean()",
      "description": "The average age of our customer base, crucial for understanding demographic trends and tailoring marketing efforts.",
      "icon": "groups"
    },
    {
      "kpi_name": "Average Customer Satisfaction Level",
      "target_column": "satisfaction_level",
      "pandas_function": ".mean()",
      "logic_hint": "df['satisfaction_level'].mean()",
      "description": "Overall customer sentiment, a key indicator of product and service quality and potential for loyalty.",
      "icon": "sentiment_satisfied_alt"
    }
  ],
  "dashboard_charts": [
    {
      "rank": 1,
      "title": "Price per Cup by Continent",
      "relationship": "price_per_cup <-> continent",
      "strength_score": 0.9,
      "x_axis_column": "continent",
      "y_axis_column": "price_per_cup",
      "pandas_grouping": "df.groupby('continent')['price_per_cup'].sum().sort_values(ascending=False)",
      "chart_type": "Bar",
      "business_insight": "Highlights which continents contribute most to total price per cup, guiding regional strategy and resource allocation."
    },
    {
      "rank": 2,
      "title": "Monthly Spend by Income Level",
      "relationship": "monthly_spend <-> income_level",
      "strength_score": 0.7,
      "x_axis_column": "income_level",
      "y_axis_column": "monthly_spend",
      "pandas_grouping": "df.groupby('income_level')['monthly_spend'].sum().sort_values(ascending=False)",
      "chart_type": "Bar",
      "business_insight": "Reveals how customer monthly spending varies across different income levels, informing segmentation and product offerings."
    },
    {
      "rank": 3,
      "title": "Price per Cup vs. Monthly Spend",
      "relationship": "price_per_cup <-> monthly_spend",
      "strength_score": 0.6,
      "x_axis_column": "monthly_spend",
      "y_axis_column": "price_per_cup",
      "pandas_grouping": "df.plot.scatter(x='monthly_spend', y='price_per_cup')",
      "chart_type": "Scatter",
      "business_insight": "Explores the correlation between the price customers pay per cup and their overall monthly spending, identifying potential customer value segments."
    },
    {
      "rank": 4,
      "title": "Drink Preference by Drink Category",
      "relationship": "drink_preference <-> drink_category",
      "strength_score": 1,
      "x_axis_column": "drink_category",
      "y_axis_column": "drink_preference",
      "pandas_grouping": "pd.crosstab(df['drink_preference'], df['drink_category'], normalize='index')",
      "chart_type": "Heatmap",
      "business_insight": "Illustrates strong associations between customer drink preferences and broader drink categories, aiding product development and marketing."
    },
    {
      "rank": 5,
      "title": "Average Price per Cup by Gender",
      "relationship": "price_per_cup <-> gender",
      "strength_score": 0.5,
      "x_axis_column": "gender",
      "y_axis_column": "price_per_cup",
      "pandas_grouping": "df.groupby('gender')['price_per_cup'].mean().sort_values(ascending=False)",
      "chart_type": "Bar",
      "business_insight": "Compares the average price per cup across different genders, providing insights for targeted pricing or product positioning."
    },
    {
      "rank": 6,
      "title": "Age vs. Heart Rate Correlation",
      "relationship": "age <-> heart_rate",
      "strength_score": 0.4,
      "x_axis_column": "age",
      "y_axis_column": "heart_rate",
      "pandas_grouping": "df.plot.scatter(x='age', y='heart_rate')",
      "chart_type": "Scatter",
      "business_insight": "Examines the relationship between customer age and heart rate, potentially useful for health-related product recommendations or wellness programs."
    },
    {
      "rank": 7,
      "title": "Favorite Drink by Drink Preference",
      "relationship": "favorite_drink <-> drink_preference",
      "strength_score": 0.733,
      "x_axis_column": "drink_preference",
      "y_axis_column": "favorite_drink",
      "pandas_grouping": "pd.crosstab(df['favorite_drink'], df['drink_preference'], normalize='index')",
      "chart_type": "Heatmap",
      "business_insight": "Shows how specific favorite drinks align with broader drink preferences, helping to understand customer choices and cross-selling opportunities."
    },
    {
      "rank": 8,
      "title": "Monthly Spend by Gender",
      "relationship": "monthly_spend <-> gender",
      "strength_score": 0.5,
      "x_axis_column": "gender",
      "y_axis_column": "monthly_spend",
      "pandas_grouping": "df.groupby('gender')['monthly_spend'].sum().sort_values(ascending=False)",
      "chart_type": "Bar",
      "business_insight": "Analyzes total monthly spending patterns across genders, informing marketing campaigns and product targeting."
    }
  ],
  "rag_meta": {}
}

executor = KPIExecutor(df)

kpis = executor.execute_all_kpis(dashboard_config["executive_summary_kpis"])
print("=== KPIs ===")
print(json.dumps(kpis, indent=2))

charts = executor.execute_all_charts(dashboard_config["dashboard_charts"])
print("\n=== Charts ===")
print(json.dumps(charts, indent=2))
