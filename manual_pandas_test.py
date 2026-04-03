import pandas as pd
import json

df = pd.read_csv('c:/stage/auth/IA-service/uploads/e84e6a85-576a-42cb-a155-71537fc8f164.csv')

print("====== KPIs ======")

print("\n1. Total Revenue Potential from Price per Cup")
print("HINT: df['price_per_cup'].sum()")
print("RES:", df['price_per_cup'].sum())

print("\n2. Average Price per Cup")
print("HINT: df['price_per_cup'].mean()")
print("RES:", df['price_per_cup'].mean())

print("\n3. Total Customer Monthly Spend")
print("HINT: df['monthly_spend'].sum()")
print("RES:", df['monthly_spend'].sum())

print("\n4. Leading Continent by Price per Cup")
print("HINT: df.groupby('continent')['price_per_cup'].sum().sort_values(ascending=False).head(1)")
print("RES:\n", df.groupby('continent')['price_per_cup'].sum().sort_values(ascending=False).head(1))

print("\n5. Average Customer Age")
print("HINT: df['age'].mean()")
print("RES:", df['age'].mean())

print("\n6. Average Customer Satisfaction Level")
print("HINT: df['satisfaction_level'].mean()")
print("RES:", df['satisfaction_level'].mean())

print("\n====== CHARTS ======")

print("\n1. Price per Cup by Continent")
print("HINT: df.groupby('continent')['price_per_cup'].sum().sort_values(ascending=False)")
print("RES:\n", df.groupby('continent')['price_per_cup'].sum().sort_values(ascending=False))

print("\n2. Monthly Spend by Income Level")
print("HINT: df.groupby('income_level')['monthly_spend'].sum().sort_values(ascending=False)")
print("RES:\n", df.groupby('income_level')['monthly_spend'].sum().sort_values(ascending=False))

print("\n3. Price per Cup vs. Monthly Spend (Head 5 points to save space)")
print("HINT: df.plot.scatter(x='monthly_spend', y='price_per_cup')")
print("RES (just columns):\n", df[['monthly_spend', 'price_per_cup']].head())

print("\n4. Drink Preference by Drink Category")
print("HINT: pd.crosstab(df['drink_preference'], df['drink_category'], normalize='index')")
print("RES:\n", pd.crosstab(df['drink_preference'], df['drink_category'], normalize='index'))

print("\n5. Average Price per Cup by Gender")
print("HINT: df.groupby('gender')['price_per_cup'].mean().sort_values(ascending=False)")
print("RES:\n", df.groupby('gender')['price_per_cup'].mean().sort_values(ascending=False))

print("\n6. Age vs. Heart Rate Correlation (Head 5 points to save space)")
print("HINT: df.plot.scatter(x='age', y='heart_rate')")
print("RES (just columns):\n", df[['age', 'heart_rate']].head())

print("\n7. Favorite Drink by Drink Preference")
print("HINT: pd.crosstab(df['favorite_drink'], df['drink_preference'], normalize='index')")
print("RES:\n", pd.crosstab(df['favorite_drink'], df['drink_preference'], normalize='index'))

print("\n8. Monthly Spend by Gender")
print("HINT: df.groupby('gender')['monthly_spend'].sum().sort_values(ascending=False)")
print("RES:\n", df.groupby('gender')['monthly_spend'].sum().sort_values(ascending=False))
