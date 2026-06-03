import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load Data
df = pd.read_csv('cleaned_superstore.csv')

# ---- KPIs ----
print("=== CORE KPIs ===")
print(f"Total Revenue: ${df['Sales'].sum():,.2f}")
print(f"Total Profit: ${df['Profit'].sum():,.2f}")
print(f"Avg Order Value: ${df['Sales'].mean():,.2f}")
print(f"Profit Margin: {(df['Profit'].sum()/df['Sales'].sum()*100):.2f}%")
print(f"Total Customers: {df['Customer ID'].nunique()}")

# ---- Customer Segmentation ----
customer_df = df.groupby('Customer ID').agg(
    Total_Sales=('Sales', 'sum'),
    Total_Profit=('Profit', 'sum'),
    Total_Orders=('Order ID', 'nunique')
).reset_index()

# Scale the data
scaler = StandardScaler()
scaled = scaler.fit_transform(customer_df[['Total_Sales','Total_Profit','Total_Orders']])

# KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
customer_df['Segment'] = kmeans.fit_predict(scaled)
customer_df['Segment'] = customer_df['Segment'].map({
    0: 'Low Value',
    1: 'Mid Value',
    2: 'High Value',
    3: 'VIP'
})

print("\n=== CUSTOMER SEGMENTS ===")
print(customer_df['Segment'].value_counts())

# ---- Plot 1: Segment Distribution ----
plt.figure(figsize=(8,5))
customer_df['Segment'].value_counts().plot(kind='bar', color=['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4'])
plt.title('Customer Segment Distribution')
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.savefig('segment_distribution.png')
plt.show()

# ---- Plot 2: Sales by Segment ----
plt.figure(figsize=(8,5))
customer_df.groupby('Segment')['Total_Sales'].mean().plot(kind='bar', color=['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4'])
plt.title('Average Sales by Customer Segment')
plt.xlabel('Segment')
plt.ylabel('Average Sales')
plt.tight_layout()
plt.savefig('sales_by_segment.png')
plt.show()

# ---- Plot 3: Scatter Plot ----
plt.figure(figsize=(8,5))
colors = {'Low Value':'red','Mid Value':'blue','High Value':'green','VIP':'gold'}
for seg, group in customer_df.groupby('Segment'):
    plt.scatter(group['Total_Sales'], group['Total_Profit'], 
                label=seg, color=colors[seg], alpha=0.6)
plt.title('Customer Segments - Sales vs Profit')
plt.xlabel('Total Sales')
plt.ylabel('Total Profit')
plt.legend()
plt.tight_layout()
plt.savefig('segment_scatter.png')
plt.show()

# Save segmented data
customer_df.to_csv('customer_segments.csv', index=False)
print("\n✅ All files saved!")