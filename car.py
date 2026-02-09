import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

df = pd.read_csv('Car_Price_Prediction.csv')


print("--- Data Exploration ---")
print(df.info()) 
print(df.describe()) 


plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], kde=True, color='blue')
plt.title('Distribution of Car Prices')
plt.savefig('price_distribution.png')


plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')


initial_rows = len(df)

df = df.drop_duplicates()
rows_after_dup = len(df)

if df.isnull().values.any():
    df = df.dropna()
rows_after_null = len(df)

df = df[(df['Price'] > 0) & (df['Mileage'] >= 0) & (df['Engine Size'] > 0)]
rows_after_logic = len(df)

numeric_cols = ['Price', 'Mileage', 'Engine Size']
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    iqr = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * iqr) & (df[col] <= Q3 + 1.5 * iqr)]
rows_after_outlier = len(df)

categorical_cols = ['Make', 'Model', 'Fuel Type', 'Transmission']
for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip()

print("\n" + "="*30)
print(" PRE-PROCESSING SUMMARY ")
print("="*30)
print(f"Original Data: {initial_rows} rows")
print(f"1. Removed Duplicates: {initial_rows - rows_after_dup} rows")
print(f"2. Removed Null Values: {rows_after_dup - rows_after_null} rows")
print(f"3. Logic Filtering (Price/Mileage/Engine): {rows_after_null - rows_after_logic} rows")
print(f"4. Outlier Removal: {rows_after_logic - rows_after_outlier} rows")
print(f"Final Data Remaining: {len(df)} rows (Retention: {(len(df)/initial_rows)*100:.2f}%)")
print("="*30 + "\n")

df['Car_Age'] = 2026 - df['Year']

le = LabelEncoder()
for col in ['Make', 'Model', 'Fuel Type', 'Transmission']:
    df[col] = le.fit_transform(df[col])

df_prepared = df.drop(columns=['Year'])
df_prepared.to_csv('prepared_car_data.csv', index=False, encoding='utf-8-sig')

X = df_prepared.drop(['Price'], axis=1)
y = df_prepared['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    results.append({
        "Model": name,
        "R2_Score": round(r2_score(y_test, y_pred), 4),
        "CV_R2_Mean": round(cv_scores.mean(), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        "MAE": round(mean_absolute_error(y_test, y_pred), 2)
    })

results_df = pd.DataFrame(results)
results_df.to_csv('model_performance_results.csv', index=False, encoding='utf-8-sig')
print(results_df.to_string(index=False))

plt.figure(figsize=(10, 6))
feat_importances = pd.Series(models["Random Forest"].feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh', color='teal')
plt.title('Top Factors Affecting Price')
plt.savefig('car_feature_importance.png')

plt.figure(figsize=(8, 8))
plt.scatter(y_test, models["Random Forest"].predict(X_test_scaled), alpha=0.5, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.savefig('car_actual_vs_predicted.png')

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='R2_Score', data=results_df, palette='magma')
plt.title('Comparison of R2 Score')
plt.savefig('car_r2_comparison.png')

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')
the_table = ax.table(cellText=results_df.values, colLabels=results_df.columns, loc='center', cellLoc='center')
the_table.auto_set_font_size(False)
the_table.set_fontsize(12)
the_table.scale(1.2, 2.5)

for (row, col), cell in the_table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2c3e50')
    else:
        cell.set_facecolor('#f2f2f2')

plt.title('Final Model Performance Summary', fontsize=16, pad=20, weight='bold')
plt.savefig('model_performance_table.png', bbox_inches='tight', dpi=300)

print("\nAll files and visualizations have been generated successfully.")