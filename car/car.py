import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

df = pd.read_csv('Car_Price_Prediction.csv')

missing_count = df.isnull().sum().sum()
if missing_count > 0:
    df = df.dropna()

anomalies = df[(df['Price'] <= 0) | (df['Mileage'] < 0) | (df['Engine Size'] <= 0)]
if not anomalies.empty:
    df = df[(df['Price'] > 0) & (df['Mileage'] >= 0) & (df['Engine Size'] > 0)]

Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)]

df['Car_Age'] = 2026 - df['Year']

le = LabelEncoder()
cat_cols = ['Make', 'Model', 'Fuel Type', 'Transmission']
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

df_prepared = df.drop(columns=['Year'])
df_prepared.to_csv('transformed_car_data.csv', index=False, encoding='utf-8-sig')

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
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    results.append({
        "Model": name,
        "R2_Score": round(r2, 4),
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "MSE": round(mse, 2)
    })

results_df = pd.DataFrame(results)
results_df.to_csv('car_model_comparison.csv', index=False, encoding='utf-8-sig')

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='R2_Score', data=results_df, palette='magma')
plt.title('Comparison of R2 Score (Car Price Prediction)')
plt.ylim(0, 1.0)
plt.savefig('car_r2_comparison.png')

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')
table = ax.table(cellText=results_df.values, colLabels=results_df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2)
plt.savefig('car_result_table.png', bbox_inches='tight', dpi=300)

print(results_df.to_string(index=False))