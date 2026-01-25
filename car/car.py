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

# ปิดการแจ้งเตือนเพื่อความสะอาดของหน้าจอ
warnings.filterwarnings('ignore')

# 1. โหลดข้อมูล
print("--- [1] โหลดข้อมูลจากไฟล์ CSV ---")
df = pd.read_csv('Car_Price_Prediction.csv')

# ==========================================================
# 2. ขั้นตอนการตรวจสอบและทำความสะอาดข้อมูล (Data Cleaning & Checking)
# ==========================================================
print("\n--- [2] เริ่มขั้นตอนการตรวจสอบคุณภาพข้อมูล (Data Quality Check) ---")

# 2.1 ตรวจสอบค่าว่าง (Missing Values)
missing_count = df.isnull().sum().sum()
print(f"- ตรวจสอบค่าว่าง: พบ {missing_count} จุด")
if missing_count > 0:
    df = df.dropna()
    print("  (ทำการลบแถวที่มีค่าว่างออกเรียบร้อยแล้ว)")

# 2.2 ตรวจสอบค่าที่ผิดตรรกะ (Logical Anomaly Check)
anomalies = df[(df['Price'] <= 0) | (df['Mileage'] < 0) | (df['Engine Size'] <= 0)]
if not anomalies.empty:
    print(f"- พบข้อมูลผิดตรรกะ {len(anomalies)} แถว (กำลังลบออก...)")
    df = df[(df['Price'] > 0) & (df['Mileage'] >= 0) & (df['Engine Size'] > 0)]
else:
    print("- ตรวจสอบค่าผิดตรรกะ: ไม่พบความผิดปกติ (ราคา, เลขไมล์ และขนาดเครื่องยนต์ถูกต้อง)")

# 2.3 ตรวจสอบและจัดการค่าที่สูง/ต่ำผิดปกติ (Outlier Detection using IQR)
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Price'] < lower_bound) | (df['Price'] > upper_bound)]
print(f"- ตรวจสอบ Outliers: พบราคาที่โดดผิดปกติ {len(outliers)} แถว")

df = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)]
print(f"--- สรุป: คงเหลือข้อมูลคุณภาพดีทั้งหมด {len(df)} แถว ---")

# ==========================================================
# 3. เตรียมข้อมูลและสร้างคุณสมบัติใหม่ (Feature Engineering)
# ==========================================================
print("\n--- [3] การเตรียมข้อมูล (Preprocessing) ---")

# สร้างตัวแปร อายุรถ (Car_Age) โดยอ้างอิงปีปัจจุบัน 2026
df['Car_Age'] = 2026 - df['Year']

# แปลงข้อมูลตัวอักษรเป็นตัวเลข (Encoding) เพื่อให้ AI คำนวณได้
le = LabelEncoder()
cat_cols = ['Make', 'Model', 'Fuel Type', 'Transmission']
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# ตัดคอลัมน์ Year ออกเพราะเราใช้ Car_Age แทนแล้ว
df_prepared = df.drop(columns=['Year'])

# --- [ ส่วนที่เพิ่มเข้ามา: บันทึกไฟล์ข้อมูลที่แปลงเสร็จแล้ว ] ---
# บันทึกข้อมูลที่สะอาดและแปลงเป็นตัวเลขแล้วลง CSV ก่อนเอาไปเทรน
df_prepared.to_csv('transformed_car_data.csv', index=False, encoding='utf-8-sig')
print(">>> บันทึกไฟล์ข้อมูลที่แปลงแล้วไว้ที่: transformed_car_data.csv (กรุณาตรวจสอบไฟล์นี้) <<<")
# --------------------------------------------------------

# แยกตัวแปรต้น (X) และตัวแปรเป้าหมาย (y)
X = df_prepared.drop(['Price'], axis=1)
y = df_prepared['Price']

# แบ่งข้อมูลสำหรับ Train (80%) และ Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# การปรับมาตรฐานข้อมูล (Scaling) ให้ทุกตัวแปรอยู่ในสเกลเดียวกัน
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================================
# 4. การสร้างและประเมินผลโมเดล (Model Training & Evaluation)
# ==========================================================
print("\n--- [4] กำลังฝึกสอนโมเดล 3 รูปแบบตามเงื่อนไขโปรเจค ---")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
}

results = []

for name, model in models.items():
    print(f"กำลังรัน: {name}...")
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

# ==========================================================
# 5. การสร้าง Output สำหรับรายงาน (Graphs & Tables)
# ==========================================================
results_df = pd.DataFrame(results)

# 5.1 บันทึกตารางเปรียบเทียบเป็นไฟล์ CSV สำหรับ Excel
results_df.to_csv('car_model_comparison.csv', index=False, encoding='utf-8-sig')

# 5.2 สร้างกราฟแท่งเปรียบเทียบ R2 Score
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='R2_Score', data=results_df, palette='magma')
plt.title('Comparison of R2 Score (Car Price Prediction)')
plt.ylim(0, 1.0)
plt.savefig('car_r2_comparison.png')

# 5.3 สร้างรูปภาพตารางสรุปผล
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')
table = ax.table(cellText=results_df.values, colLabels=results_df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2)
plt.title('Model Performance Summary', pad=20)
plt.savefig('car_result_table.png', bbox_inches='tight', dpi=300)

print("\n" + "="*50)
print("รันเสร็จสมบูรณ์!")
print("1. transformed_car_data.csv (ข้อมูลที่แปลงแล้วก่อนเทรน)")
print("2. car_model_comparison.csv (ตารางเปรียบเทียบผล)")
print("3. car_r2_comparison.png (กราฟ)")
print("4. car_result_table.png (รูปตารางสรุป)")
print("="*50)
print(results_df.to_string(index=False))