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

# ปิดการแจ้งเตือนคำเตือน
warnings.filterwarnings('ignore')

# ==========================================
# 1. การสำรวจและโหลดข้อมูล (Data Exploration)
# ==========================================
print("--- [1] กำลังโหลดและสำรวจข้อมูล ---")
try:
    df = pd.read_csv('Car_Price_Prediction.csv')
    print(f"โหลดข้อมูลสำเร็จ: {len(df)} แถว")
except FileNotFoundError:
    print("Error: ไม่พบไฟล์ Car_Price_Prediction.csv ในโฟลเดอร์")
    exit()

# ==========================================
# 2. การเตรียมข้อมูล & Feature Engineering
# ==========================================
print("--- [2] กำลังจัดการข้อมูลและสร้างคุณสมบัติใหม่ ---")
# จัดการค่าว่าง
df = df.dropna()

# สร้างตัวแปร อายุรถ (Car_Age) โดยอ้างอิงปี 2026
df['Car_Age'] = 2026 - df['Year']

# แปลงข้อมูลตัวอักษรเป็นตัวเลข (Encoding)
le = LabelEncoder()
cat_cols = ['Make', 'Model', 'Fuel Type', 'Transmission']
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# เลือกตัวแปรต้น (X) และตัวแปรเป้าหมาย (y)
X = df.drop(['Price', 'Year'], axis=1) # ลบ Price เพราะเป็นคำตอบ และ Year เพราะใช้ Car_Age แทนแล้ว
y = df['Price']

# แบ่งข้อมูลสำหรับฝึกสอนและทดสอบ (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# การปรับมาตรฐานข้อมูล (Scaling) - สำคัญมากสำหรับ Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. การสร้างโมเดล (Model Development)
# ==========================================
print("--- [3] กำลังฝึกสอนโมเดลทั้ง 3 รูปแบบ ---")
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
}

# ==========================================
# 4. การประเมินผล (Model Evaluation)
# ==========================================
print("--- [4] กำลังประเมินผลลัพธ์ ---")
results = []
for name, model in models.items():
    print(f"รันโมเดล: {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # คำนวณค่าสถิติต่างๆ
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    results.append({
        "Model": name,
        "R2_Score": round(r2, 4),
        "MSE": round(mse, 2),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2)
    })

results_df = pd.DataFrame(results)

# ==========================================
# 5. การบันทึกผลลัพธ์ (Output Generation)
# ==========================================
# 5.1 บันทึกเป็นไฟล์ CSV (สำหรับเปิดใน Excel)
results_df.to_csv('car_model_comparison.csv', index=False, encoding='utf-8-sig')

# 5.2 บันทึกเป็นรูปภาพกราฟแท่ง
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='R2_Score', data=results_df, palette='viridis')
plt.title('Comparison of R2 Score (Accuracy)')
plt.ylim(0, 1.1)
plt.savefig('car_r2_plot.png')

# 5.3 บันทึกตารางสรุปผลเป็นรูปภาพ (เพื่อให้แปะในรายงานได้ทันที)
fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('off')
table = ax.table(cellText=results_df.values, colLabels=results_df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
plt.savefig('car_table_image.png', bbox_inches='tight', dpi=300)

print("\n" + "="*40)
print("สรุปผลการรันสำเร็จ!")
print("-" * 40)
print(results_df.to_string(index=False))
print("="*40)
print("บันทึกไฟล์เรียบร้อย:")
print("1. car_model_comparison.csv (ตารางสำหรับ Excel)")
print("2. car_r2_plot.png (กราฟเปรียบเทียบ)")
print("3. car_table_image.png (รูปภาพตารางสรุปผล)")