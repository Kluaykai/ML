import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ==========================================
# ส่วนที่ 1: การสำรวจข้อมูล (Data Exploration)
# ==========================================
print("--- [1] กำลังเริ่มขั้นตอนการสำรวจข้อมูล ---")
# โหลดข้อมูล 10,000 แถวเพื่อให้ตรวจสอบได้ง่ายและรันรวดเร็ว
df = pd.read_csv('train.csv', )
print(f"ขนาดข้อมูล: {df.shape}")
print(df[['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'TARGET']].describe())

# ==========================================
# ส่วนที่ 2: การเตรียมข้อมูล (Data Preparation)
# ==========================================
print("\n--- [2] กำลังเตรียมข้อมูลและทำความสะอาด ---")
# เลือก Feature ที่เกี่ยวข้องกับการทำนายความเสี่ยง
features = [
    'TARGET', 'NAME_CONTRACT_TYPE', 'GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
    'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 
    'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'DAYS_AGE', 'DAYS_EMPLOYMENT',
    'EXT_SOURCE_2', 'EXT_SOURCE_3'
]
df = df[features].copy()

# ล้างข้อมูล: เปลี่ยนค่า Error ใน DAYS_EMPLOYMENT ให้เป็นค่าว่าง
df['DAYS_EMPLOYMENT'] = df['DAYS_EMPLOYMENT'].replace(365243, np.nan)

# เติมค่าว่างด้วยค่ากลาง (Median) สำหรับตัวเลข
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# เติมค่าว่างด้วยฐานนิยม (Mode) สำหรับตัวอักษร
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ==========================================
# ส่วนที่ 3: การสร้างคุณสมบัติ (Feature Engineering)
# ==========================================
print("--- [3] กำลังสร้าง Feature ใหม่และการทำ Encoding ---")
# แปลงวันเป็นปีเพื่อให้ข้อมูลสื่อความหมายดีขึ้น
df['Age'] = df['DAYS_AGE'].abs() / 365.25
df['Years_Employment'] = df['DAYS_EMPLOYMENT'].abs() / 365.25
# สร้าง DTI (Debt-to-Income Ratio)
df['DTI_Ratio'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] / 12)

# แปลงตัวอักษรเป็นตัวเลข (Label Encoding)
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# ลบคอลัมน์เก่าที่ไม่ใช้
df = df.drop(columns=['DAYS_AGE', 'DAYS_EMPLOYMENT'])

# แบ่งข้อมูล X และ y
X = df.drop('TARGET', axis=1)
y = df['TARGET']

# แบ่งข้อมูลเป็น Train 80% และ Test 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# การทำ Scaling (สำคัญมากสำหรับ Neural Network)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# ส่วนที่ 4: การสร้างโมเดล (Model Development)
# ==========================================
print("\n--- [4] กำลังสร้างและฝึกสอนโมเดล (3 Models) ---")
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
    "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

# ==========================================
# ส่วนที่ 5: การประเมินผลโมเดล (Model Evaluation)
# ==========================================
print("--- [5] กำลังประเมินผลและเปรียบเทียบโมเดล ---")
results = []
for name, model in models.items():
    print(f"รันโมเดล: {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # คำนวณค่า Metrics ต่างๆ
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    results.append({"Model": name, "R2_Score": r2, "MSE": mse, "MAE": mae, "RMSE": rmse})

# แสดงตารางเปรียบเทียบใน Terminal
results_df = pd.DataFrame(results)
print("\n--- ตารางเปรียบเทียบผลลัพธ์ ---")
print(results_df.to_string(index=False))

# สร้างกราฟเปรียบเทียบ R2_Score และ RMSE
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

sns.barplot(x='Model', y='R2_Score', data=results_df, ax=ax[0], palette='viridis')
ax[0].set_title('Comparison of R2 Score (Higher is Better)')

sns.barplot(x='Model', y='RMSE', data=results_df, ax=ax[1], palette='magma')
ax[1].set_title('Comparison of RMSE (Lower is Better)')

plt.tight_layout()
plt.savefig('evaluation_comparison.png')
print("\n--- ระบบรันเสร็จสิ้น: กราฟเปรียบเทียบถูกบันทึกในไฟล์ 'evaluation_comparison.png' ---")