import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib

# تحميل البيانات من ملف CSV
df = pd.read_csv("data.csv")

# إعادة تسمية الأعمدة حسب الحاجة
if "المساحة" not in df.columns and "المساحة (متر مربع)" in df.columns:
    df.rename(columns={"المساحة (متر مربع)": "المساحة"}, inplace=True)
if "السعر" not in df.columns and "سعر الصفقة (ريال)" in df.columns:
    df.rename(columns={"سعر الصفقة (ريال)": "السعر"}, inplace=True)

# التأكد من أن الأعمدة المطلوبة موجودة
required_columns = ["المساحة", "النوع", "المدينة", "السعر"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"الأعمدة التالية مفقودة: {missing_columns}")

# استبدال الفاصل العشري "٫" بـ "." في الأعمدة الرقمية
df["المساحة"] = df["المساحة"].astype(str).str.replace("٫", ".")
df["السعر"] = df["السعر"].astype(str).str.replace("٫", ".")

# تحويل الأعمدة الرقمية إلى أرقام
df["المساحة"] = pd.to_numeric(df["المساحة"], errors="coerce")
df["السعر"] = pd.to_numeric(df["السعر"], errors="coerce")

# تنظيف البيانات: إزالة الصفوف التي بها قيم مفقودة أو أسعار <= 0
df = df.dropna(subset=["المساحة", "السعر", "النوع", "المدينة"])
df = df[df["السعر"] > 0]

if df.empty:
    raise ValueError("لا توجد بيانات كافية بعد التنظيف لتدريب النموذج. يرجى التحقق من الملف.")

# تحويل الأعمدة النصية إلى أرقام
df["نوع"] = df["النوع"].astype("category").cat.codes
df["مدينة"] = df["المدينة"].astype("category").cat.codes

X = df[["المساحة", "نوع", "مدينة"]]
y = df["السعر"]

# تقسيم البيانات إلى تدريب واختبار (80% تدريب، 20% اختبار)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# استخدام النموذج المدرب إذا كان موجوداً، أو تدريب نموذج جديد
try:
    model = joblib.load("model.pkl")
except:
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    joblib.dump(model, "model.pkl")

# توقع القيم لمجموعة الاختبار
y_pred = model.predict(X_test)

# حساب R²
r2 = r2_score(y_test, y_pred)
print("R²:", r2)
