from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import joblib
import shutil
import os
from sklearn.ensemble import RandomForestRegressor

app = FastAPI()

# نخدم الملفات الثابتة من مسار "/static" بدلاً من الجذر
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    المساحة: float
    النوع: str
    المدينة: str

try:
    model = joblib.load("model.pkl")
except:
    model = None

@app.get("/")
def root():
    # نقوم بإرجاع index.html عند الطلب على الجذر
    return FileResponse("index.html", media_type="text/html")

@app.post("/predict")
def predict(data: InputData):
    if model is None:
        return {"message": "❌ النموذج غير مدرب بعد."}
    df = pd.DataFrame([{
        "المساحة": data.المساحة,
        "نوع": pd.Series([data.النوع]).astype("category").cat.codes[0],
        "مدينة": pd.Series([data.المدينة]).astype("category").cat.codes[0]
    }])
    prediction = model.predict(df)[0]
    safe_zone = prediction * 0.85
    return {
        "السعر المتوقع": f"{round(prediction):,} ريال",
        "الحد الآمن للمزايدة": f"{round(safe_zone):,} ريال"
    }

@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    with open("data.csv", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    df = pd.read_csv("data.csv")
    
    # إعادة تسمية الأعمدة إذا كانت بأسماء بديلة
    col_mapping = {
        "المساحة (متر مربع)": "المساحة",
        "سعر الصفقة (ريال)": "السعر"
    }
    for old, new in col_mapping.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)
    
    required_columns = ["المساحة", "النوع", "المدينة", "السعر"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return {"message": f"الأعمدة التالية مفقودة: {missing_columns}"}
    
    # استبدال الفاصل العشري "٫" بـ "." وتحويل الأعمدة الرقمية إلى أرقام
    df["المساحة"] = df["المساحة"].astype(str).str.replace("٫", ".")
    df["المساحة"] = pd.to_numeric(df["المساحة"], errors="coerce")
    
    df["السعر"] = df["السعر"].astype(str).str.replace("٫", ".")
    df["السعر"] = pd.to_numeric(df["السعر"], errors="coerce")
    
    # تنظيف البيانات: إزالة الصفوف التي بها قيم مفقودة أو أسعار <= 0
    df = df.dropna(subset=["المساحة", "السعر", "النوع", "المدينة"])
    df = df[df["السعر"] > 0]
    
    if df.empty:
        return {"message": "لا توجد بيانات كافية بعد التنظيف لتدريب النموذج. يرجى التحقق من الملف."}
    
    # تحويل الأعمدة النصية إلى رموز رقمية
    df["نوع"] = df["النوع"].astype("category").cat.codes
    df["مدينة"] = df["المدينة"].astype("category").cat.codes
    
    X = df[["المساحة", "نوع", "مدينة"]]
    y = df["السعر"]
    
    global model
    model = RandomForestRegressor()
    model.fit(X, y)
    joblib.dump(model, "model.pkl")
    
    return {"message": "✅ تم تدريب النموذج بنجاح!"}



