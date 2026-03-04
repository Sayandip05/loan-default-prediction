# 🚀 Quick Start Guide - Loan Default Prediction

## ✅ What I've Created For You

Your project is now **COMPLETE** with:

### 📁 **Complete Project Structure**
```
loan-default-prediction/
├── data/                          # Data directories
├── models/                        # Saved models
├── mlruns/                        # MLflow experiments
├── notebooks/                     # Jupyter notebook
├── src/
│   ├── data_pipeline/            ✅ Preprocessing & Feature Engineering
│   ├── model/                    ✅ Training & Prediction
│   ├── api/                      ✅ FastAPI backend
│   └── frontend/                 ✅ Streamlit UI
├── Dockerfile                    ✅ Multi-stage Docker
├── docker-compose.yml            ✅ All services
├── dvc.yaml                      ✅ Pipeline definition
├── params.yaml                   ✅ Hyperparameters
└── requirements.txt              ✅ Dependencies
```

---

## 🎯 Step-by-Step Execution

### **Phase 1: Download Dataset (5 mins)**

1. **Go to Kaggle**:
   - https://www.kaggle.com/c/GiveMeSomeCredit/data
   - Download `cs-training.csv`

2. **Place in project**:
   ```bash
   # Copy to data/raw/
   cp ~/Downloads/cs-training.csv "data/raw/"
   ```

3. **Track with DVC**:
   ```bash
   dvc add data/raw/cs-training.csv
   git add data/raw/.gitignore data/raw/cs-training.csv.dvc
   git commit -m "Add raw data"
   ```

---

### **Phase 2: Run DVC Pipeline (10 mins)**

```bash
# Run entire pipeline (preprocess → feature_engineering → train)
dvc repro

# This will:
# ✅ Preprocess data (handle nulls, outliers)
# ✅ Engineer features
# ✅ Train XGBoost model with MLflow tracking
# ✅ Save model to models/model.pkl
```

**Expected Output:**
```
✅ Preprocessing complete!
✅ Feature engineering complete!
✅ Training complete! ROC-AUC: 0.86
💾 Model saved to: models/model.pkl
```

---

### **Phase 3: Start MLflow UI (Optional)**

```bash
# Terminal 1: Start MLflow tracking server
mlflow ui

# Visit: http://localhost:5000
# View experiments, metrics, parameters
```

---

### **Phase 4: Test Prediction Module**

```bash
# Test if model works
python src/model/predict.py

# Expected output:
# 🔮 Prediction Result:
#   Prediction: Default / No Default
#   Default Probability: 45.2%
#   Risk Level: Medium Risk
```

---

### **Phase 5: Run FastAPI Backend**

```bash
# Terminal 2: Start API
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Visit API docs: http://localhost:8000/docs
# Test endpoints interactively
```

**Test API:**
```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "RevolvingUtilizationOfUnsecuredLines": 0.77,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 2,
    "DebtRatio": 0.80,
    "MonthlyIncome": 9120,
    "NumberOfOpenCreditLinesAndLoans": 13,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 6,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2
  }'
```

---

### **Phase 6: Run Streamlit Frontend**

```bash
# Terminal 3: Start Streamlit
streamlit run src/frontend/streamlit_app.py

# Visit: http://localhost:8501
```

**Features:**
- 🔮 Single prediction with interactive form
- 📊 Batch prediction via CSV upload
- 📈 Visualizations (gauges, charts)
- 📄 Model info and feature importance

---

### **Phase 7: Docker Deployment (All-in-One)**

```bash
# Build and start all services
docker-compose up --build

# Services running:
# - MLflow UI:     http://localhost:5000
# - FastAPI:       http://localhost:8000
# - Streamlit:     http://localhost:8501
```

**Stop services:**
```bash
docker-compose down
```

---

## 🧪 Testing Checklist

- [ ] Dataset downloaded to `data/raw/cs-training.csv`
- [ ] DVC pipeline runs successfully (`dvc repro`)
- [ ] Model file exists at `models/model.pkl`
- [ ] MLflow UI shows experiment runs
- [ ] FastAPI `/health` endpoint returns healthy
- [ ] Streamlit app loads without errors
- [ ] Single prediction works
- [ ] Batch prediction works with sample CSV
- [ ] Docker services start successfully

---

## 📊 Sample CSV for Batch Testing

Create `sample_test.csv`:

```csv
RevolvingUtilizationOfUnsecuredLines,age,NumberOfTime30-59DaysPastDueNotWorse,DebtRatio,MonthlyIncome,NumberOfOpenCreditLinesAndLoans,NumberOfTimes90DaysLate,NumberRealEstateLoansOrLines,NumberOfTime60-89DaysPastDueNotWorse,NumberOfDependents
0.766127,45,2,0.802982,9120,13,0,6,0,2
0.123456,32,0,0.456789,5500,8,0,2,0,1
0.987654,58,5,1.234567,12000,20,2,8,1,3
```

Upload this in Streamlit → Batch Prediction

---

## 🐛 Troubleshooting

### **Issue: Model not found**
```bash
# Make sure you ran DVC pipeline
dvc repro
```

### **Issue: MLflow connection error**
```bash
# Start MLflow server
mlflow ui --host 0.0.0.0 --port 5000
```

### **Issue: Import errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### **Issue: DVC errors**
```bash
# Re-initialize DVC
dvc init --force
dvc repro
```

---

## 📈 Next Steps (Future Enhancements)

1. **Jupyter Notebook**: Open `notebooks/01_model_development.ipynb` for EDA
2. **AWS Deployment**: Setup EKS, ECR, CodePipeline
3. **CI/CD**: Add GitHub Actions workflow
4. **Monitoring**: Add Prometheus + Grafana
5. **A/B Testing**: Deploy multiple model versions

---

## 🎉 You're All Set!

Your complete MLOps project is ready. Start with:

```bash
# 1. Download dataset
# 2. Run pipeline
dvc repro

# 3. Start services
streamlit run src/frontend/streamlit_app.py
```

**Need help?** Check:
- API docs: http://localhost:8000/docs
- MLflow UI: http://localhost:5000
- Streamlit: http://localhost:8501

---

**Happy Predicting! 🚀**
