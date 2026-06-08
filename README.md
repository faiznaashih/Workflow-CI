# Workflow-CI

Repository CI/CD untuk training model Machine Learning secara otomatis menggunakan **MLflow Project** dan **GitHub Actions**.

**Author**: M. Faiz Naashih Rozaq

---

## 📁 Struktur Folder

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml              ← GitHub Actions CI workflow
└── MLProject/
    ├── modelling.py            ← Script training model
    ├── conda.yaml              ← Environment dependencies
    ├── MLProject               ← MLflow Project config
    └── telco_preprocessing/    ← Dataset preprocessed
        ├── X_train.csv
        ├── X_test.csv
        ├── y_train.csv
        └── y_test.csv
```

---

## ⚙️ Setup GitHub Secrets

Tambahkan dua secrets di **Settings → Secrets and variables → Actions**:

| Secret | Value |
|---|---|
| `MLFLOW_TRACKING_USERNAME` | `faiznaashih` |
| `MLFLOW_TRACKING_PASSWORD` | `<dagshub_token>` |

---

## 🚀 Cara Kerja CI

Workflow otomatis berjalan ketika ada **push ke branch `main`**:
1. Install dependencies
2. Jalankan `mlflow run MLProject/`
3. Training model & log ke DagsHub
4. Simpan artefak (confusion matrix, ROC curve, classification report) ke repo

---

## 🔗 DagsHub

Hasil experiment tersimpan di:
[https://dagshub.com/faiznaashih/Eksperimen_SML_Faiz-Naashih](https://dagshub.com/faiznaashih/Eksperimen_SML_Faiz-Naashih)
