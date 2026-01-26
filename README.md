## 👤 Author

- **Student Names:*Muhammad Zakariya(37581)* 
*Muhammad Mehtab(37582)*
*Muhammad Subhan(37819)*
- **Course:** Artificial Intelligence 
- **Institution:** IQRA UNIVERSITY H9


# KG-Inspired Medication Recommendation System (MIMIC-IV)

This project implements a **KG-inspired Random Forest model** for medication recommendation using the **MIMIC-IV Clinical Database (Demo version)**.  
The system analyzes **diseases, drugs, disease–drug relationships, and drug–drug interactions (DDI)** and predicts whether **Aspirin** is prescribed for a hospital admission.

---

## 📌 Project Overview

- Dataset: MIMIC-IV Clinical Database (Demo 2.2)
- Model: Random Forest (KG-inspired)
- Task: Binary classification (Aspirin prescribed or not)
- Platform: Google Colab
- Language: Python

---

## 📂 Dataset Used

The following tables from MIMIC-IV were used:

- `patients.csv` – Patient demographics (age, gender)
- `admissions.csv` – Hospital admission records
- `diagnoses_icd.csv` – Disease codes (ICD)
- `pharmacy.csv` – Prescribed medications
- `d_icd_diagnoses.csv` – ICD codes with disease names

The dataset is downloaded automatically using **kagglehub**.

---

## ⚙️ Methodology

1. Load and preprocess MIMIC-IV data  
2. Merge patient, admission, diagnosis, and pharmacy records  
3. Convert ICD codes to disease names  
4. Analyze:
   - Top diseases
   - Top prescribed drugs
   - Disease–drug relationships
   - Drug–drug interactions (same admission)  
5. Create KG-inspired features using diagnosis information  
6. Train a Random Forest classifier  
7. Evaluate using standard metrics  

---

## 📊 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve

All results are computed on unseen test data.

---

## 🧠 Model Description

- **Model**: Random Forest Classifier  
- **Features**:
  - Patient age
  - Patient gender
  - Diagnosis-based KG-inspired features  
- **Target**:
  - `1` → Aspirin prescribed  
  - `0` → Aspirin not prescribed  

The approach is **lightweight, interpretable**, and suitable for academic projects.

---

## 🔍 Drug–Drug Interaction (DDI)

Drug–drug interactions are analyzed by identifying **drug pairs prescribed together in the same hospital admission**.  
This provides a simple and realistic **DDI proxy** for safety analysis.

---

## ▶️ How to Run

1. Open the notebook in **Google Colab**
2. Run all cells
3. The dataset will be downloaded automatically
4. Results, tables, and graphs will be generated

---

## 📝 Notes

- This is a **KG-inspired approach**, not a full Graph Neural Network (KGDNet).
- The project focuses on **interpretability and simplicity**.
- Designed for **AI Lab / Final Year Project / Academic evaluation**.

---
---

## 📜 License

This project is for **educational and research purposes only**.
