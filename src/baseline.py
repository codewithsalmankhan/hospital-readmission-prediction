"""Explore UCI diabetes data and fit an initial patient-separated baseline."""
from pathlib import Path
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / 'data/diabetic_data.csv', na_values='?')
y = (df.readmitted == '<30').astype(int)
features = ['race', 'gender', 'age', 'admission_type_id', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses', 'change', 'diabetesMed']
cat = ['race', 'gender', 'age', 'admission_type_id', 'change', 'diabetesMed']
num = [c for c in features if c not in cat]
X = df[features].copy()
X[cat] = X[cat].astype('object').fillna('Missing').astype(str)
train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups=df.patient_nbr))
pre = ColumnTransformer([('numeric', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), num), ('categorical', OneHotEncoder(handle_unknown='ignore'), cat)])
model = make_pipeline(pre, LogisticRegression(max_iter=1000))
model.fit(X.iloc[train], y.iloc[train])
prob = model.predict_proba(X.iloc[test])[:, 1]
pred = (prob >= .5).astype(int)
results = {'encounters': len(df), 'patients': int(df.patient_nbr.nunique()), 'target_counts': df.readmitted.value_counts().to_dict(), 'missing_percent': (df.isna().mean().sort_values(ascending=False).head(5)*100).round(2).to_dict(), 'train_encounters': len(train), 'test_encounters': len(test), 'patient_overlap': int(len(set(df.patient_nbr.iloc[train]) & set(df.patient_nbr.iloc[test]))), 'test_positive_rate': float(y.iloc[test].mean()), 'roc_auc': roc_auc_score(y.iloc[test], prob), 'average_precision': average_precision_score(y.iloc[test], prob), 'precision_at_0_5': precision_score(y.iloc[test], pred, zero_division=0), 'recall_at_0_5': recall_score(y.iloc[test], pred, zero_division=0), 'confusion_matrix_at_0_5': confusion_matrix(y.iloc[test], pred).tolist()}
(ROOT / 'results').mkdir(exist_ok=True)
(ROOT / 'results/baseline.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
