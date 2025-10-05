# Kaggle_Obesity_Full_Setup.py
import numpy as np
import pandas as pd
import warnings
import sys
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
import optuna

# -----------------------------
# Load data
# -----------------------------
train_path = "./train.csv"  # adjust paths
test_path = "./test.csv"
sub_path = "./sample_submission.csv"

df = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_sub = pd.read_csv(sub_path)

target = 'Weight_Category'

# Drop ID columns
if 'PersonID' in df.columns:
    df.drop('PersonID', axis=1, inplace=True)
    df_test.drop('PersonID', axis=1, inplace=True)

# -----------------------------
# Preprocessing
# -----------------------------
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
if target in num_cols: num_cols.remove(target)

# Boolean-like columns
bool_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype != 'int']
for col in bool_cols:
    df[col] = df[col].astype(int)
    df_test[col] = df_test[col].astype(int)

# Label encoding
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
    df_test[col] = le.transform(df_test[col])

# Standard scaling
scaler = StandardScaler()
X_train = df.drop(target, axis=1)
y_train = df[target]
X_test = df_test.copy()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# -----------------------------
# Mode selection
# -----------------------------
mode = "baseline"
n_trials = 50
if "--mode" in sys.argv:
    mode_idx = sys.argv.index("--mode") + 1
    mode = sys.argv[mode_idx]
if "--n_trials" in sys.argv:
    trials_idx = sys.argv.index("--n_trials") + 1
    n_trials = int(sys.argv[trials_idx])

# -----------------------------
# Helper: OOF CV scoring
# -----------------------------
def get_oof_predictions(model, X, y, X_test, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    test_preds = np.zeros((len(X_test), n_splits))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict(X_val)
        test_preds[:, fold] = model.predict(X_test)
        
        acc = accuracy_score(y_val, oof_preds[val_idx])
        print(f"Fold {fold+1} Accuracy: {acc:.4f}")
    
    oof_acc = accuracy_score(y, oof_preds)
    print(f"OOF CV Accuracy: {oof_acc:.4f}")
    final_test_pred = np.round(np.mean(test_preds, axis=1))
    return oof_preds, final_test_pred

# -----------------------------
# Baseline XGBoost
# -----------------------------
if mode == "baseline":
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    oof_preds, test_preds = get_oof_predictions(model, X_train, y_train, X_test)
    df_sub[target] = test_preds
    df_sub.to_csv("submission_baseline.csv", index=False)
    print("Baseline submission saved as submission_baseline.csv")

# -----------------------------
# Stacking
# -----------------------------
elif mode == "stack":
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=500, max_depth=8, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=5)),
        ('xgb', XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6, use_label_encoder=False, eval_metric='mlogloss'))
    ]
    stack_model = StackingClassifier(
        estimators=estimators,
        final_estimator=XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4, use_label_encoder=False, eval_metric='mlogloss'),
        cv=5
    )
    oof_preds, test_preds = get_oof_predictions(stack_model, X_train, y_train, X_test)
    df_sub[target] = test_preds
    df_sub.to_csv("submission_stack.csv", index=False)
    print("Stacking submission saved as submission_stack.csv")

# -----------------------------
# Hyperparameter tuning
# -----------------------------
elif mode == "tune":
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
        }
        clf = XGBClassifier(**param, eval_metric='mlogloss', use_label_encoder=False)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            clf.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            pred = clf.predict(X_train.iloc[val_idx])
            scores.append(accuracy_score(y_train.iloc[val_idx], pred))
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    print("Best parameters:", study.best_params)

    tuned_model = XGBClassifier(**study.best_params, eval_metric='mlogloss', use_label_encoder=False)
    oof_preds, test_preds = get_oof_predictions(tuned_model, X_train, y_train, X_test)
    df_sub[target] = test_preds
    df_sub.to_csv("submission_tuned.csv", index=False)
    print("Tuned submission saved as submission_tuned.csv")
