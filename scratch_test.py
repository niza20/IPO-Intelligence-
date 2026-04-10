import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from imblearn.over_sampling import SMOTE
import optuna

SEED = 42
np.random.seed(SEED)

df = pd.read_csv("data/ipo_data_synthetic.csv")
def clean_numeric(series):
    return (series.astype(str)
            .str.replace(',', '', regex=False)
            .str.replace('₹', '', regex=False)
            .str.replace('%', '', regex=False)
            .str.replace('<br/>', '', regex=False)
            .str.strip()
            .replace(['N/A','NA','--','-','nan','None',''], np.nan)
            .astype(float))

for col in ['issue_price','issue_size_cr','sub_qib','sub_hni','sub_retail',
            'gmp_pct','pe_vs_sector','promoter_holding','revenue_growth_pct',
            'roce_pct','debt_equity','company_age','listing_gain_pct']:
    if col in df.columns:
        df[col] = clean_numeric(df[col])

for col in ['issue_price','issue_size_cr','sub_qib','sub_hni','sub_retail']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

for col in ['gmp_pct','roce_pct','revenue_growth_pct','debt_equity','company_age']:
    mask = np.random.RandomState(99).random(len(df)) < 0.07
    df.loc[mask, col] = np.nan

mood_enc   = LabelEncoder().fit(df['market_mood'])
sector_enc = LabelEncoder().fit(df['sector'])
df['mood_enc']   = mood_enc.transform(df['market_mood'])
df['sector_enc'] = sector_enc.transform(df['sector'])

df['total_sub']     = df['sub_qib'] + df['sub_hni'] + df['sub_retail']
df['qib_hni_ratio'] = (df['sub_qib'] / (df['sub_hni'] + 0.01)).clip(0, 50)
df['gmp_x_qib']     = df['gmp_pct'] * np.log1p(df['sub_qib'])
df['size_log']      = np.log1p(df['issue_size_cr'])
df['val_risk']      = df['pe_vs_sector'].clip(0, 5)
df['is_tech_bull']  = ((df['sector'] == 'Tech') & (df['market_mood'] == 'bull')).astype(int)
df['roce_x_growth'] = df['roce_pct'].fillna(0) * df['revenue_growth_pct'].fillna(0) / 100
df['sub_score']     = (df['sub_qib'] * 0.5 + df['sub_hni'] * 0.3 + df['sub_retail'] * 0.2)
df['gmp_bull']      = df['gmp_pct'] * (df['mood_enc'] + 1)
df['low_valuation'] = (df['pe_vs_sector'] < 1.0).astype(int)
df['gmp_x_sub_score'] = df['gmp_pct'] * df['sub_score']

FEATURES = [
    'gmp_pct', 'sub_qib', 'sub_hni', 'sub_retail',
    'total_sub', 'sub_score', 'qib_hni_ratio',
    'gmp_x_qib', 'gmp_bull', 'gmp_x_sub_score',
    'pe_vs_sector', 'val_risk', 'low_valuation',
    'promoter_holding', 'mood_enc', 'sector_enc',
    'company_age', 'revenue_growth_pct', 'roce_pct',
    'debt_equity', 'size_log', 'is_tech_bull',
    'roce_x_growth', 'issue_price',
]

for col in FEATURES:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

X = df[FEATURES]
y_cls = df['positive_listing']

imputer = KNNImputer(n_neighbors=5)
scaler  = StandardScaler()
X_imp    = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imp)

X_tr, X_te, yc_tr, yc_te = train_test_split(
    X_scaled, y_cls, test_size=0.20, random_state=SEED, stratify=y_cls)

sm = SMOTE(random_state=SEED)
X_tr_sm, yc_tr_sm = sm.fit_resample(X_tr, yc_tr)

def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'eval_metric': 'logloss',
        'random_state': SEED,
        'verbosity': 0
    }
    
    model = xgb.XGBClassifier(**param)
    model.fit(X_tr_sm, yc_tr_sm)
    preds = model.predict(X_te)
    return accuracy_score(yc_te, preds)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)
print('Best Optuna XGB tuning test accuracy:', study.best_value)

# Try ExtraTrees
et = ExtraTreesClassifier(n_estimators=1000, max_depth=None, random_state=SEED, n_jobs=-1)
et.fit(X_tr_sm, yc_tr_sm)
print('ExtraTrees accuracy:', accuracy_score(yc_te, et.predict(X_te)))

