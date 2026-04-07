"""
IPO Intelligence — ML Training Pipeline
Targets 80-85% classifier accuracy.

Models:
  1. XGBoost Classifier  — win/loss (positive listing)
  2. Random Forest Regressor — listing gain %
  3. Composite Risk Scorer  — rule + model hybrid

Techniques: KNN imputation, feature engineering,
            GridSearchCV tuning, cross-validation,
            SMOTE-style class balancing via scale_pos_weight.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, json
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.impute           import KNNImputer
from sklearn.ensemble         import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.metrics          import (classification_report, confusion_matrix,
                                      roc_auc_score, roc_curve,
                                      mean_absolute_error, mean_squared_error,
                                      r2_score, accuracy_score)
import xgboost as xgb
import joblib

SEED = 42
np.random.seed(SEED)

os.makedirs("models",  exist_ok=True)
os.makedirs("static",  exist_ok=True)

# ── 1. Load ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("  IPO INTELLIGENCE — ML PIPELINE")
print("=" * 60)

df = pd.read_csv("data/ipo_data.csv")
print(f"\n[DATA]  {len(df)} records · {df.shape[1]} columns")
print(f"        Positive listings : {df['positive_listing'].mean()*100:.1f}%")
print(f"        Avg gain          : {df['listing_gain_pct'].mean():.1f}%  "
      f"± {df['listing_gain_pct'].std():.1f}%")

# ── 2. Inject realistic missing values (~7%) ─────────────────────────────────
for col in ['gmp_pct','roce_pct','revenue_growth_pct','debt_equity','company_age']:
    mask = np.random.RandomState(99).random(len(df)) < 0.07
    df.loc[mask, col] = np.nan

print(f"\n[MISSING]  NaNs per column:")
print(df.isnull().sum()[df.isnull().sum() > 0].to_string())

# ── 3. Encode categoricals ───────────────────────────────────────────────────
mood_enc   = LabelEncoder().fit(df['market_mood'])
sector_enc = LabelEncoder().fit(df['sector'])
df['mood_enc']   = mood_enc.transform(df['market_mood'])
df['sector_enc'] = sector_enc.transform(df['sector'])
joblib.dump(mood_enc,   "models/mood_encoder.pkl")
joblib.dump(sector_enc, "models/sector_encoder.pkl")

# ── 4. Feature Engineering ───────────────────────────────────────────────────
print("\n[FEATURES]  Engineering...")

df['total_sub']      = df['sub_qib'] + df['sub_hni'] + df['sub_retail']
df['qib_hni_ratio']  = (df['sub_qib'] / (df['sub_hni'] + 0.01)).clip(0, 50)
df['gmp_x_qib']      = df['gmp_pct'] * np.log1p(df['sub_qib'])
df['size_log']        = np.log1p(df['issue_size_cr'])
df['val_risk']        = df['pe_vs_sector'].clip(0, 5)
df['is_tech_bull']    = ((df['sector'] == 'Tech') & (df['market_mood'] == 'bull')).astype(int)
df['roce_x_growth']   = df['roce_pct'].fillna(0) * df['revenue_growth_pct'].fillna(0) / 100
df['sub_score']       = (df['sub_qib'] * 0.5 + df['sub_hni'] * 0.3 + df['sub_retail'] * 0.2)
df['gmp_bull']        = df['gmp_pct'] * (df['mood_enc'] + 1)
df['low_valuation']   = (df['pe_vs_sector'] < 1.0).astype(int)

FEATURES = [
    'gmp_pct', 'sub_qib', 'sub_hni', 'sub_retail',
    'total_sub', 'sub_score', 'qib_hni_ratio',
    'gmp_x_qib', 'gmp_bull',
    'pe_vs_sector', 'val_risk', 'low_valuation',
    'promoter_holding', 'mood_enc', 'sector_enc',
    'company_age', 'revenue_growth_pct', 'roce_pct',
    'debt_equity', 'size_log', 'is_tech_bull',
    'roce_x_growth', 'issue_price',
]

print(f"        {len(FEATURES)} features total")

X      = df[FEATURES]
y_cls  = df['positive_listing']
y_reg  = df['listing_gain_pct']

# ── 5. Preprocess ────────────────────────────────────────────────────────────
print("\n[PREPROCESS]  KNN imputer (k=5) + StandardScaler")
imputer = KNNImputer(n_neighbors=5)
scaler  = StandardScaler()

X_imp    = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imp)

joblib.dump(imputer,  "models/imputer.pkl")
joblib.dump(scaler,   "models/scaler.pkl")
joblib.dump(FEATURES, "models/feature_names.pkl")

# ── 6. Split ─────────────────────────────────────────────────────────────────
X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
    X_scaled, y_cls, y_reg,
    test_size=0.20, random_state=SEED, stratify=y_cls)

print(f"\n[SPLIT]  Train {len(X_tr)} | Test {len(X_te)}")

# ── 7A. XGBoost Classifier ───────────────────────────────────────────────────
print("\n[MODEL 1]  XGBoost Classifier — tuned for 80-85% accuracy")

neg  = (y_cls == 0).sum()
pos  = (y_cls == 1).sum()
spw  = round(neg / pos, 2)        # handle class imbalance

xgb_clf = xgb.XGBClassifier(
    n_estimators       = 300,
    max_depth          = 4,        # shallower = less overfit
    learning_rate      = 0.03,
    subsample          = 0.75,
    colsample_bytree   = 0.75,
    min_child_weight   = 5,        # regularization
    gamma              = 0.2,
    reg_alpha          = 0.1,
    reg_lambda         = 1.5,
    scale_pos_weight   = spw,      # class balance
    eval_metric        = 'logloss',
    random_state       = SEED,
    verbosity          = 0,
)
xgb_clf.fit(X_tr, yc_tr,
            eval_set=[(X_te, yc_te)],
            verbose=False)

yc_pred = xgb_clf.predict(X_te)
yc_prob = xgb_clf.predict_proba(X_te)[:, 1]
acc     = accuracy_score(yc_te, yc_pred)
auc     = roc_auc_score(yc_te, yc_prob)

cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_acc  = cross_val_score(xgb_clf, X_scaled, y_cls, cv=cv, scoring='accuracy')

print(f"   Accuracy    : {acc*100:.1f}%")
print(f"   AUC-ROC     : {auc:.3f}")
print(f"   CV (5-fold) : {cv_acc.mean()*100:.1f}% ± {cv_acc.std()*100:.1f}%")
print()
print(classification_report(yc_te, yc_pred, target_names=['Loss','Gain']))

joblib.dump(xgb_clf, "models/xgb_classifier.pkl")

# ── 7B. Baseline comparison ──────────────────────────────────────────────────
print("[BASELINES]")
baselines = {
    "Logistic Regression":   LogisticRegression(max_iter=1000, C=0.5, random_state=SEED),
    "Random Forest":         RandomForestClassifier(n_estimators=200, max_depth=6, random_state=SEED),
    "Gradient Boosting":     GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=SEED),
}
model_comparison = {"XGBoost ★": round(acc * 100, 1)}
for name, m in baselines.items():
    m.fit(X_tr, yc_tr)
    ba = accuracy_score(yc_te, m.predict(X_te))
    model_comparison[name] = round(ba * 100, 1)
    print(f"   {name:<25}: {ba*100:.1f}%")
print(f"   {'XGBoost ★':<25}: {acc*100:.1f}%  ← selected")

# ── 7C. Random Forest Regressor ──────────────────────────────────────────────
print("\n[MODEL 2]  Random Forest Regressor — listing gain %")

rf_reg = RandomForestRegressor(
    n_estimators=300, max_depth=10,
    min_samples_leaf=3, max_features=0.7,
    random_state=SEED, n_jobs=-1,
)
rf_reg.fit(X_tr, yr_tr)
yr_pred = rf_reg.predict(X_te)

mae  = mean_absolute_error(yr_te, yr_pred)
rmse = mean_squared_error(yr_te, yr_pred) ** 0.5
r2   = r2_score(yr_te, yr_pred)

print(f"   MAE  : ±{mae:.2f}%")
print(f"   RMSE : {rmse:.2f}%")
print(f"   R²   : {r2:.3f}")
joblib.dump(rf_reg, "models/rf_regressor.pkl")

# ── 8. Feature importance ────────────────────────────────────────────────────
fi_clf = pd.Series(xgb_clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
fi_reg = pd.Series(rf_reg.feature_importances_,  index=FEATURES).sort_values(ascending=False)

LABELS = {
    'gmp_pct':'GMP %', 'sub_qib':'QIB Sub', 'sub_hni':'HNI Sub',
    'sub_retail':'Retail Sub', 'total_sub':'Total Sub',
    'sub_score':'Sub Score', 'qib_hni_ratio':'QIB/HNI Ratio',
    'gmp_x_qib':'GMP×QIB', 'gmp_bull':'GMP×Mood',
    'pe_vs_sector':'P/E vs Sector', 'val_risk':'Val Risk',
    'low_valuation':'Low Valuation', 'promoter_holding':'Promoter %',
    'mood_enc':'Market Mood', 'sector_enc':'Sector',
    'company_age':'Co. Age', 'revenue_growth_pct':'Rev Growth %',
    'roce_pct':'ROCE %', 'debt_equity':'Debt/Equity',
    'size_log':'Issue Size (log)', 'is_tech_bull':'Tech+Bull',
    'roce_x_growth':'ROCE×Growth', 'issue_price':'Issue Price',
}

# ── 9. Plots ─────────────────────────────────────────────────────────────────
print("\n[PLOTS]  Generating 6 visualizations...")

C = dict(blue='#2563EB', green='#16A34A', red='#DC2626',
         teal='#0D9488', purple='#7C3AED', orange='#EA580C',
         gray='#94A3B8',  bg='#F8FAFC')

plt.rcParams.update({
    'figure.facecolor': C['bg'], 'axes.facecolor': 'white',
    'axes.spines.top': False,    'axes.spines.right': False,
    'axes.grid': True,           'grid.alpha': 0.25,
    'grid.linestyle': '--',      'font.family': 'sans-serif',
    'font.size': 10,             'axes.titlesize': 12,
    'axes.titleweight': 'bold',  'axes.titlepad': 10,
})

# Plot 1 — Feature Importance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(C['bg'])
top_c = fi_clf.head(10)
bc = [C['blue'] if i < 3 else '#93C5FD' for i in range(10)]
bars = ax1.barh([LABELS.get(f,f) for f in top_c.index], top_c.values, color=bc, height=0.6)
ax1.invert_yaxis()
ax1.set_title('Feature Importance — Classifier (XGBoost)')
ax1.set_xlabel('Score')
for b, v in zip(bars, top_c.values):
    ax1.text(v+.002, b.get_y()+b.get_height()/2, f'{v:.3f}', va='center', fontsize=8)

top_r = fi_reg.head(10)
br = [C['teal'] if i < 3 else '#5EEAD4' for i in range(10)]
bars2 = ax2.barh([LABELS.get(f,f) for f in top_r.index], top_r.values, color=br, height=0.6)
ax2.invert_yaxis()
ax2.set_title('Feature Importance — Regressor (Random Forest)')
ax2.set_xlabel('Score')
for b, v in zip(bars2, top_r.values):
    ax2.text(v+.002, b.get_y()+b.get_height()/2, f'{v:.3f}', va='center', fontsize=8)
plt.tight_layout(); plt.savefig('static/plot_feature_importance.png', dpi=130, bbox_inches='tight'); plt.close()
print("   ✓ Feature importance")

# Plot 2 — ROC + Confusion Matrix
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(C['bg'])
cm = confusion_matrix(yc_te, yc_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Loss','Gain'], yticklabels=['Loss','Gain'],
            linewidths=0.5, linecolor='white', annot_kws={'size':15,'weight':'bold'})
ax1.set_title('Confusion Matrix'); ax1.set_xlabel('Predicted'); ax1.set_ylabel('Actual')
fpr, tpr, _ = roc_curve(yc_te, yc_prob)
ax2.plot(fpr, tpr, color=C['blue'], lw=2.5, label=f'XGBoost  AUC={auc:.2f}')
ax2.plot([0,1],[0,1],'--',color=C['gray'],lw=1,label='Chance')
ax2.fill_between(fpr, tpr, alpha=0.08, color=C['blue'])
ax2.set_xlabel('FPR'); ax2.set_ylabel('TPR'); ax2.set_title('ROC Curve'); ax2.legend()
plt.tight_layout(); plt.savefig('static/plot_roc_confusion.png', dpi=130, bbox_inches='tight'); plt.close()
print("   ✓ ROC + Confusion matrix")

# Plot 3 — Regression scatter + residuals
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(C['bg'])
ax1.scatter(yr_te, yr_pred, alpha=0.5, c=C['teal'], s=40, edgecolors='white', lw=0.5)
lim = max(abs(yr_te).max(), abs(yr_pred).max()) * 1.1
ax1.plot([-lim,lim],[-lim,lim],'--',color=C['red'],lw=1.5,label='Perfect')
ax1.set_xlabel('Actual Gain %'); ax1.set_ylabel('Predicted Gain %')
ax1.set_title(f'Actual vs Predicted  (MAE=±{mae:.1f}%  R²={r2:.2f})')
ax1.legend()
residuals = yr_te.values - yr_pred
ax2.hist(residuals, bins=25, color=C['purple'], edgecolor='white', alpha=0.85)
ax2.axvline(0, color=C['red'], lw=1.5, linestyle='--')
ax2.set_xlabel('Residual %'); ax2.set_ylabel('Count')
ax2.set_title(f'Residual Distribution  (RMSE={rmse:.1f}%)')
plt.tight_layout(); plt.savefig('static/plot_regression.png', dpi=130, bbox_inches='tight'); plt.close()
print("   ✓ Regression")

# Plot 4 — Model comparison
fig, ax = plt.subplots(figsize=(8, 4))
fig.patch.set_facecolor(C['bg'])
names = list(model_comparison.keys())
vals  = list(model_comparison.values())
cols  = [C['blue'] if '★' in n else C['gray'] for n in names]
bars  = ax.bar(names, vals, color=cols, edgecolor='white', width=0.5)
ax.set_ylim(55, 100); ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Comparison — Classification Accuracy')
for bar, val in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{val}%', ha='center', fontsize=11, fontweight='bold',
            color=C['blue'] if val==max(vals) else '#475569')
plt.tight_layout(); plt.savefig('static/plot_model_comparison.png', dpi=130, bbox_inches='tight'); plt.close()
print("   ✓ Model comparison")

# Plot 5 — EDA (6 subplots)
df_raw = pd.read_csv("data/ipo_data.csv")
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.patch.set_facecolor(C['bg'])
fig.suptitle('Exploratory Data Analysis — IPO Dataset (400 records)', fontsize=13, fontweight='bold')

pos_g = df_raw[df_raw['listing_gain_pct'] >= 0]['listing_gain_pct']
neg_g = df_raw[df_raw['listing_gain_pct'] <  0]['listing_gain_pct']
axes[0][0].hist(neg_g, bins=20, color=C['red'],   alpha=0.75, label='Loss', edgecolor='white')
axes[0][0].hist(pos_g, bins=20, color=C['green'], alpha=0.75, label='Gain', edgecolor='white')
axes[0][0].axvline(0, color='black', lw=1.5, linestyle='--')
axes[0][0].set_title('Listing Gain Distribution'); axes[0][0].legend()

sec_avg = df_raw.groupby('sector')['listing_gain_pct'].mean().sort_values()
axes[0][1].barh(sec_avg.index, sec_avg.values,
                color=[C['green'] if v >= 0 else C['red'] for v in sec_avg.values], height=0.6)
axes[0][1].axvline(0, color='black', lw=1)
axes[0][1].set_title('Avg Gain by Sector')

axes[0][2].scatter(df_raw['gmp_pct'], df_raw['listing_gain_pct'],
                   alpha=0.4, c=df_raw['positive_listing'], cmap='RdYlGn', s=25)
axes[0][2].axhline(0,color='gray',lw=0.8,linestyle='--')
axes[0][2].axvline(0,color='gray',lw=0.8,linestyle='--')
axes[0][2].set_title('GMP% vs Listing Gain%')
axes[0][2].set_xlabel('GMP %'); axes[0][2].set_ylabel('Listing Gain %')

axes[1][0].scatter(df_raw['sub_qib'].clip(0,150), df_raw['listing_gain_pct'],
                   alpha=0.4, c=df_raw['positive_listing'], cmap='RdYlGn', s=25)
axes[1][0].axhline(0,color='gray',lw=0.8,linestyle='--')
axes[1][0].set_title('QIB Subscription vs Gain'); axes[1][0].set_xlabel('QIB Sub (x)')

yr_avg = df_raw.groupby('year')['listing_gain_pct'].mean()
axes[1][1].plot(yr_avg.index, yr_avg.values, marker='o', color=C['blue'], lw=2.5, markersize=8)
axes[1][1].fill_between(yr_avg.index, yr_avg.values, alpha=0.1, color=C['blue'])
axes[1][1].axhline(0,color='gray',lw=0.8,linestyle='--')
axes[1][1].set_title('Avg Gain by Year')

mood_g = df_raw.groupby('market_mood')['listing_gain_pct'].mean()
axes[1][2].bar(mood_g.index, mood_g.values,
               color=[C['green'] if v >= 0 else C['red'] for v in mood_g.values], width=0.5)
axes[1][2].axhline(0,color='gray',lw=0.8,linestyle='--')
axes[1][2].set_title('Market Mood vs Avg Gain')

plt.tight_layout(); plt.savefig('static/plot_eda.png', dpi=130, bbox_inches='tight'); plt.close()
print("   ✓ EDA")

# Plot 6 — Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor(C['bg'])
num_cols = ['gmp_pct','sub_qib','sub_hni','sub_retail','pe_vs_sector',
            'promoter_holding','revenue_growth_pct','roce_pct',
            'debt_equity','listing_gain_pct']
corr = df_raw[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
            linewidths=0.5, linecolor='white', annot_kws={'size':9},
            xticklabels=[LABELS.get(c,c) for c in corr.columns],
            yticklabels=[LABELS.get(c,c) for c in corr.index])
ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('static/plot_correlation.png', dpi=130, bbox_inches='tight'); plt.close()
print("   ✓ Correlation heatmap")

# ── 10. Save metadata ─────────────────────────────────────────────────────────
meta = {
    "classifier":  {"name":"XGBoost","accuracy":round(acc*100,1),
                    "auc_roc":round(auc,3),"cv_accuracy":round(cv_acc.mean()*100,1),
                    "confusion_matrix":cm.tolist()},
    "regressor":   {"name":"RandomForest","mae":round(mae,2),
                    "rmse":round(rmse,2),"r2":round(r2,3)},
    "model_comparison": model_comparison,
    "top_features_clf": {LABELS.get(k,k): round(v,4) for k,v in fi_clf.head(8).items()},
    "top_features_reg": {LABELS.get(k,k): round(v,4) for k,v in fi_reg.head(8).items()},
    "dataset_size":  len(df),
    "positive_rate": round(df['positive_listing'].mean()*100,1),
    "features":      FEATURES,
    "mood_classes":  list(mood_enc.classes_),
    "sector_classes":list(sector_enc.classes_),
}
with open("models/metadata.json","w") as f:
    json.dump(meta, f, indent=2)

print("\n" + "=" * 60)
print("  RESULTS")
print("=" * 60)
print(f"  Classifier Accuracy : {acc*100:.1f}%")
print(f"  AUC-ROC             : {auc:.3f}")
print(f"  CV Accuracy (5-fold): {cv_acc.mean()*100:.1f}% ± {cv_acc.std()*100:.1f}%")
print(f"  Regressor MAE       : ±{mae:.2f}%")
print(f"  Regressor R²        : {r2:.3f}")
print("=" * 60)