"""
IPO Intelligence — Flask Web App
3 pages: Dashboard / Predict / Model Info
"""
from flask import Flask, render_template, request, jsonify
import numpy as np, joblib, json, os

BASE = os.path.dirname(os.path.abspath(__file__))
app  = Flask(__name__, static_folder=os.path.join(BASE, 'static'))

def load():
    d = os.path.join(BASE, 'models')
    return {
        "clf":        joblib.load(os.path.join(d, 'xgb_classifier.pkl')),
        "reg":        joblib.load(os.path.join(d, 'rf_regressor.pkl')),
        "imputer":    joblib.load(os.path.join(d, 'imputer.pkl')),
        "scaler":     joblib.load(os.path.join(d, 'scaler.pkl')),
        "mood_enc":   joblib.load(os.path.join(d, 'mood_encoder.pkl')),
        "sector_enc": joblib.load(os.path.join(d, 'sector_encoder.pkl')),
        "features":   joblib.load(os.path.join(d, 'feature_names.pkl')),
        "meta":       json.load(open(os.path.join(d, 'metadata.json'))),
    }
A = load()

SECTORS = ["Tech","Finance","Pharma","Infra","FMCG","Auto","Retail","Mfg"]
MOODS   = ["bull","neutral","bear"]

def build_features(raw):
    gmp_pct  = float(raw.get('gmp_pct',  0))
    sub_qib  = float(raw.get('sub_qib', 10))
    sub_hni  = float(raw.get('sub_hni',  5))
    sub_ret  = float(raw.get('sub_retail', 2))
    pe       = float(raw.get('pe_vs_sector', 1.0))
    prom     = float(raw.get('promoter_holding', 60))
    mood     = raw.get('market_mood', 'neutral')
    sector   = raw.get('sector', 'Tech')
    age      = float(raw.get('company_age', 8))
    rev_gr   = float(raw.get('revenue_growth_pct', 25))
    roce     = float(raw.get('roce_pct', 15))
    de       = float(raw.get('debt_equity', 0.5))
    size     = float(raw.get('issue_size_cr', 1000))
    price    = float(raw.get('issue_price', 500))

    me  = A['mood_enc'].transform([mood])[0]
    se  = A['sector_enc'].transform([sector])[0]

    total   = sub_qib + sub_hni + sub_ret
    ratio   = sub_qib / (sub_hni + 0.01)
    gxq     = gmp_pct * np.log1p(sub_qib)
    slog    = np.log1p(size)
    val_r   = min(pe, 5)
    is_tb   = int(sector == 'Tech' and mood == 'bull')
    rxg     = roce * rev_gr / 100
    sscore  = sub_qib * 0.5 + sub_hni * 0.3 + sub_ret * 0.2
    gbull   = gmp_pct * (me + 1)
    low_val = int(pe < 1.0)

    row = [gmp_pct, sub_qib, sub_hni, sub_ret,
           total, sscore, ratio, gxq, gbull,
           pe, val_r, low_val, prom, me, se,
           age, rev_gr, roce, de, slog, is_tb, rxg, price]

    X = np.array(row, dtype=float).reshape(1, -1)
    X = A['imputer'].transform(X)
    X = A['scaler'].transform(X)
    return X, price

def predict(raw):
    X, price = build_features(raw)
    prob   = A['clf'].predict_proba(X)[0][1]
    gain   = float(A['reg'].predict(X)[0])
    target = round(price * (1 + gain / 100), 2)

    gmp_pct = float(raw.get('gmp_pct', 0))
    risk = int(np.clip(80 - prob*38 - min(gmp_pct,30)*0.5
                           - float(raw.get('sub_qib',10))*0.08, 8, 92))

    if   gain > 20 and risk < 40 and prob > 0.70: rec, cls = "Strong Apply", "green"
    elif gain > 10 and risk < 60:                  rec, cls = "Apply",        "green"
    elif gain > 0  and risk < 68:                  rec, cls = "Cautious",     "amber"
    else:                                           rec, cls = "Avoid",        "red"

    return {
        "win_prob": round(prob*100, 1),
        "gain_pct": round(gain, 1),
        "target":   target,
        "risk":     risk,
        "rec":      rec,
        "rec_cls":  cls,
    }

@app.route('/')
def dashboard():
    plots = [
        ("EDA Overview",        "plot_eda.png"),
        ("Feature Importance",  "plot_feature_importance.png"),
        ("ROC + Confusion",     "plot_roc_confusion.png"),
        ("Regression Analysis", "plot_regression.png"),
        ("Model Comparison",    "plot_model_comparison.png"),
        ("Correlation Heatmap", "plot_correlation.png"),
    ]
    return render_template('dashboard.html', meta=A['meta'], plots=plots)

@app.route('/predict', methods=['GET','POST'])
def pred_page():
    result, form = None, {}
    if request.method == 'POST':
        form   = request.form.to_dict()
        result = predict(form)
    return render_template('predict.html', result=result, form=form,
                           sectors=SECTORS, moods=MOODS)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    return jsonify(predict(request.get_json(force=True)))

@app.route('/about')
def about():
    return render_template('about.html', meta=A['meta'])

if __name__ == '__main__':
    app.run(debug=True, port=5000)