"""
IPO Data Generator — Rule-Based Synthesis
==========================================
Drops CTGAN entirely. Generates synthetic IPO rows using real GMP→label
probabilities derived from actual data analysis:

  GMP < −10%   → 0.0% positive listings  (always a loss)
  GMP −10→0%  → 23%  positive listings
  GMP   0→10% → 85%  positive listings
  GMP  10→25% → 100% positive listings   (always a gain)
  GMP   > 25% → 100% positive listings   (large gain)

Combines: 490 real rows + 2510 rule-based synthetic = 3,000 total.
"""

import pandas as pd
import numpy as np
import os

SEED = 42
rng  = np.random.default_rng(SEED)
N_SYNTH = 2510

print("=" * 60)
print("  IPO DATA GENERATOR — RULE-BASED SYNTHESIS")
print("=" * 60)

# ── Load & clean real data ────────────────────────────────────────────────────
df_real = pd.read_csv("data/ipo_data.csv")
print(f"\nLoaded real dataset  : {len(df_real)} rows")

def clean_numeric(series):
    return (series.astype(str)
            .str.replace(',',    '', regex=False)
            .str.replace('₹',   '', regex=False)
            .str.replace('%',   '', regex=False)
            .str.replace('<br/>','', regex=False)
            .str.strip()
            .replace(['N/A','NA','--','-','nan','None',''], np.nan)
            .astype(float))

NUM_COLS = ['issue_price','issue_size_cr','sub_qib','sub_hni','sub_retail',
            'gmp_pct','pe_vs_sector','promoter_holding','revenue_growth_pct',
            'roce_pct','debt_equity','company_age','listing_gain_pct']

for col in NUM_COLS:
    if col in df_real.columns:
        df_real[col] = clean_numeric(df_real[col])

df_real['positive_listing'] = pd.to_numeric(df_real['positive_listing'], errors='coerce')

# ── Compute per-feature statistics from real data ─────────────────────────────
stats = {}
for col in NUM_COLS:
    if col in df_real.columns:
        s = df_real[col].dropna()
        stats[col] = {
            'mean': s.mean(), 'std': s.std(),
            'p02':  s.quantile(0.02), 'p98': s.quantile(0.98),
        }

sectors_dist = df_real['sector'].value_counts(normalize=True)

print(f"  GMP distribution : mean={stats['gmp_pct']['mean']:.1f}%  "
      f"std={stats['gmp_pct']['std']:.1f}%")
print(f"  Positive rate    : {df_real['positive_listing'].mean()*100:.1f}%")

# ── Helper ────────────────────────────────────────────────────────────────────
def sample(mean, std, lo, hi, size):
    return np.clip(rng.normal(mean, std, size), lo, hi)

# ── Step 1: GMP — same distribution as real data ─────────────────────────────
gmp = sample(stats['gmp_pct']['mean'], stats['gmp_pct']['std'],
             stats['gmp_pct']['p02'],  stats['gmp_pct']['p98'], N_SYNTH)

# ── Step 2: Listing gain derived from GMP bucket (real data probabilities) ────
# Label is then derived from gain — ensures perfect consistency.
gains = np.zeros(N_SYNTH)

for i, g in enumerate(gmp):
    if g < -10:
        # 0% positive → always a loss
        gains[i] = rng.uniform(-35, -1)

    elif g < 0:
        # 23% positive
        if rng.random() < 0.77:            # loss
            gains[i] = float(np.clip(rng.normal(-9, 5),   -40, -0.1))
        else:                              # gain
            gains[i] = float(np.clip(rng.normal(4,  3),    0.1,  15))

    elif g < 10:
        # 85% positive
        if rng.random() < 0.15:           # loss
            gains[i] = float(np.clip(rng.normal(-4, 3),   -20, -0.1))
        else:                              # gain
            gains[i] = float(np.clip(rng.normal(11, 7),    0.1,  45))

    elif g < 25:
        # 100% positive — moderate/large gain
        gains[i] = float(np.clip(rng.normal(28, 12),       0.1,  90))

    else:
        # 100% positive — large gain
        gains[i] = float(np.clip(rng.normal(58, 22),       0.1, 160))

labels = (gains > 0).astype(int)   # derived → always consistent

# ── Step 3: Market mood correlated with GMP ───────────────────────────────────
#   Bull market drives high GMP; bear market drags it down.
moods = []
for g in gmp:
    if   g > 20: p = [0.70, 0.25, 0.05]
    elif g > 5:  p = [0.45, 0.40, 0.15]
    elif g > -5: p = [0.25, 0.40, 0.35]
    else:        p = [0.10, 0.30, 0.60]
    moods.append(rng.choice(['bull','neutral','bear'], p=p))

# ── Step 4: Subscription tiers correlated with GMP ───────────────────────────
#   Institutions subscribe heavily to high-GMP IPOs.
sub_qib = np.where(
    gmp > 20, sample(90, 50, 10, 300, N_SYNTH),
    np.where(gmp > 5, sample(30, 22,  2, 120, N_SYNTH),
                      sample( 7,  8,  0.5, 40, N_SYNTH))
)
sub_hni    = np.clip(sub_qib * rng.uniform(0.25, 0.75, N_SYNTH),  0.5, 220)
sub_retail = np.clip(sub_qib * rng.uniform(0.10, 0.35, N_SYNTH),  0.5,  85)

# ── Step 5: Remaining features from real data distributions ──────────────────
sec_list = list(sectors_dist.index)
sec_prob = list(sectors_dist.values)
sectors  = rng.choice(sec_list, size=N_SYNTH, p=sec_prob)
years    = rng.choice(list(range(2016, 2027)), size=N_SYNTH)

def rs(col): return sample(stats[col]['mean'], stats[col]['std'],
                           stats[col]['p02'],  stats[col]['p98'],  N_SYNTH)

issue_price  = rs('issue_price')
issue_size   = rs('issue_size_cr')
pe           = rs('pe_vs_sector')
promoter     = np.clip(rs('promoter_holding'),   0, 100)
rev_growth   = rs('revenue_growth_pct')
roce         = rs('roce_pct')
debt_eq      = np.clip(rs('debt_equity'),         0, None)
co_age       = np.clip(rs('company_age'),         1, None)

# ── Build synthetic DataFrame ─────────────────────────────────────────────────
df_synth = pd.DataFrame({
    'company':                      [f'IPO_Synth_{i+1:04d}' for i in range(N_SYNTH)],
    'Opening Date':                 pd.NaT,
    'issue_price':                  issue_price,
    'issue_size_cr':                issue_size,
    'sub_qib':                      sub_qib,
    'sub_hni':                      sub_hni,
    'sub_retail':                   sub_retail,
    'Total (x)':                    sub_qib + sub_hni + sub_retail,
    'Listing Date':                 pd.NaT,
    'Open Price on Listing (Rs.)':  np.nan,
    'Close Price on Listing (Rs.)': np.nan,
    'listing_gain_pct':             gains,
    'year':                         years,
    'positive_listing':             labels,
    'gmp_pct':                      gmp,
    'sector':                       sectors,
    'market_mood':                  moods,
    'pe_vs_sector':                 pe,
    'promoter_holding':             promoter,
    'revenue_growth_pct':           rev_growth,
    'roce_pct':                     roce,
    'debt_equity':                  debt_eq,
    'company_age':                  co_age,
})

# ── Combine & save ────────────────────────────────────────────────────────────
df_combined = pd.concat([df_real, df_synth], ignore_index=True)

print(f"\n{'─'*45}")
print(f"  Real rows        : {len(df_real)}")
print(f"  Synthetic rows   : {len(df_synth)}")
print(f"  Total rows       : {len(df_combined)}")
print(f"  Positive rate    : {df_combined['positive_listing'].mean()*100:.1f}%")
print(f"  Avg GMP          : {df_combined['gmp_pct'].mean():.1f}%")
print(f"  Avg listing gain : {df_combined['listing_gain_pct'].mean():.1f}%")
print(f"{'─'*45}")

os.makedirs("data", exist_ok=True)
df_combined.to_csv("data/ipo_data_synthetic.csv", index=False)
print(f"\n✅  Saved {len(df_combined)} rows → data/ipo_data_synthetic.csv")
