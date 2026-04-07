import pandas as pd
import numpy as np

np.random.seed(42)

SECTORS = ["Tech", "Finance", "Pharma", "Infra", "FMCG", "Auto", "Retail", "Mfg"]

def sector_pe_avg(sector):
    return {"Tech":60,"Finance":18,"Pharma":35,"Infra":22,
            "FMCG":45,"Auto":20,"Retail":55,"Mfg":25}.get(sector, 30)

def generate_ipo(idx, year, sector, issue_price, listing_gain_target):
    rng = np.random.RandomState(idx * 7 + 13)
    is_positive = listing_gain_target > 0
    mood = rng.choice(["bull","bull","neutral"] if is_positive else ["bear","neutral","bear"])
    mood_mult = {"bull":1.4, "neutral":1.0, "bear":0.6}[mood]
    gmp_pct = np.clip(listing_gain_target * 0.6 + rng.normal(0, 5), -20, 90)
    gmp = round(issue_price * gmp_pct / 100, 0)
    base_qib = max(0.5, (listing_gain_target * 2.0 + rng.normal(15, 10)) * mood_mult)
    sub_qib = round(np.clip(base_qib, 0.3, 200), 1)
    sub_hni = round(np.clip(sub_qib * rng.uniform(0.3, 0.6), 0.2, 120), 1)
    sub_ret = round(np.clip(sub_qib * rng.uniform(0.08, 0.22), 0.3, 20), 1)
    pe_vs_sector = np.clip(rng.normal(1.0 if is_positive else 1.8, 0.3), 0.4, 3.5)
    promoter = round(np.clip(rng.normal(65 if is_positive else 45, 15), 20, 95), 1)
    issue_size = round(issue_price * rng.uniform(0.3, 8) * 1e6 / 1e7, 0) * 10
    rev_growth = round(np.clip(rng.normal(35 if is_positive else 10, 18), -15, 120), 1)
    roce = round(np.clip(rng.normal(20 if is_positive else 8, 10), -15, 55), 1)
    co_age = max(1, int(rng.normal(14, 9)))
    debt_equity = round(np.clip(rng.exponential(0.5 if is_positive else 1.2), 0, 5), 2)
    noise = rng.normal(0, 4)
    listing_gain = round(listing_gain_target + noise, 2)
    return {
        "year": year, "sector": sector,
        "issue_price": issue_price, "issue_size_cr": issue_size,
        "gmp": gmp, "gmp_pct": round(gmp_pct, 2),
        "sub_qib": sub_qib, "sub_hni": sub_hni, "sub_retail": sub_ret,
        "pe_vs_sector": round(pe_vs_sector, 2),
        "promoter_holding": promoter, "market_mood": mood,
        "company_age": co_age, "revenue_growth_pct": rev_growth,
        "roce_pct": roce, "debt_equity": debt_equity,
        "listing_gain_pct": listing_gain,
        "positive_listing": int(listing_gain > 0),
    }

REAL_IPOS = [
    ("Paytm",2021,"Tech",2150,-27.2),
    ("Zomato",2021,"Tech",76,52.7),
    ("Nykaa",2021,"Tech",1125,79.4),
    ("Latent View",2021,"Tech",197,278.0),
    ("Tata Tech",2023,"Tech",500,42.7),
    ("IREDA",2023,"Finance",32,56.3),
    ("JSW Infra",2023,"Infra",119,27.3),
    ("Bajaj Housing",2024,"Finance",70,114.0),
    ("Premier Energies",2024,"Infra",450,120.6),
    ("Honasa",2023,"FMCG",324,-11.7),
    ("Ola Electric",2024,"Tech",76,-5.3),
    ("Hyundai India",2024,"Auto",1960,-1.3),
    ("LIC",2022,"Finance",949,-8.1),
    ("Delhivery",2022,"Tech",487,-1.1),
    ("Mankind Pharma",2023,"Pharma",1080,19.8),
    ("NTPC Green",2024,"Infra",108,12.5),
]

rows = []
for i, (name, year, sector, price, gain) in enumerate(REAL_IPOS):
    row = generate_ipo(i, year, sector, price, gain)
    row["company"] = name
    rows.append(row)

for i in range(400 - len(REAL_IPOS)):
    idx = i + len(REAL_IPOS)
    year = np.random.choice(
        [2018,2019,2020,2021,2022,2023,2024],
        p=[0.04,0.06,0.10,0.22,0.15,0.22,0.21]
    )
    sector = np.random.choice(SECTORS)
    issue_price = int(np.random.choice([
        np.random.randint(20, 200),
        np.random.randint(200, 800),
        np.random.randint(800, 3000)
    ]))
    era_boost = {"2021":14,"2020":7,"2022":-5,"2023":9,"2024":8}.get(str(year), 3)
    sector_boost = {"Tech":6,"Finance":4,"Infra":7,"Pharma":3}.get(sector, 1)
    base_gain = np.random.normal(10 + era_boost + sector_boost, 20)
    row = generate_ipo(idx, year, sector, issue_price, base_gain)
    row["company"] = f"IPO_{idx:04d}"
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("data/ipo_data.csv", index=False)
print(f"Generated {len(df)} IPO records")
print(f"Positive listings: {df['positive_listing'].mean()*100:.1f}%")
print(f"Avg listing gain : {df['listing_gain_pct'].mean():.1f}%")
