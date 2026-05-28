"""
generate_dataset.py
-------------------
Generates a synthetic dataset simulating SME financial data.
We create realistic-looking data with three risk classes:
  - Low Risk (0)
  - Medium Risk (1)
  - High Risk (2)

The data is saved as dataset.csv in the ml/ directory.
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure output goes to ml/ regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Reproducibility
np.random.seed(42)

N = 1500  # total samples (500 per class)

# ── Helper: clamp values to realistic ranges ────────────────────────────────
def clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


# ── LOW RISK businesses ──────────────────────────────────────────────────────
n_low = 500
low = pd.DataFrame({
    "monthly_revenue":    clip(np.random.normal(120_000, 20_000, n_low), 50_000, 300_000),
    "pending_invoices":   clip(np.random.normal(10_000,  5_000,  n_low), 0,      50_000),
    "avg_payment_delay":  clip(np.random.normal(5,       3,      n_low), 0,      15),    # days
    "monthly_expenses":   clip(np.random.normal(60_000,  10_000, n_low), 20_000, 150_000),
    "payroll_ratio":      clip(np.random.normal(0.25,    0.05,   n_low), 0.05,   0.45),  # fraction of revenue
    "cash_reserve":       clip(np.random.normal(80_000,  15_000, n_low), 20_000, 200_000),
    "vendor_due_amount":  clip(np.random.normal(8_000,   3_000,  n_low), 0,      30_000),
    "cash_flow_risk": "Low",
})

# ── MEDIUM RISK businesses ───────────────────────────────────────────────────
n_med = 500
med = pd.DataFrame({
    "monthly_revenue":    clip(np.random.normal(70_000, 15_000, n_med), 20_000, 200_000),
    "pending_invoices":   clip(np.random.normal(25_000, 8_000,  n_med), 5_000,  80_000),
    "avg_payment_delay":  clip(np.random.normal(18,     5,      n_med), 8,      35),
    "monthly_expenses":   clip(np.random.normal(55_000, 12_000, n_med), 15_000, 130_000),
    "payroll_ratio":      clip(np.random.normal(0.42,   0.08,   n_med), 0.25,   0.65),
    "cash_reserve":       clip(np.random.normal(25_000, 8_000,  n_med), 3_000,  70_000),
    "vendor_due_amount":  clip(np.random.normal(20_000, 6_000,  n_med), 5_000,  60_000),
    "cash_flow_risk": "Medium",
})

# ── HIGH RISK businesses ─────────────────────────────────────────────────────
n_high = 500
high = pd.DataFrame({
    "monthly_revenue":    clip(np.random.normal(35_000, 10_000, n_high), 5_000,  100_000),
    "pending_invoices":   clip(np.random.normal(50_000, 12_000, n_high), 20_000, 120_000),
    "avg_payment_delay":  clip(np.random.normal(40,     8,      n_high), 25,     70),
    "monthly_expenses":   clip(np.random.normal(42_000, 10_000, n_high), 10_000, 90_000),
    "payroll_ratio":      clip(np.random.normal(0.65,   0.10,   n_high), 0.40,   0.95),
    "cash_reserve":       clip(np.random.normal(5_000,  3_000,  n_high), 0,      20_000),
    "vendor_due_amount":  clip(np.random.normal(45_000, 10_000, n_high), 15_000, 100_000),
    "cash_flow_risk": "High",
})

# ── Combine and shuffle ──────────────────────────────────────────────────────
df = pd.concat([low, med, high], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Round numeric columns for readability
numeric_cols = [c for c in df.columns if c != "cash_flow_risk"]
df[numeric_cols] = df[numeric_cols].round(2)

# Save
output_path = os.path.join(SCRIPT_DIR, "dataset.csv")
df.to_csv(output_path, index=False)
print(f"[OK] Dataset saved -> {output_path}  ({len(df)} rows)")
print(df["cash_flow_risk"].value_counts())
