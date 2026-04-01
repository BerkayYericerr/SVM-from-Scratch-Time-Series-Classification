# Main

import numpy as np
import warnings
warnings.filterwarnings('ignore')

np.random.seed(409)

print("SVM Project")
print("="*40)

# load data
from data_preparation import prepare_energy_dataset, prepare_air_pollution_dataset

energy = prepare_energy_dataset('Hourly_Energy _Consumption_AEP_hourly.csv')
air = prepare_air_pollution_dataset('Air_Pollution_in_Seoul_Measurement_summary.csv')

print(f"Energy samples: {len(energy['y'])}")
print(f"Air samples: {len(air['y'])}")

# eda
from eda import run_eda
run_eda(energy)
run_eda(air)

# training
from training import train_and_test

energy_res = train_and_test(energy['X'], energy['y'], "Energy")
air_res = train_and_test(air['X'], air['y'], "Air Pollution")

# results
print("\n" + "="*40)
print("Summary (TEST metrics on 20% holdout)")
print("="*40)

def print_metrics(name, res):
    print(f"\n{name}:")
    for k in ["linear", "rbf", "poly", "custom"]:
        if k in res and "f1" in res[k]:
            m = res[k]
            print(f"  {k:6s} | Acc={m['acc']:.4f}  Prec={m['prec']:.4f}  Rec={m['rec']:.4f}  F1={m['f1']:.4f}")

print_metrics("Energy", energy_res)
print_metrics("Air Pollution", air_res)
