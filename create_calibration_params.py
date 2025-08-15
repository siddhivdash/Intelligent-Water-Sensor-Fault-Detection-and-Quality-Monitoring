import pandas as pd
import joblib
from pathlib import Path

# 1. Load your train.csv
df = pd.read_csv(Path("artifacts")/"train.csv")

# 2. Define target water-property ranges per sensor
target_ranges = {
    f"Sensor-{i}": (0.0, 14.0)   if i==1 else  # pH on Sensor-1
                  (0.0, 100.0)  if i==2 else  # Turbidity on Sensor-2
                  (0.0, 2000.0) if i==3 else  # Conductivity on Sensor-3
                  (0.0, 50.0)   if i==4 else
                  (0.0, 10.0)   if i==5 else
                  (0.0, 50.0)   if i==6 else
                  (0.0, 14.0)   if i==7 else
                  (0.0, 500.0)  if i==8 else
                  (0.0, 200.0)  if i==9 else
                  (0.0, 10.0)   for i in range(1,11)
}

# 3. Compute observed min/max per channel
calibration_params = {}
for ch,(ymin,ymax) in target_ranges.items():
    xmin, xmax = df[ch].min(), df[ch].max()
    calibration_params[ch] = {"xmin": xmin,"xmax": xmax,"ymin": ymin,"ymax": ymax}

# 4. Save to artifacts
joblib.dump(calibration_params, "artifacts/calibration_params.pkl")
print("Saved calibration_params.pkl")
