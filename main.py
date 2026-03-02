import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import pickle

# ────────────────────────────────────────────────
# 1. Load and prepare data
# ────────────────────────────────────────────────

# Use raw string or forward slashes to avoid path issues
df = pd.read_csv(r'C:\Users\ryans\Downloads\city_day_file.csv\city_day.csv', parse_dates=['Date'])

# Fix column name inconsistency (your file has space, code expects underscore)
df = df.rename(columns={'AQI Bucket': 'AQI_Bucket'})

print("Columns after loading and rename:", df.columns.tolist())

# Convert Date and set as index
df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
df.set_index('Date', inplace=True)

# ────────────────────────────────────────────────
# 2. Handle missing values - fill with column means
# ────────────────────────────────────────────────

pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3',
              'Benzene', 'Toluene', 'Xylene', 'AQI']

for col in pollutants:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())

# Extra safety: re-fill just in case
df[pollutants] = df[pollutants].fillna(df[pollutants].mean())

print("\nRemaining NaNs after filling:\n", df[pollutants].isna().sum())


# ────────────────────────────────────────────────
# 3. Define Sub-Index functions (corrected & clean)
# ────────────────────────────────────────────────

def get_PM10_subindex(x):
    if x <= 50:
        return x
    elif x <= 100:
        return x
    elif x <= 250:
        return 100 + (x - 100) * 100 / 150
    elif x <= 350:
        return 200 + (x - 250)
    elif x <= 430:
        return 300 + (x - 350) * 100 / 80
    elif x > 430:
        return 400 + (x - 430) * 100 / 80
    return 0


def get_PM25_subindex(x):
    if x <= 30:
        return x * 50 / 30
    elif x <= 60:
        return 50 + (x - 30) * 50 / 30
    elif x <= 90:
        return 100 + (x - 60) * 100 / 30
    elif x <= 120:
        return 200 + (x - 90) * 100 / 30
    elif x <= 250:
        return 300 + (x - 120) * 100 / 130
    elif x > 250:
        return 400 + (x - 250) * 100 / 130
    return 0


def get_SO2_subindex(x):
    if x <= 40:
        return x * 50 / 40
    elif x <= 80:
        return 50 + (x - 40) * 50 / 40
    elif x <= 380:
        return 100 + (x - 80) * 100 / 300
    elif x <= 800:
        return 200 + (x - 380) * 100 / 420
    elif x <= 1600:
        return 300 + (x - 800) * 100 / 800
    elif x > 1600:
        return 400 + (x - 1600) * 100 / 800
    return 0


def get_NOx_subindex(x):
    if x <= 40:
        return x * 50 / 40
    elif x <= 80:
        return 50 + (x - 40) * 50 / 40
    elif x <= 180:
        return 100 + (x - 80) * 100 / 100
    elif x <= 280:
        return 200 + (x - 180) * 100 / 100
    elif x <= 400:
        return 300 + (x - 280) * 100 / 120
    elif x > 400:
        return 400 + (x - 400) * 100 / 120
    return 0


def get_NH3_subindex(x):
    if x <= 200:
        return x * 50 / 200
    elif x <= 400:
        return 50 + (x - 200) * 50 / 200
    elif x <= 800:
        return 100 + (x - 400) * 100 / 400
    elif x <= 1200:
        return 200 + (x - 800) * 100 / 400
    elif x <= 1800:
        return 300 + (x - 1200) * 100 / 600
    elif x > 1800:
        return 400 + (x - 1800) * 100 / 600
    return 0


def get_CO_subindex(x):
    if x <= 1:
        return x * 50 / 1
    elif x <= 2:
        return 50 + (x - 1) * 50 / 1
    elif x <= 10:
        return 100 + (x - 2) * 100 / 8
    elif x <= 17:
        return 200 + (x - 10) * 100 / 7
    elif x <= 34:
        return 300 + (x - 17) * 100 / 17
    elif x > 34:
        return 400 + (x - 34) * 100 / 17
    return 0


def get_O3_subindex(x):
    if x <= 50:
        return x * 50 / 50
    elif x <= 100:
        return 50 + (x - 50) * 50 / 50
    elif x <= 168:
        return 100 + (x - 100) * 100 / 68
    elif x <= 208:
        return 200 + (x - 168) * 100 / 40
    elif x <= 748:
        return 300 + (x - 208) * 100 / 539
    elif x > 748:
        return 400 + (x - 748) * 100 / 539
    return 0


# ────────────────────────────────────────────────
# 4. Calculate Sub-Indices (using nullable Int64)
# ────────────────────────────────────────────────

df['PM10_SubIndex'] = df['PM10'].round(0).astype('Int64').apply(get_PM10_subindex)
df['PM2.5_SubIndex']  = df['PM2.5'].round(0).astype('Int64').apply(get_PM25_subindex)
df['SO2_SubIndex']    = df['SO2'].round(0).astype('Int64').apply(get_SO2_subindex)
df['NOx_SubIndex']    = df['NOx'].round(0).astype('Int64').apply(get_NOx_subindex)
df['NH3_SubIndex']    = df['NH3'].round(0).astype('Int64').apply(get_NH3_subindex)
df['CO_SubIndex']     = df['CO'].round(0).astype('Int64').apply(get_CO_subindex)
df['O3_SubIndex']     = df['O3'].round(0).astype('Int64').apply(get_O3_subindex)

# Fill AQI using maximum sub-index when missing
subindex_cols = ['PM2.5_SubIndex', 'PM10_SubIndex', 'SO2_SubIndex', 'NOx_SubIndex',
                 'NH3_SubIndex', 'CO_SubIndex', 'O3_SubIndex']

df['AQI'] = df['AQI'].fillna(
    df[subindex_cols].max(axis=1).round(0).astype('Int64')
)


# ────────────────────────────────────────────────
# 5. AQI Bucket function & filling
# ────────────────────────────────────────────────

def get_AQI_bucket(x):
    if x <= 50:
        return "Good"
    elif x <= 100:
        return "Satisfactory"
    elif x <= 200:
        return "Moderate"
    elif x <= 300:
        return "Poor"
    elif x <= 400:
        return "Very Poor"
    else:
        return "Severe"


# Fill missing AQI_Bucket based on calculated AQI
df['AQI_Bucket'] = df['AQI_Bucket'].fillna(
    df['AQI'].apply(get_AQI_bucket)
)

print("\nSample after bucket calculation:")
print(df[['AQI', 'AQI_Bucket']].tail(8))

# ────────────────────────────────────────────────
# 6. Visualizations
# ────────────────────────────────────────────────

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', annot=True, fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# Average AQI by city
plt.figure(figsize=(10, 6))
df.groupby('City')['AQI'].mean().sort_values().plot(kind='bar', color='skyblue')
plt.title("Average AQI by City")
plt.ylabel("AQI")
plt.show()

# ────────────────────────────────────────────────
# 7. Prepare data for classification
# ────────────────────────────────────────────────

mapping = {
    'Good': 0,
    'Satisfactory': 1,
    'Moderate': 2,
    'Poor': 3,
    'Very Poor': 4,
    'Severe': 5
}

final_df = df[['AQI', 'AQI_Bucket']].copy()
final_df['AQI_Bucket_num'] = final_df['AQI_Bucket'].map(mapping)

X = final_df[['AQI']]
y = final_df['AQI_Bucket_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ────────────────────────────────────────────────
# 8. Train & Evaluate Models
# ────────────────────────────────────────────────

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "AdaBoost": AdaBoostClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))   # optional

    # Save model
    filename = f'AQI_model_{name.replace(" ", "_")}.pkl'
    pickle.dump(model, open(filename, 'wb'))
    print(f"Saved: {filename}")

print("\nDone. You can now use the saved models for prediction.")
