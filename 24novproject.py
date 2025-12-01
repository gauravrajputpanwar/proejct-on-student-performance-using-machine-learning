
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


# --------------------------------------------------------------
# 1) GENERATE SYNTHETIC DATASET OF 1 LAKH ROWS
# --------------------------------------------------------------
np.random.seed(42)

rows = 100000  # 1 Lakh

schools  = np.random.choice(['GP', 'MS'], rows)
sex      = np.random.choice(['M', 'F'], rows)
age      = np.random.randint(15, 22, rows)
address  = np.random.choice(['U', 'R'], rows)
study    = np.random.randint(1, 5, rows)
fail     = np.random.randint(0, 4, rows)
absent   = np.random.randint(0, 30, rows)

g1 = np.random.randint(0, 20, rows)
g2 = g1 + np.random.randint(-2, 3, rows)
g3 = g2 + np.random.randint(-2, 3, rows)

# final grade = G3
final_grade = g3

df = pd.DataFrame({
    "School": schools,
    "Sex": sex,
    "Age": age,
    "Address": address,
    "Studytime": study,
    "Failures": fail,
    "Absences": absent,
    "G1": g1,
    "G2": g2,
    "G3": g3,
    "FinalGrade": final_grade
})

print("\nSample of Generated Dataset:\n")
print(df.head())


# --------------------------------------------------------------
# 2) PREPROCESSING
# --------------------------------------------------------------
df = df.fillna(df.mean(numeric_only=True))
df = df.fillna(df.mode().iloc[0])

# Label Encoding object columns
label_cols = ["School", "Sex", "Address"]
encoder = LabelEncoder()

for col in label_cols:
    df[col] = encoder.fit_transform(df[col])


# --------------------------------------------------------------
# 3) TRAIN–TEST SPLIT
# --------------------------------------------------------------
X = df.drop("FinalGrade", axis=1)
y = df["FinalGrade"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# --------------------------------------------------------------
# 4) MODEL TRAINING
# --------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=7)
}

results = {}

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


for name, model in models.items():
    print(f"\nTraining: {name}")

    if name == "KNN":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    results[name] = [mae, mse, rmse, r2]

    print(f"{name} Results:")
    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R2  :", r2)


# --------------------------------------------------------------
# 5) MODEL COMPARISON
# --------------------------------------------------------------
results_df = pd.DataFrame(results, index=["MAE", "MSE", "RMSE", "R2 Score"])
print("\nMODEL COMPARISON:\n")
print(results_df)


# --------------------------------------------------------------
# 6) BEST MODEL SELECTION
# --------------------------------------------------------------
best_model_name = max(results, key=lambda x: results[x][3])
print(f"\nBest Model = {best_model_name}")

best_model = models[best_model_name]

if best_model_name == "KNN":
    best_model.fit(X_train_scaled, y_train)
    y_pred_best = best_model.predict(X_test_scaled)
else:
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)


# --------------------------------------------------------------
# 7) SCATTER PLOT – TRUE VS PREDICTED
# --------------------------------------------------------------
plt.scatter(y_test, y_pred_best)
plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.title(f"True vs Predicted - {best_model_name}")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()


# --------------------------------------------------------------
# 8) CORRELATION HEATMAP
# --------------------------------------------------------------
corr = df.corr()

plt.figure(figsize=(10, 6))
plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap (Matplotlib)")
plt.show()
