import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# --- Load and clean dataset ---
df = pd.read_csv("/Users/kamisama/Desktop/intro/Life Expectancy Data.csv")
df = df[df["Year"] == 2014].copy()
df = df.rename(columns={" thinness  1-19 years": "Thinness (1-19)", " HIV/AIDS": "HIV/AIDS"})
df = df.dropna(subset=["Schooling", "Income composition of resources", "Thinness (1-19)", "HIV/AIDS", "Life expectancy "])
df["Score_Edu_HIV"] = df["Schooling"] / df["HIV/AIDS"]

# --- Define predictors and target ---
predictor_list = ["Schooling", "Income composition of resources", "Thinness (1-19)", "Score_Edu_HIV"]
y = df["Life expectancy "]

# --- Define regression models ---
regression_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "KNN": KNeighborsRegressor(),  # Will tune k later
    "Neural Network": MLPRegressor(max_iter=1000, random_state=42)
}

# --- Store model evaluation results ---
combo_model_results = []

for r in [1, 2, 3, 4]:
    for combo in combinations(predictor_list, r):
        X_subset = df[list(combo)]
        X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=42)

        for model_name, model in regression_models.items():
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model)
            ])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            combo_model_results.append({
                "Model": model_name,
                "Features": combo,
                "R2 Score": r2,
                "MSE": mse
            })

# --- Create and display results ---
results_df = pd.DataFrame(combo_model_results).sort_values(by="R2 Score", ascending=False).reset_index(drop=True)

# Show full output in terminal
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)

print("Top performing model/feature combos (before tuning):\n")
print(results_df.head(10))

# --- GridSearchCV for best KNN k ---
top_combo = results_df.loc[0]
best_features = list(top_combo["Features"])
print(f"\nSelected for GridSearchCV: {top_combo['Model']} with features {best_features}")

if top_combo["Model"] == "KNN":
    X_best = df[best_features]
    X_train, X_test, y_train, y_test = train_test_split(X_best, y, test_size=0.2, random_state=42)

    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor())
    ])
    param_grid = {"model__n_neighbors": list(range(1, 21))}
    grid_search = GridSearchCV(knn_pipeline, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    best_k = grid_search.best_params_["model__n_neighbors"]
    best_cv_r2 = grid_search.best_score_

    print(f"\n🔍 GridSearchCV Result for KNN:")
    print(f"Best k: {best_k}")
    print(f"Cross-validated R²: {best_cv_r2:.4f}")

# As a result, we know that the best model is KNN and k = 4 !