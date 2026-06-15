import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from scraper import fetch_github_users
from features import generate_features


MODEL_PATH = "model.pkl"


FINAL_FEATURES = [
    "days_since_last_activity",
    "account_age_days",
    "repos_per_year",
    "follower_ratio",
    "followers_per_repo",
    "public_repos",
    "degree_centrality",
    "pagerank",
]


def train_model():
    """
    Fetch data, generate features, train model, save model.pkl.
    """

    print("Fetching GitHub data...")
    raw_df = fetch_github_users()

    print("Generating features...")
    df, feature_cols = generate_features(raw_df)

    X = df[feature_cols]
    y = df["churned"]

    print("Class balance:")
    print(y.value_counts())

    if y.nunique() < 2:
        raise ValueError(
            "Only one class found. Adjust churn threshold or fetch different users."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # Feature Selection Method 1: Filter
    # -----------------------------
    var_selector = VarianceThreshold(threshold=0.01)
    var_selector.fit(X_train)

    kbest = SelectKBest(score_func=f_classif, k=5)
    kbest.fit(X_train, y_train)

    filter_selected = X.columns[kbest.get_support()].tolist()

    # -----------------------------
    # Feature Selection Method 2: RFE
    # -----------------------------
    log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    rfe = RFE(estimator=log_model, n_features_to_select=5)
    rfe.fit(X_train, y_train)

    rfe_selected = X.columns[rfe.support_].tolist()

    # -----------------------------
    # Feature Selection Method 3: Decision Tree
    # -----------------------------
    dt = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced")
    dt.fit(X_train, y_train)

    dt_importance = pd.Series(
        dt.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    # -----------------------------
    # Feature Selection Method 4: Random Forest
    # -----------------------------
    rf_selection = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    rf_selection.fit(X_train, y_train)

    rf_importance = pd.Series(
        rf_selection.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\nFilter selected:")
    print(filter_selected)

    print("\nRFE selected:")
    print(rfe_selected)

    print("\nDecision Tree importance:")
    print(dt_importance)

    print("\nRandom Forest importance:")
    print(rf_importance)

    # Final model using selected final features
    X_train_final = X_train[FINAL_FEATURES]
    X_test_final = X_test[FINAL_FEATURES]

    final_model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )

    final_model.fit(X_train_final, y_train)

    y_pred = final_model.predict(X_test_final)

    print("\nFinal model accuracy:")
    print(accuracy_score(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(final_model, MODEL_PATH)

    print(f"\nModel saved to {MODEL_PATH}")

    return final_model


def load_model():
    """
    Load saved model. If missing, train it.
    """

    if not os.path.exists(MODEL_PATH):
        return train_model()

    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    train_model()