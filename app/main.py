from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from model import load_model, FINAL_FEATURES
from recommender import SVDRecommender


app = FastAPI(title="GitHub User Churn Predictor API")

model = load_model()
recommender = SVDRecommender().fit()


class UserFeatures(BaseModel):
    days_since_last_activity: float
    account_age_days: float
    repos_per_year: float
    follower_ratio: float
    followers_per_repo: float
    public_repos: float
    degree_centrality: float = 0
    pagerank: float = 0


class RecommendationRequest(BaseModel):
    username: str
    top_n: int = 5
    days_since_last_activity: float
    account_age_days: float
    repos_per_year: float
    follower_ratio: float
    followers_per_repo: float
    public_repos: float
    degree_centrality: float = 0
    pagerank: float = 0


@app.get("/")
def home():
    return {
        "message": "GitHub User Churn Predictor API",
        "endpoints": ["/health", "/features", "/predict", "/recommend", "/docs"]
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/features")
def features():
    return {
        "expected_features": FINAL_FEATURES,
        "description": "Send these fields to /predict as JSON."
    }


@app.post("/predict")
def predict_churn(user: UserFeatures):
    features_array = np.array([[
        user.days_since_last_activity,
        user.account_age_days,
        user.repos_per_year,
        user.follower_ratio,
        user.followers_per_repo,
        user.public_repos,
        user.degree_centrality,
        user.pagerank,
    ]])

    pred = model.predict(features_array)[0]
    prob = model.predict_proba(features_array)[0][1]

    return {
        "churned": bool(pred),
        "churn_probability": round(float(prob), 3)
    }


@app.post("/recommend")
def recommend(req: RecommendationRequest):
    features_array = np.array([[
        req.days_since_last_activity,
        req.account_age_days,
        req.repos_per_year,
        req.follower_ratio,
        req.followers_per_repo,
        req.public_repos,
        req.degree_centrality,
        req.pagerank,
    ]])

    churn_prob = model.predict_proba(features_array)[0][1]

    if churn_prob < 0.5:
        return {
            "username": req.username,
            "churn_probability": round(float(churn_prob), 3),
            "message": "User is not high risk, so no retention recommendation is needed.",
            "recommendations": []
        }

    recommendations = recommender.recommend_for_user(
        username=req.username,
        top_n=req.top_n
    )

    return {
        "username": req.username,
        "churn_probability": round(float(churn_prob), 3),
        "message": "User is high risk. Recommended repositories are provided for re-engagement.",
        "recommendations": recommendations
    }