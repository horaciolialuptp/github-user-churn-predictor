# GitHub User Churn Predictor App

This project predicts GitHub user churn using public GitHub API data. It includes feature engineering, feature selection, PCA visualization, SVD-based recommendations, network analysis, and a Dockerized FastAPI application.

## Main Features

* Churn prediction using engineered GitHub user features
* PCA elbow plot and 2D PCA visualization
* SVD recommendation system using user-repository interactions
* NetworkX graph with centrality features
* FastAPI endpoints:

  * `/health`
  * `/features`
  * `/predict`
  * `/recommend`
* Docker Compose deployment

## How to Run

Run the project with:

`docker compose up --build`

Then open:

`http://localhost:8000/docs`

## Report

See `report.md` for the full project explanation, model comparison, recommendation logic, and PM reflection.
