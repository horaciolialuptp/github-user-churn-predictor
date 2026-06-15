# GitHub User Churn Predictor App

## From Prediction to Retention

## 1. Project Overview

The goal of this project is to build a Dockerized machine learning application that predicts whether a GitHub user is likely to churn, meaning that the user has become inactive on the platform. The project uses public data from the GitHub REST API, generates meaningful churn-related features, applies feature selection methods, trains a machine learning model, and deploys the model through a FastAPI prediction endpoint.

The upgraded version of the project goes beyond prediction. It also includes a recommendation endpoint that suggests repositories to re-engage high-risk users. In other words, the system answers two questions:

1. Who is likely to churn?
2. What can we recommend to bring the user back?

The final application runs inside Docker and provides the following endpoints:

* `/health`: checks if the API is running
* `/features`: shows the input features expected by the model
* `/predict`: returns a churn prediction and churn probability
* `/recommend`: returns repository recommendations for high-risk users

---

## 2. Data Source and Churn Definition

The data source used in this project is the public GitHub REST API. The API provides public information about GitHub users, including:

* public repositories
* public gists
* followers
* following
* account creation date
* last updated date

In this project, churn is defined as inactivity for more than 180 days. A user is labeled as churned if the number of days since their last public GitHub activity is greater than 180.

This threshold was chosen because GitHub users may naturally have periods of inactivity, but a period longer than 180 days can reasonably indicate disengagement.

The dataset used in the notebook contained 39 users after one API request timed out. The churn balance was acceptable for this small academic project, although a larger dataset would be necessary for a production model.

---

## 3. Feature Engineering

Feature engineering was the most important part of the project. Raw API fields were transformed into meaningful behavioral indicators. The objective was to represent user inactivity, productivity, account maturity, social engagement, and network position.

The main engineered features were:

* `days_since_last_activity`: number of days since the user was last active
* `account_age_days`: number of days since the GitHub account was created
* `repos_per_year`: public repositories normalized by account age
* `gists_per_year`: public gists normalized by account age
* `follower_ratio`: followers divided by following
* `following_ratio`: following divided by followers
* `followers_per_repo`: followers normalized by repository count
* `has_no_repos`: binary variable showing whether the user has zero public repositories
* `is_new_account`: binary variable showing whether the account is less than one year old
* `high_following_low_followers`: binary indicator for users who follow many accounts but have few followers
* `public_repos`: total public repositories
* `public_gists`: total public gists

These features are more useful than raw fields because they convert basic API data into behavioral signals. For example, `updated_at` by itself is just a timestamp, but `days_since_last_activity` directly measures inactivity, which is strongly related to churn.

---

## 4. Original Feature Selection Methods

Four feature selection methods were applied:

1. Filter methods
2. Recursive Feature Elimination
3. Decision Tree importance
4. Random Forest importance

### 4.1 Filter Method

The filter method used variance threshold and ANOVA F-test. The strongest feature was clearly:

* `days_since_last_activity`

Other useful features included:

* `repos_per_year`
* `public_repos`
* `following_ratio`
* `followers_per_repo`

Some binary features produced warnings because they had low or constant variance. This means they did not vary enough across the dataset to provide useful information for classification.

### 4.2 Recursive Feature Elimination

RFE selected features by repeatedly training a model and removing weaker predictors. It confirmed that inactivity and activity-based variables were important. This supported the result from the filter method.

### 4.3 Decision Tree Importance

The Decision Tree gave very high importance to `days_since_last_activity`. This makes sense because the churn label is directly related to inactivity. However, a single Decision Tree can be unstable because it depends heavily on one training split.

### 4.4 Random Forest Importance

Random Forest provided a more stable feature importance ranking because it averages results across many trees. The most important features were related to inactivity, activity level, repository productivity, and social engagement.

The main conclusion from the original feature selection stage is that feature engineering was more important than model complexity. The most useful variables were not raw API fields, but transformed behavioral indicators.

---

## 5. PCA Dimensionality Reduction

PCA was applied to the engineered feature matrix to reduce dimensionality and visualize users in a compressed space.

The PCA elbow plot showed that around 5 to 6 components explained most of the variance. After that point, the cumulative explained variance curve became almost flat. This means that many of the original features could be compressed, but using only two components loses a significant amount of information.

A 2D PCA scatter plot was created with users colored by churn label. The plot showed partial separation between churned and retained users, but also some overlap. Therefore, PCA was useful for visualization, but not as strong for prediction.

The model comparison confirmed this:

| Model Setup                 | Mean F1 | Std F1 |
| --------------------------- | ------: | -----: |
| Original selected features  |    1.00 |   0.00 |
| PCA 2 components            |    0.28 |  0.254 |
| Original + network features |    0.80 |  0.400 |

The PCA model performed much worse than the original selected features. This suggests that the two principal components lost important churn information. PCA helped visualize the data, but the original engineered features were better for prediction and interpretation.

---

## 6. SVD Recommendation Engine

The upgraded project includes a recommendation system based on SVD. The goal is to move from churn prediction to retention action.

The recommender uses a user-repository interaction matrix based on public starred repositories. Rows represent users, columns represent repositories, and values represent whether a user interacted with a repository.

SVD decomposes this matrix into latent user and repository factors. The idea is that users with similar interaction patterns may be interested in similar repositories. Therefore, the system can recommend repositories that similar users engaged with.

The application includes a `/recommend` endpoint. This endpoint first checks the churn probability. If the user is not high risk, no recommendation is returned. If the user is high risk, the system returns repository recommendations.

An example response was:

```json
{
  "username": "octocat",
  "churn_probability": 0.593,
  "message": "User is high risk. Recommended repositories are provided for re-engagement.",
  "recommendations": [
    "freeCodeCamp/freeCodeCamp",
    "EbookFoundation/free-programming-books",
    "public-apis/public-apis",
    "codecrafters-io/build-your-own-x",
    "jwasham/coding-interview-university"
  ]
}
```

A fallback recommendation list was added because some users have no public starred repositories or the matrix can be sparse. This is similar to real recommender systems, where cold-start users receive popular or default recommendations until more personalized data is available.

---

## 7. Network Analysis

Network analysis was added to include structural information about users. The idea is that isolated users may be more likely to churn, while users who are more connected may be more engaged.

The add-on required centrality features such as:

* degree centrality
* betweenness centrality
* PageRank

Because unauthenticated GitHub API access does not efficiently provide the full follower graph, this project used a proxy network. Users were connected when they had similar public repository and follower patterns.

The network graph showed users as nodes colored by churn label. This helped visualize the relationship between network structure and churn. However, because the graph was built as a proxy network and the dataset was small, the network features did not strongly improve model performance.

The updated feature selection table included:

* `degree_centrality`
* `betweenness_centrality`
* `pagerank`

These network features ranked low or had weak importance compared to activity-based features. This suggests that in this dataset, inactivity and repository activity were stronger churn signals than the proxy network features.

---

## 8. Updated Model Comparison

Three model setups were compared:

1. Original selected features
2. PCA 2 components
3. Original features plus network features

The results were:

| Model Setup                 | Mean F1 | Std F1 |
| --------------------------- | ------: | -----: |
| Original selected features  |    1.00 |   0.00 |
| PCA 2 components            |    0.28 |  0.254 |
| Original + network features |    0.80 |  0.400 |

The best model was the one using the original selected engineered features. This is because those features preserve direct churn signals such as inactivity, account age, repository productivity, and social engagement.

The PCA model performed poorly because reducing the data to two components lost important predictive information.

The model with network features performed reasonably well, but worse than the original feature set. This does not mean network features are useless. It means that in this small dataset, and with a proxy network rather than a real follower graph, the network variables did not add enough signal to improve performance.

---

## 9. FastAPI and Docker Deployment

The final application was deployed using FastAPI and Docker. Docker makes the project reproducible because the app runs with the same dependencies inside a container.

The API includes:

* `GET /health`
* `GET /features`
* `POST /predict`
* `POST /recommend`

The `/predict` endpoint returns a churn prediction and churn probability. For example:

```json
{
  "churned": true,
  "churn_probability": 0.593
}
```

The `/recommend` endpoint adds the retention action. It checks churn probability and returns repository recommendations only for high-risk users.

The project was successfully tested using Docker Compose, and the API was accessed through:

```text
http://localhost:8000/docs
```

---

## 10. Retention Strategy and Product Interpretation

The main business insight is that inactivity is the strongest churn signal. Users who have not been active for a long time are more likely to churn. Repository activity and social engagement also help explain churn risk.

If this were used as a product tool, the company could take actions such as:

* sending re-engagement notifications to inactive users
* recommending repositories related to user interests
* suggesting communities or users to follow
* highlighting beginner-friendly projects for low-activity users
* offering personalized content before the user fully disengages

The recommendation endpoint closes the loop between prediction and action. Instead of only saying that a user may churn, the system suggests something that could help retain the user.

---

## 11. Ethical Considerations

Churn predictions should be used carefully. A model prediction is probabilistic, not certain. A user predicted as high risk is not guaranteed to churn.

It is important not to use churn predictions in manipulative ways. For example, a company should avoid unfair treatment or excessive targeting of users classified as high risk. The model should support better user experience, not pressure users.

The recommendation system should also be transparent and helpful. Recommendations should be relevant and should aim to improve engagement naturally.

---

## 12. Conclusion

This project demonstrates a complete data science workflow, including:

* public API data collection
* churn label design
* feature engineering
* feature selection
* dimensionality reduction
* network analysis
* recommendation systems
* model training
* Dockerized FastAPI deployment

The strongest result is that the original engineered features performed best. PCA helped visualize the data but reduced predictive performance. Network features were conceptually valuable, but did not improve the model in this small proxy-network dataset.

The final application successfully predicts churn risk and recommends repositories for high-risk users. This moves the project from simple prediction toward a practical retention system.
