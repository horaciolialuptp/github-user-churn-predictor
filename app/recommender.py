import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from scraper import fetch_user_repo_interactions


FALLBACK_RECOMMENDATIONS = [
    "freeCodeCamp/freeCodeCamp",
    "EbookFoundation/free-programming-books",
    "public-apis/public-apis",
    "codecrafters-io/build-your-own-x",
    "jwasham/coding-interview-university"
]


class SVDRecommender:
    """
    Simple SVD-based recommender using GitHub starred repositories.
    """

    def __init__(self):
        self.user_item_matrix = None
        self.predicted_scores = None
        self.users = []
        self.items = []

    def fit(self):
        interactions = fetch_user_repo_interactions(max_repos=10)

        if interactions.empty:
            self.user_item_matrix = pd.DataFrame()
            self.predicted_scores = np.array([])
            self.users = []
            self.items = []
            return self

        matrix = interactions.pivot_table(
            index="username",
            columns="repo",
            values="interaction",
            fill_value=0
        )

        self.user_item_matrix = matrix
        self.users = matrix.index.tolist()
        self.items = matrix.columns.tolist()

        matrix_values = matrix.values.astype(float)

        min_dim = min(matrix_values.shape)

        if min_dim <= 2:
            self.predicted_scores = matrix_values
            return self

        k = min(5, min_dim - 1)

        try:
            U, sigma, Vt = svds(matrix_values, k=k)
            sigma_diag = np.diag(sigma)
            self.predicted_scores = np.dot(np.dot(U, sigma_diag), Vt)
        except Exception:
            self.predicted_scores = matrix_values

        return self

    def recommend_for_user(self, username, top_n=5):
        if self.user_item_matrix is None:
            self.fit()

        # If no interaction data was available, use fallback repos
        if self.user_item_matrix is None or self.user_item_matrix.empty:
            return FALLBACK_RECOMMENDATIONS[:top_n]

        # Cold-start fallback: user not in matrix
        if username not in self.users:
            popular_items = (
                self.user_item_matrix.sum(axis=0)
                .sort_values(ascending=False)
                .head(top_n)
                .index
                .tolist()
            )

            if len(popular_items) == 0:
                return FALLBACK_RECOMMENDATIONS[:top_n]

            return popular_items

        user_idx = self.users.index(username)

        user_scores = self.predicted_scores[user_idx].copy()
        already_seen = self.user_item_matrix.iloc[user_idx].values > 0

        user_scores[already_seen] = -999

        top_indices = np.argsort(user_scores)[::-1][:top_n]

        recommendations = [
            self.items[i]
            for i in top_indices
            if user_scores[i] > -999
        ]

        # If SVD has no unseen items to recommend, use popular/fallback repos
        if len(recommendations) == 0:
            popular_items = (
                self.user_item_matrix.sum(axis=0)
                .sort_values(ascending=False)
                .head(top_n)
                .index
                .tolist()
            )

            if len(popular_items) == 0:
                return FALLBACK_RECOMMENDATIONS[:top_n]

            return popular_items

        return recommendations