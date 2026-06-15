from datetime import datetime, timezone

import networkx as nx
import numpy as np
import pandas as pd


def add_network_features(df):
    """
    Create a simple user network based on follower/following-style similarity.
    Since the unauthenticated GitHub API does not directly give all follow edges
    efficiently, this builds a proxy network:
    users with similar social profiles are connected.
    """

    df = df.copy()

    G = nx.Graph()

    usernames = df["username"].tolist()

    for username in usernames:
        G.add_node(username)

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            u1 = df.iloc[i]
            u2 = df.iloc[j]

            repos_close = abs(u1["public_repos"] - u2["public_repos"]) <= 10
            followers_close = abs(u1["followers"] - u2["followers"]) <= 50

            if repos_close and followers_close:
                G.add_edge(u1["username"], u2["username"])

    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G) if len(G.nodes) > 0 else {}

    df["degree_centrality"] = df["username"].map(degree_centrality).fillna(0)
    df["betweenness_centrality"] = df["username"].map(betweenness_centrality).fillna(0)
    df["pagerank"] = df["username"].map(pagerank).fillna(0)

    return df, G


def generate_features(df):
    """
    Transform raw GitHub API data into model-ready churn features.
    """

    df = df.copy()

    now = datetime.now(timezone.utc)

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)

    df["days_since_last_activity"] = (now - df["updated_at"]).dt.days
    df["account_age_days"] = (now - df["created_at"]).dt.days

    df["account_age_years"] = df["account_age_days"] / 365.25

    df["repos_per_year"] = df["public_repos"] / (df["account_age_years"] + 1)
    df["gists_per_year"] = df["public_gists"] / (df["account_age_years"] + 1)

    df["follower_ratio"] = df["followers"] / (df["following"] + 1)
    df["following_ratio"] = df["following"] / (df["followers"] + 1)
    df["followers_per_repo"] = df["followers"] / (df["public_repos"] + 1)

    df["has_no_repos"] = (df["public_repos"] == 0).astype(int)
    df["is_new_account"] = (df["account_age_days"] < 365).astype(int)
    df["high_following_low_followers"] = (
        (df["following"] > 50) & (df["followers"] < 10)
    ).astype(int)

    # Network features
    df, graph = add_network_features(df)

    # Churn label:
    # A GitHub user is considered churned if inactive for more than 180 days.
    df["churned"] = (df["days_since_last_activity"] > 180).astype(int)

    feature_cols = [
        "days_since_last_activity",
        "account_age_days",
        "repos_per_year",
        "gists_per_year",
        "follower_ratio",
        "following_ratio",
        "followers_per_repo",
        "has_no_repos",
        "is_new_account",
        "high_following_low_followers",
        "public_repos",
        "public_gists",
        "degree_centrality",
        "betweenness_centrality",
        "pagerank",
    ]

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0)
    df[feature_cols] = df[feature_cols].fillna(0)

    return df, feature_cols