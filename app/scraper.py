import os
import time
import pandas as pd
import requests


GITHUB_USERS = [
    "torvalds", "gaearon", "sindresorhus", "tj", "yyx990803",
    "addyosmani", "getify", "defunkt", "mojombo", "pjhyett",
    "octocat", "kennethreitz", "mitsuhiko", "pallets", "psf",
    "numpy", "pandas-dev", "scikit-learn", "matplotlib", "fastapi",
    "tiangolo", "docker", "kubernetes", "tensorflow", "pytorch",
    "microsoft", "google", "facebook", "openai", "vercel",
    "nodejs", "rust-lang", "golang", "django", "laravel",
    "spring-projects", "apache", "mozilla", "wordpress", "electron"
]


def fetch_github_users(usernames=None, sleep_time=0.5):
    """
    Fetch public GitHub profile data for a list of usernames.
    """

    if usernames is None:
        usernames = GITHUB_USERS

    records = []

    for username in usernames:
        url = f"https://api.github.com/users/{username}"

        try:
            headers = {
                "Accept": "application/vnd.github+json",
                "User-Agent": "churn-predictor-app"
            }

            token = os.getenv("GITHUB_TOKEN")

            if token:
                headers["Authorization"] = f"Bearer {token}"

            response = requests.get(url, timeout=10, headers=headers)

            if response.status_code != 200:
                print(f"Skipping {username}: status {response.status_code}")
                continue

            data = response.json()

            records.append({
                "username": username,
                "public_repos": data.get("public_repos", 0),
                "public_gists": data.get("public_gists", 0),
                "followers": data.get("followers", 0),
                "following": data.get("following", 0),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
            })

            time.sleep(sleep_time)

        except Exception as e:
            print(f"Error fetching {username}: {e}")

    return pd.DataFrame(records)


def fetch_starred_repos(username, max_repos=10):
    """
    Fetch public repositories starred by a GitHub user.
    Returns a list of repository full names.
    """

    url = f"https://api.github.com/users/{username}/starred"

    try:
        response = requests.get(
            url,
            params={"per_page": max_repos},
            timeout=10,
            headers={"Accept": "application/vnd.github+json"}
        )

        if response.status_code != 200:
            return []

        data = response.json()

        repos = [
            repo.get("full_name")
            for repo in data
            if repo.get("full_name") is not None
        ]

        return repos

    except Exception as e:
        print(f"Error fetching starred repos for {username}: {e}")
        return []


def fetch_user_repo_interactions(usernames=None, max_repos=10, sleep_time=0.5):
    """
    Build a user-repository interaction table.
    Each row means that a user starred a repository.
    """

    if usernames is None:
        usernames = GITHUB_USERS

    interactions = []

    for username in usernames:
        repos = fetch_starred_repos(username, max_repos=max_repos)

        for repo in repos:
            interactions.append({
                "username": username,
                "repo": repo,
                "interaction": 1
            })

        time.sleep(sleep_time)

    return pd.DataFrame(interactions)


if __name__ == "__main__":
    users_df = fetch_github_users()
    print(users_df.head())
    print(users_df.shape)

    interactions_df = fetch_user_repo_interactions()
    print(interactions_df.head())
    print(interactions_df.shape)