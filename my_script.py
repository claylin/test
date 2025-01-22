import httpx
from prefect import flow, task # Prefect flow and task decorators
from prefect.concurrency.sync import rate_limit


@flow(log_prints=True)
# def show_stars(github_repos: list[str]):
#     """Show the number of stars that GitHub repos have"""
#     for repo in github_repos:
#         repo_stats = fetch_stats(repo)
#         stars = get_stars(repo_stats)
#         print(f"{repo}: {stars} stars")
def show_stars(github_repos: list[str]) -> None:
    """Flow: Show number of GitHub repo stars"""

    # Task 1: Make HTTP requests concurrently
    stats_futures = fetch_stats.map(github_repos)

    # Task 2: Once each concurrent task completes, get the star counts
    stars = get_stars.map(stats_futures).result()

    # Show the results
    for repo, star_count in zip(github_repos, stars):
        print(f"{repo}: {star_count} stars")

@task
def fetch_stats(github_repo: str):
    """Fetch the statistics for a GitHub repo"""
    rate_limit("github-api")
    api_response = httpx.get(f"https://api.github.com/repos/{github_repo}")
    api_response.raise_for_status() # Force a retry if not a 2xx status code
    return api_response.json()


@task
def get_stars(repo_stats: dict):
    """Get the number of stars from GitHub repo statistics"""
    return repo_stats['stargazers_count']


if __name__ == "__main__":
    show_stars([
        "PrefectHQ/prefect",
        "pydantic/pydantic",
        "huggingface/transformers"
    ])
