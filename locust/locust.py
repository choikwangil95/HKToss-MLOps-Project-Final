from locust import HttpUser, task
import random


class WinePredictionUser(HttpUser):
    @task(1)
    def healthcheck(self):
        self.client.get("/healthcheck")

    @task(10)
    def similarity(self):
        self.client.get(
            "/news/v2/20250523_0028/similar?top_n=5&min_gap_days=90&min_gap_between=30"
        )
