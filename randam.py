import random
import os
import requests
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List


app = FastAPI()

class Movie(BaseModel):
    title: str
    evaluation: float


class WeightedRandomSampler:
    def __init__(self, data):
        self.data = data
        self.total_weight = sum(item["Evaluation"] for item in data)
        self.cumulative_weights = self.calculate_cumulative_weights()

    def calculate_cumulative_weights(self):
        cumulative_weights = []
        current_weight = 0
        for item in self.data:
            current_weight += item["Evaluation"]
            cumulative_weights.append(current_weight)
        return cumulative_weights

    def sample(self, selected_indices):
        remaining_weight = self.total_weight
        for idx in selected_indices:
            remaining_weight -= self.data[idx]["Evaluation"]

        # 残りの評価値を基にランダムサンプリング
        random_value = random.uniform(0, remaining_weight)
        index = self.find_index(random_value)
        return index

    def find_index(self, value):
        # 二分木探索を実行
        low, high = 0, len(self.cumulative_weights) - 1
        while low < high:
            mid = (low + high) // 2
            if value > self.cumulative_weights[mid]:
                low = mid + 1
            else:
                high = mid
        return low
    
def fetch_endpoint(category):
    endpoint_url = os.getenv("BACKEND_URL") + category 

    response = requests.get(endpoint_url)

    return response.json()

def recommend_movies(movie_list, num_recommendations=3):
    sampler = WeightedRandomSampler(movie_list)
    selected_indices = []
    for _ in range(num_recommendations):
        # 重複を避けるために選択済みの映画のインデックスを保持
        index = sampler.sample(selected_indices)
        selected_indices.append(index)

    # 選択された映画を取得
    selected_movies = [movie_list[index] for index in selected_indices]
    return selected_movies

@app.post("/recommend")
async def recommend_movies_endpoint(category: str):
    try:
        res = fetch_endpoint(category)
        recommended_movies = recommend_movies(res)
        return JSONResponse(content={"recommendations": recommended_movies}, status_code=status.HTTP_200_OK)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))