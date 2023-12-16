import random
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
        self.total_weight = sum(item["evaluation"] for item in data)
        self.cumulative_weights = self.calculate_cumulative_weights()

    def calculate_cumulative_weights(self):
        cumulative_weights = []
        current_weight = 0
        for item in self.data:
            current_weight += item["evaluation"]
            cumulative_weights.append(current_weight)
        return cumulative_weights

    def sample(self):
        random_value = random.uniform(0, self.total_weight)
        index = self.find_index(random_value)
        return self.data[index]

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

def recommend_movies(movie_list, num_recommendations=3):
    sampler = WeightedRandomSampler(movie_list)
    selected_movies = [sampler.sample() for _ in range(num_recommendations)]
    return selected_movies

# 映画詳細情報が含まれるJSON
# movies = [
#     {
#         "title": "Movie 1",
#         "evaluation": 4.5
#     },
#     {
#         "title": "Movie 2",
#         "evaluation": 3.5
#     },
#     {
#         "title": "Movie 3",
#         "evaluation": 2.1
#     },
#     {
#         "title": "Movie 4",
#         "evaluation": 4.7
#     },
#     {
#         "title": "Movie 5",
#         "evaluation": 3.2
#     },
#     {
#         "title": "Movie 6",
#         "evaluation": 4.1
#     },
#     {
#         "title": "Movie 7",
#         "evaluation": 1.2
#     },
#     {
#         "title": "Movie 8",
#         "evaluation": 2.5
#     },
#     {
#         "title": "Movie 9",
#         "evaluation": 4.1
#     }
# ]

@app.post("/recommend")
async def recommend_movies_endpoint(requestBody: List[Movie]):
    try:
        recommended_movies = recommend_movies(requestBody)
        return JSONResponse(content={"recommendations": recommended_movies}, status_code=status.HTTP_200_OK)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))