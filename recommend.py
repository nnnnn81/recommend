import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List


app = FastAPI()


class MovieRating(BaseModel):
    userId: int
    movieId: int
    evaluation: float

def user_similarity_matrix(user_ratings):
    # ゆーざーの評価情報から評価行列つくる
    user_item_matrix = {}
    movie_ids = set()
    for entry in user_ratings:
        user_id, movie_id, evaluation = entry['userId'], entry['movieId'], entry['evaluation']
        if user_id not in user_item_matrix:
            user_item_matrix[user_id] = {}
        user_item_matrix[user_id][movie_id] = evaluation
        movie_ids.add(movie_id)

    # numpy配列にへんかんする
    ratings_array = []
    for user_id, ratings in user_item_matrix.items():
        row = [ratings.get(movie_id, 0) for movie_id in sorted(movie_ids)]
        ratings_array.append(row)

    ratings_array = np.array(ratings_array)

    # ユーザー間の類似性行列計算、return
    similarity_matrix = cosine_similarity(ratings_array)

    return similarity_matrix

# # テストデータ
# user_ratings_data = [
#     {"userId": 1, "movieId": 101, "evaluation": 4.5},
#     {"userId": 1, "movieId": 1, "evaluation": 3.5},
#     {"userId": 2, "movieId": 101, "evaluation": 5.0},
#     {"userId": 2, "movieId": 103, "evaluation": 4.0},
#     {"userId": 3, "movieId": 101, "evaluation": 5.0},
#     {"userId": 3, "movieId": 102, "evaluation": 4.0},
#     {"userId": 4, "movieId": 102, "evaluation": 1.0},
#     {"userId": 4, "movieId": 101, "evaluation": 4.0},
# ]

def recommend_movies_for_user(user_id, user_similarity_matrix, user_ratings_data, num_recommendations=5):
    # ユーザーが評価済みの映画を取得
    watched_movies = {entry['movieId']: entry['evaluation'] for entry in user_ratings_data if entry['userId'] == user_id}

    # 類似ユーザー取得
    similar_users = np.argsort(user_similarity_matrix[user_id - 1])[::-1] + 1  # 似てる順でそーと
    similar_users = similar_users[similar_users != user_id]  # 自分をのぞく

    # おすすめ映画を格納する辞書
    recommended_movies = {}

    # 類似ユーザーの評価からおすすめ映画を取得、ただしすでに評価済みの映画は含めない
    for similar_user_id in similar_users:
        similar_user_ratings = {entry['movieId']: entry['evaluation'] for entry in user_ratings_data if entry['userId'] == similar_user_id}
        for movie_id, evaluation in similar_user_ratings.items():
            if movie_id not in watched_movies:
                if movie_id not in recommended_movies:
                    recommended_movies[movie_id] = [evaluation, 1]
                else:
                    recommended_movies[movie_id][0] += evaluation
                    recommended_movies[movie_id][1] += 1

    # おすすめ映画を評価順でソート
    sorted_recommendations = sorted(recommended_movies.items(), key=lambda x: x[1][0] / x[1][1], reverse=True)

    # 上位{num_recommendations}個のおすすめ映画を取得
    top_recommendations = sorted_recommendations[:num_recommendations]

    return top_recommendations


@app.post("/userbasedrecommend/{userid}")
async def user_recommend_movies_endpoint(userid: int, requestBody: List[MovieRating]):
    try:
        # 類似度計算
        similarity_matrix = user_similarity_matrix(requestBody)

        # ユーザーへのおすすめえいがの情報取得
        recommendations = recommend_movies_for_user(userid, similarity_matrix, requestBody)

        # おすすめ映画の情報をJSONレスポンスとして返す
        response = JSONResponse(content={"recommendations": [movie_id for movie_id, _ in recommendations]}, status_code=status.HTTP_200_OK)
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))