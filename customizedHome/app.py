from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import pandas as pd
from sqlalchemy import create_engine
import pymysql
import requests
from openai import OpenAI
from fastapi.responses import FileResponse
from sqlalchemy import inspect

# .env 파일 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, '../.env')
load_dotenv(dotenv_path)

# 환경 변수 가져오기
api_key = os.getenv("OPENAI_API_KEY")
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')

engine = create_engine(f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/aiteam2')

def category(name):
    info_query = f"SELECT agegroup, gender FROM member where username = '{name}' LIMIT 1;"
    info_df = pd.read_sql(info_query, engine)

    agegroup = info_df.iloc[0]['agegroup']
    gender = info_df.iloc[0]['gender']
    gender = gender.lower()

    db = f'{gender}_{agegroup}_241115'
    inspector = inspect(engine)

    query = f"SELECT * FROM {db}"
    df = pd.read_sql(query, engine)
    if db not in inspector.get_table_names():
        pass #신규회원

    # 기타, 교통 제외
    filtered_data = df[~df['category_name'].isin(['기타', '교통(대중)'])].copy()
    filtered_data['year_month'] = pd.to_datetime(filtered_data['datetime']).dt.to_period('M').astype(str)

    # 2024-10 데이터 필터링
    october_data = filtered_data[(filtered_data['year_month'] == '2024-10')]
    october_category_totals = (
        october_data.groupby('category_name')['amount']
        .sum()
        .reset_index()
        .sort_values(by='amount', ascending=False)
    )

    # 가장 많이 소비한 카테고리 추출
    predicted_top_categories = october_category_totals.head(1)
    if predicted_top_categories.empty:
        raise HTTPException(status_code=404, detail="No data found for user.")

    category = predicted_top_categories['category_name'].iloc[0]
    if '/' in category:
        category = category.split('/')[0]

    return category

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=api_key)

# 제한된 동물 리스트
allowed_animals = ["햄스터", "다람쥐", "토끼", "고양이", "강아지", "수달", "아기곰", 
                   "펭귄", "돼지", "아기호랑이", "아기사자", "북극곰", "달팽이", 
                   "거북이", "참새", "돌고래", "고슴도치", "너구리", "오리"]

def gen_prompt(category):
    RESPONSE = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Create witty sentences in Korean with emojis. Always ensure the response adheres to the prompt instructions.",
            },
            {
                "role": "user",
                "content": f"Create a witty sentence for a user who primarily spends in the category of {category}, using animals and emojis. The animal must be chosen from the following list only: {', '.join(allowed_animals)}",
            }
        ],
        max_tokens=200,
        temperature=1.0,
    )

    prompt = RESPONSE.choices[0].message.content
    return prompt

def gen_image(prompt):
    prompt_input = f"Create an illustration based on the sentence: {prompt} in a 2D Disney-like style."
    image_response = client.images.generate(
        model="dall-e-3",
        prompt=prompt_input,
        size="1024x1024",
        quality="hd",
        n=1,
    )
    image_url = image_response.data[0].url
    return image_url

app = FastAPI()

class NameRequest(BaseModel):
    name: str

@app.post("/receive-name")
def receive_name(request: NameRequest):
    name = request.name
    category_result = category(name)
    
    
    prompt = gen_prompt(category_result)
    image_url = gen_image(prompt)

    print(category_result, prompt, image_url)

    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    return {"prompt": prompt, "image_url": image_url}