from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import os
import pandas as pd
from sqlalchemy import create_engine
import pymysql
import requests
from openai import OpenAI
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, MetaData, Table, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import base64
from fastapi.middleware.cors import CORSMiddleware



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

# 템플릿 설정
templates = Jinja2Templates(directory="templates")

def category(user_id):
    with engine.connect() as conn:
        print(f"Fetching category for user_id: {user_id}")  # 디버깅 로그 추가
        # 결제 기록 개수 확인
        count_query = text("SELECT COUNT(*) FROM payment WHERE member_id = :user_id")
        count_result = conn.execute(count_query, {"user_id": user_id}).scalar()

        # 카테고리 결정
        if count_result == 0:
            category_query = text("SELECT category FROM member WHERE id = :user_id LIMIT 1")
        else:
            category_query = text(
                """
                SELECT category_name 
                FROM payment 
                WHERE member_id = :user_id 
                GROUP BY category_name 
                ORDER BY MAX(amount) DESC 
                LIMIT 1
                """  
            ) #날짜 추가

        # 카테고리 결과 가져오기
        category_result = conn.execute(category_query, {"user_id": user_id}).fetchone()

        if not category_result:
            raise HTTPException(status_code=404, detail="No category found for user.")

        # 카테고리 반환
        category = category_result[0]
    return category
        

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=api_key)


def gen_prompt(category, character):
    RESPONSE = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Create witty sentences in Korean with emojis. Always ensure the response adheres to the prompt instructions.",
            },
            {
                "role": "user",
                "content": f"Create a witty sentence for a user who primarily spends in the category of {category}, \
                using animals and emojis. The animal must be: {character}. If the sentence spans more than one line, please add a line break (\\n) between sentences."

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

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 특정 도메인을 지정하거나 ["http://localhost:3000"]로 설정
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# 데이터 모델 정의
class UserData(BaseModel):
    userId: str
    character: str

@app.post("/api/process-user-data")
async def process_user_data(data: UserData):
    import base64
    try:
        # 데이터 출력 (또는 필요한 로직 처리)
        print(f"Received userId: {data.userId}, character: {data.character}")
        user_id = data.userId
        character = data.character

        user_category = category(user_id)
        prompt = gen_prompt(user_category, character)
        img_url = gen_image(prompt)

        print(prompt, img_url)

        print("Requesting image...")
        image_response = requests.get(img_url)
        if image_response.status_code == 200:
            image_data = image_response.content
            print("Image successfully downloaded")
        else:
            print(f"Failed to download image. Status code: {image_response.status_code}")
            raise HTTPException(status_code=500, detail="Image download failed")


        # 새로 생성된 데이터를 데이터베이스에 저장
        try:
            with engine.begin() as conn:
                update_query = text("""
                    UPDATE member
                    SET prompt = :prompt, image = :image
                    WHERE id = :user_id
                """)
                conn.execute(
                    update_query,
                    {"prompt": prompt, "image": image_data, "user_id": user_id}
                )
                print("Data successfully saved in the database.")
        except Exception as e:
            print(f"Error while saving data: {e}")
            raise HTTPException(status_code=500, detail="Failed to save data to the database")

    except Exception as e:
        # 에러 처리
        raise HTTPException(status_code=500, detail=str(e))

