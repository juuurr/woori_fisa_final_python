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
from sqlalchemy import create_engine, MetaData, Table, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

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



import base64
@app.post("/receive-name")
def receive_name(request: NameRequest):
    import base64  # Base64 인코딩 라이브러리

    name = request.name
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    
    # 데이터베이스에서 해당 사용자의 prompt와 image 확인
    metadata = MetaData()
    metadata.reflect(bind=engine)
    member_table = Table("member", metadata, autoload_with=engine)

    # 데이터베이스 쿼리 실행
    query = f"SELECT prompt, image FROM member WHERE username = '{name}' LIMIT 1;"
    user_data = pd.read_sql(query, engine)
    print(user_data)

    if not user_data.empty:
        prompt = user_data.iloc[0]["prompt"]
        image = user_data.iloc[0]["image"]

        if prompt and image:
            # Base64로 인코딩하여 반환
            image_base64 = base64.b64encode(image).decode("utf-8")
            return {"prompt": prompt, "image": image_base64}

    # 데이터가 없으면 새로운 데이터 생성
    category_result = category(name)
    prompt = gen_prompt(category_result)
    image_url = gen_image(prompt)

    print(prompt, image_url)

    # 이미지 다운로드
    image_response = requests.get(image_url)
    if image_response.status_code == 200:
        image_data = image_response.content
    else:
        raise HTTPException(status_code=500, detail="Image download failed")

    # 새로 생성된 데이터를 데이터베이스에 저장
    try:
        with engine.connect() as conn:
            trans = conn.begin()
            update_query = text("""
                UPDATE member
                SET prompt = :prompt, image = :image
                WHERE username = :username
            """)
            conn.execute(
                update_query,
                {"prompt": prompt, "image": image_data, "username": name}
            )
            trans.commit()
            print("Data successfully saved in the database.")
    except Exception as e:
        print(f"Error while saving data: {e}")
        raise HTTPException(status_code=500, detail="Failed to save data to the database")

    # 저장된 데이터를 Base64로 인코딩하여 반환
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    return {"prompt": prompt, "image": image_base64}




@app.get("/", response_class=HTMLResponse)
def show_index(request: Request, name: str = None):
    if not name:
        # 기본적으로 데이터를 보여주지 않음
        return templates.TemplateResponse("index.html", {"request": request, "prompt": None, "image": None})

    # 데이터베이스에서 데이터 조회
    metadata = MetaData()
    metadata.reflect(bind=engine)
    member_table = Table("member", metadata, autoload_with=engine)

    query = text("SELECT prompt, image FROM member WHERE username = :username")
    with engine.connect() as conn:
        result = conn.execute(query, {"username": name}).fetchone()

    # 데이터가 없으면 에러 메시지 반환
    if not result:
        return templates.TemplateResponse("index.html", {"request": request, "prompt": None, "image": None, "error": "No data found for this user."})

    prompt = result["prompt"]
    image_data = result["image"]

    # Base64로 이미지 인코딩
    image_base64 = base64.b64encode(image_data).decode("utf-8")

    # HTML 렌더링 시 데이터 전달
    return templates.TemplateResponse("index.html", {"request": request, "prompt": prompt, "image": image_base64})