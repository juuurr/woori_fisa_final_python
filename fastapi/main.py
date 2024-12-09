from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dotenv import load_dotenv
from sqlalchemy import create_engine, text, MetaData, Table, inspect
from sqlalchemy.orm import sessionmaker
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import os
import pandas as pd
import pickle
import time
import requests
from tqdm import tqdm
import openai as openai_lib
from openai import OpenAI

# .env 파일 로드
load_dotenv()


# 환경 변수
api_key = os.getenv("OPENAI_API_KEY")
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')

# 데이터베이스 엔진 설정
engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/aiteam2")

# FastAPI 통합 앱 생성
app = FastAPI()


# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

######################################################################################1. #프롬프트/이미지 생성

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
                "content": f"Create a witty sentence that sums up last month's expense tendency for a user who primarily spends in the category of {category}, \
                using baby animals and emojis. The animal must be: Baby{character}."

            }
        ],
        max_tokens=200,
        temperature=1.0,
    )

    prompt = RESPONSE.choices[0].message.content
    return prompt

def gen_image(prompt, category, character):
    prompt_input = f"Create a mascot-like illustration based on {prompt}. A fluffy, adorable, small baby {character} with a loving and cheerful expression, sitting in the center of the image. The baby {character} exudes warmth and coziness, with a design reminiscent of a wholesome Disney movie. The baby {character} is holding objects associated with the {category}, that can easily represent {category}. The illustration is bright and sunny, with soft, clean lighting. The background is a solid, clean one color in hex code #A0DAE3, keeping the focus entirely on the cute baby {character}. The composition is 1:1, with the baby {character} taking up the majority of the frame, evoking a cozy and heartwarming atmosphere."
    
    image_response = client.images.generate(
        model="dall-e-3",
        prompt=prompt_input,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    image_url = image_response.data[0].url
    return image_url


# 데이터 모델 정의
class UserData(BaseModel):
    userId: str
    character: str

@app.post("/api/process-user-data")
async def process_user_data(data: UserData):
    try:
        # 데이터 출력 (또는 필요한 로직 처리)
        print(f"Received userId: {data.userId}, character: {data.character}")
        user_id = data.userId
        character = data.character

        user_category = category(user_id)
        generated_prompt = gen_prompt(user_category, character)
        img_url = gen_image(generated_prompt, user_category, character)

        print(generated_prompt, img_url)

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
                    {"prompt": generated_prompt, "image": image_data, "user_id": user_id}
                )
                print("Data successfully saved in the database.")
        except Exception as e:
            print(f"Error while saving data: {e}")
            raise HTTPException(status_code=500, detail="Failed to save data to the database")

    except Exception as e:
        # 에러 처리
        raise HTTPException(status_code=500, detail=str(e))



######################################################################################2. 챗봇

# 데이터베이스 연결 및 데이터 로드
def fetch_data_from_mysql():
    engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/aiteam2")
    query = "SELECT * FROM card;"
    data = pd.read_sql(query, con=engine)
    return data.fillna('없음')

# 카드별 혜택 벡터화
def precompute_card_embeddings(data):
    if os.path.exists("embedding_cache.pkl"):
        print("Loading precomputed embeddings from cache...")
        with open("embedding_cache.pkl", "rb") as f:
            return pickle.load(f)

    embedding = OpenAIEmbeddings(api_key=api_key)
    embedding_data = []

    start_time = time.time()  # 시작 시간 기록
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Embedding cards"):
        # 카드명과 각 혜택을 개별적으로 벡터화
        # 각 혜택별로 텍스트를 생성하여 벡터화합니다.
        
        # 각 혜택 항목을 개별적으로 처리
        convenience_text = f"편의점 혜택: {row['convenience']}"
        convenience_vector = embedding.embed_query(convenience_text)
        
        restaurant_text = f"외식 혜택: {row['restaurant']}"
        restaurant_vector = embedding.embed_query(restaurant_text)
        
        oil_text = f"주유 혜택: {row['oil']}"
        oil_vector = embedding.embed_query(oil_text)
        
        movie_text = f"영화 혜택: {row['movie']}"
        movie_vector = embedding.embed_query(movie_text)
        
        shopping_text = f"쇼핑 혜택: {row['shopping']}"
        shopping_vector = embedding.embed_query(shopping_text)
        
        hospital_text = f"병원 혜택: {row['hospital']}"
        hospital_vector = embedding.embed_query(hospital_text)
        
        edu_text = f"교육 혜택: {row['edu']}"
        edu_vector = embedding.embed_query(edu_text)
        
        tel_text = f"통신 혜택: {row['tel']}"
        tel_vector = embedding.embed_query(tel_text)
        
        car_text = f"자동차 혜택: {row['car']}"
        car_vector = embedding.embed_query(car_text)
        
        travel_text = f"여행 혜택: {row['travel']}"
        travel_vector = embedding.embed_query(travel_text)
        
        transportation_text = f"대중교통 혜택: {row['transportation']}"
        transportation_vector = embedding.embed_query(transportation_text)
        
        # 카드 이미지 URL 처리
        img_url = row.get('img_url', '없음')  # 'img_url'이 없을 경우 '없음'으로 처리

        # 벡터화된 데이터를 embedding_data에 저장
        embedding_data.append({
            "text": f"카드명: {row['name']}",
            "metadata": row.to_dict(),  # 카드 정보 전체
            "vectors": {
                "convenience": convenience_vector,
                "restaurant": restaurant_vector,
                "oil": oil_vector,
                "movie": movie_vector,
                "shopping": shopping_vector,
                "hospital": hospital_vector,
                "edu": edu_vector,
                "tel": tel_vector,
                "car": car_vector,
                "travel": travel_vector,
                "transportation": transportation_vector,
            },
            "img_url": img_url
        })

    elapsed_time = time.time() - start_time  # 처리 시간 계산
    print(f"Embedding completed in {elapsed_time:.2f} seconds.")

    with open("embedding_cache.pkl", "wb") as f:
        pickle.dump(embedding_data, f)

    return embedding_data



def store_data_in_chroma(embedding_data):
    embedding = OpenAIEmbeddings(api_key=api_key)
    chroma_db = Chroma(collection_name="card_vectorstore", embedding_function=embedding)

    texts = [item["text"] for item in embedding_data]
    metadatas = [item["metadata"] for item in embedding_data]

    print("Adding texts to Chroma DB...")
    start_time = time.time()  # 시작 시간 기록
    batch_size = 166  # Chroma에서 허용하는 최대 배치 크기
    for i in range(0, len(texts), batch_size):
        # 텍스트와 메타데이터를 배치 크기만큼 나눔
        batch_texts = texts[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        
        # Chroma에 배치 추가
        chroma_db.add_texts(batch_texts, batch_metadatas)
    elapsed_time = time.time() - start_time  # 처리 시간 계산
    print(f"Chroma DB updated in {elapsed_time:.2f} seconds.")
    return chroma_db



def generate_answer_from_openai(message, context, content):
    
    prompt = f"다음 내용과 관련된 질문에 대해 답변을 해주세요.:\n{context}\n\n질문: {message}"

    response = openai_lib.chat.completions.create(
        model="gpt-4",  # 사용할 모델
        messages = [
            {"role": "system", "content": content},
            {"role": "user", "content": prompt}
            ],
        max_tokens=500,
        temperature=0.7
    )
    response_dict = response.model_dump()
    response_message = response_dict["choices"][0]["message"]["content"]
    return response_message


card_data = fetch_data_from_mysql()
embedding_card_data = precompute_card_embeddings(card_data)
card_chroma = store_data_in_chroma(embedding_card_data)



class UserRequest(BaseModel):
    question: str

flag = False
response_text1 = None
card_names1 = []
img_urls1 = []
question1 = ''
@app.post("/cardchat")  
async def chat(request: Request, user_request: UserRequest):
    global flag, response_text1, card_names1, img_urls1, question1
    try:
        question = user_request.question.strip()
        # 첫 번째 질문에서의 내용을 두 번째 질문에 포함시킬 수 있도록 question1을 활용
        if question1:
            question += " " + question1
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
        # 카드 관련 필수 키워드 확인
        required_keywords = ["편의점", "마트", "카페", "외식", "주유", "쇼핑", "병원", 
                             "교육", "통신", "자동차", "여행", "교통", "카드"]
        
        found_keywords = [keyword for keyword in required_keywords if keyword in question]
        
        is_financial_question = any(keyword in question for keyword in required_keywords)

        keyword = ', '.join(found_keywords)
        query = keyword + " 혜택 카드"
        search_results = card_chroma.similarity_search(query, k=10)

        if is_financial_question:
            if not flag:  # 첫 번째 질문 처리
                question1 = question  # 첫 번째 질문에서 사용될 변수 설정
                combined_text1 = "\n".join([result.page_content for result in search_results])

                # 첫 번째 질문에서 추천된 카드 목록 저장
                card_names1 = []
                img_urls1 = []

                for result in search_results:

                    if category in result.metadata and result.metadata[category] != '없음':
                        card_names1.append(card_name)
                        img_urls1.append(img_url)

                    card_name = result.metadata.get('name', '기타 카드')
                    img_url = result.metadata.get('img_url', '없음')

                    if card_name in combined_text1:
                        card_names1.append(card_name)
                        img_urls1.append(img_url)
                    
                    if len(card_names1) > 3 : 
                        break

                # 첫 번째 질문에 대해 답변 생성 (추천된 카드 정보 반영)
                content1 = f"Based on the customer's request, \
                provide the requested {keyword} benefits of the only 'three' card :{', '.join(card_names1)}, in Korean.\
                Only mention the cards with benefits."
                response_text1 = generate_answer_from_openai(question1, combined_text1, content1)

                flag = True  # flag 업데이트

                if response_text1:
                    return {
                        "response": response_text1,
                        "cards": card_names1, 
                        "img_urls": img_urls1,  # 첫 번째 질문에서 추천된 카드만 반환
                    }
                else:
                    raise HTTPException(status_code=500, detail="Failed to generate response.")
            
            else:  # 두 번째 질문 처리
                combined_text2 = "\n".join([result.page_content for result in search_results])

                card_names2 = []
                img_urls2 = []

                for result in search_results:

                    if category in result.metadata and result.metadata[category] != '없음':
                        card_names1.append(card_name)
                        img_urls1.append(img_url)

                    card_name = result.metadata.get('name', '기타 카드')
                    img_url = result.metadata.get('img_url', '없음')
                    
                    if (card_name in combined_text2) and (card_name not in card_names1):
                        card_names2.append(card_name)
                        img_urls2.append(img_url)
                    if len(card_names2) > 3:
                        break

                # 두 번째 질문에 대해 첫 번째 추천 카드들을 제외하고 새로운 카드 추천
                content2 = f" Based on the customer's request, \
                provide the requested {keyword} benefits of the only three card :{', '.join(card_names2)}, in Korean.\
                Only mention the cards with benefits."
                response_text2 = generate_answer_from_openai(question, combined_text2, content2)

                if response_text2:
                    return {
                        "response": response_text2,
                        "cards": card_names2, 
                        "img_urls": img_urls2,
                    }
                else:
                    raise HTTPException(status_code=500, detail="Failed to generate response.")
        
        else:
            error_response = "죄송합니다. 카드 관련 질문에만 답변을 제공할 수 있습니다."
            return {"response": error_response}

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")