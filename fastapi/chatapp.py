import os
import pickle
from multiprocessing import Pool, cpu_count
from fastapi import FastAPI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from sqlalchemy import create_engine
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import time
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import openai
from openai import OpenAI

# .env 파일 로드
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../.env')
load_dotenv(dotenv_path)

# 환경 변수
api_key = os.getenv("OPENAI_API_KEY")
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')


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
        # 각 카드에 대해 혜택 정보를 결합한 텍스트 생성
        combined_text = f"""
        카드명: {row['name']}
        편의점 혜택: {row['convenience']}
        카페 혜택: {row['cafe']}
        외식 혜택: {row['restaurant']}
        주유 혜택: {row['oil']}
        영화 혜택: {row['movie']}
        쇼핑 혜택: {row['shopping']}
        병원 혜택: {row['hospital']}
        교육 혜택: {row['edu']}
        통신 혜택: {row['tel']}
        자동차 혜택: {row['car']}
        여행 혜택: {row['travel']}
        대중교통 혜택: {row['transportation']}
        """
        vector = embedding.embed_query(combined_text)

        img_url = row.get('img_url', '없음')  # 'img_url'이 없을 경우 '없음'으로 처리
        embedding_data.append({"text": combined_text, "metadata": row.to_dict(), "vector": vector, "img_url": img_url})
        
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
    chroma_db.add_texts(texts, metadatas)
    elapsed_time = time.time() - start_time  # 처리 시간 계산
    print(f"Chroma DB updated in {elapsed_time:.2f} seconds.")
    return chroma_db


def generate_answer_from_openai(question, context):
    prompt = f"다음 내용과 관련된 질문에 대해 답변을 해주세요.:\n{context}\n\n질문: {question}"

    response = openai.chat.completions.create(
        model="gpt-4",  # 사용할 모델
        messages = [
            {"role": "system", "content": "You are a card company employee. Based on the customer's request, provide a simple list of the best cards with the requested benefits in Korean."},
            {"role": "user", "content": prompt}
            ],
        max_tokens=200,
        temperature=0.7
    )
    response_dict = response.model_dump()
    response_message = response_dict["choices"][0]["message"]["content"]
    return response_message


card_data = fetch_data_from_mysql()
embedding_card_data = precompute_card_embeddings(card_data)
card_chroma = store_data_in_chroma(embedding_card_data)


# FastAPI 설정
chatapp = FastAPI()

chatapp.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용   
    allow_headers=["*"],  # 모든 헤더 허용
)

class UserRequest(BaseModel):
    question: str


@chatapp.post("/cardchat")
def chat(request: UserRequest):
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        # Retrieve relevant information from Chroma DB
        search_results = card_chroma.similarity_search(question, k=5)
        
        # Log the search results to debug
        print(f"Search Results: {search_results}")
        
        if not search_results:
            raise HTTPException(status_code=404, detail="No relevant information found.")
        
        # Access text using the correct attribute, e.g., 'content' or 'page_content'
        combined_text = "\n".join([result.page_content for result in search_results])  # Assuming 'page_content'
        
        # Log the combined text to verify
        print(f"Combined Text: {combined_text}")

        # Generate response using OpenAI API
        response_text = generate_answer_from_openai(question, combined_text)

        # 카드 이미지 URL을 포함한 정보 생성
        img_urls = [result.metadata.get('img_url', '없음') for result in search_results]
        card_names = [result.metadata['name'] for result in search_results]

        if response_text:
            return {
                "response": response_text,
                "cards": card_names[:5],  # 카드 이름 5개만 반환
                "img_urls": img_urls[:5]  # 이미지 URL 5개만 반환
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to generate response.")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")