import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# .env 파일 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, '../.env')
load_dotenv(dotenv_path)

# MySQL 데이터베이스 연결 정보 설정
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')

engine = create_engine(f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/aiteam2')

# CSV 파일 경로
credit_file_path = 'cardcrawling/creditcard.csv'
check_file_path = 'cardcrawling/checkcard.csv'

def load_and_insert_to_db(csv_path, table_name, engine):
    # CSV 파일 읽기
    df = pd.read_csv(csv_path, header=0)
    df = df.drop(columns=['idx'])
    
    # 데이터베이스에 데이터 삽입
    try:
        df.to_sql(table_name, con=engine, if_exists='append', index=False)
        print(f"Data inserted successfully into {table_name}.")
    except Exception as e:
        print(f"An error occurred while inserting into {table_name}: {e}")

# creditcard 테이블로 데이터 삽입
load_and_insert_to_db(credit_file_path, 'creditcard', engine)

# checkcard 테이블로 데이터 삽입
load_and_insert_to_db(check_file_path, 'checkcard', engine)