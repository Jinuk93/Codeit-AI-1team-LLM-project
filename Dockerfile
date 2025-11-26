# ===== Stable CPU Dockerfile for HF Spaces =====
FROM python:3.10-slim

# 시스템 패키지 설치 (한 번에 실행)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# 의존성 복사
COPY requirements.txt .

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# llama-cpp-python (CPU 버전)
RUN pip install --no-cache-dir llama-cpp-python==0.3.16

# 나머지 의존성
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 파일 복사
COPY . .

# Streamlit 포트
EXPOSE 7860

# 실행
CMD ["streamlit", "run", "src/visualization/chatbot_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]