#!/bin/bash
set -e

echo "🔄 ChromaDB 초기화 중..."

# 원본 ChromaDB가 있는지 확인
if [ -d "/app/chroma_db" ]; then
    echo "✅ 원본 ChromaDB 발견: /app/chroma_db"
    
    # 쓰기 가능한 위치로 복사
    echo "📦 ChromaDB를 쓰기 가능한 위치로 복사 중..."
    cp -r /app/chroma_db /app/.cache/chroma_db
    
    echo "✅ ChromaDB 복사 완료: /app/.cache/chroma_db"
else
    echo "⚠️  원본 ChromaDB를 찾을 수 없습니다: /app/chroma_db"
    exit 1
fi

echo "🚀 Streamlit 앱 시작..."
exec streamlit run src/visualization/chatbot_app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.fileWatcherType=none