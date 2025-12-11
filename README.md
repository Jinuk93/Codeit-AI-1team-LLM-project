<div align="center">
  
# 📑 B2G 입찰지원 전문 컨설팅 솔루션 : RFPilot
**"수백 장의 제안요청서, 이제 읽지 말고 질문하세요."**
</div>


## 1. 챗봇 서비스 시연 (Demo)

### 🤖 메인 챗봇 서비스
사용자가 질문하면 AI가 수십 페이지의 RFP 문서를 분석하여 실시간으로 답변하고 근거를 제시합니다.

![chatbot_final](https://github.com/user-attachments/assets/1b321abb-6ba1-4063-be97-300036d8047a)

### 📊 벡터 DB 대시보드 (별도 서비스)
RAG 시스템이 문서를 어떻게 청킹하고 검색하는지 시각화하여 보여줍니다.
- **[👉 대시보드 접속하기](https://vectordb-dashboard-dong.streamlit.app/)**

![Vector_DB_v1](https://github.com/user-attachments/assets/1b12ecf9-a105-44c7-82a4-67744d82931b)

---

## 2. 프로젝트 개요 (Overview)

**RFPilot**은 입찰 컨설턴트의 업무 효율을 극대화하기 위해 개발된 **RAG(검색 증강 생성) 기반 챗봇**입니다.

| 구분 | 상세 내용 |
| :--- | :--- |
| **<br>배경 및 문제 정의** | • **배경** <br>: 매일 수백 건의 기업 및 정부 제안요청서(RFP)가 게시됨<br>• **문제** <br>: 요청서 당 수십 페이지가 넘는 문건을 일일이 검토하는 것은 물리적으로 불가능하며, 핵심 정보를 놓칠 위험이 큼<br>• **비효율** <br>: 단순 문서 검토에 과도한 시간이 소요되어 전략 수립 등 고부가가치 업무에 집중하기 어려움 |
| **목표** | • **자동화** <br>: 사용자의 질문에 실시간으로 응답하는 AI 시스템 구축<br>• **요약 및 탐색** <br>: 관련 제안서를 빠르게 탐색하고 핵심 요약 정보를 제공<br>• **생산성** <br>: 컨설턴트의 단순 업무 시간을 단축하고 업무 효율성을 획기적으로 향상 |
| **기대 효과** | • RAG 시스템을 통해 신뢰할 수 있는 정보를 신속하게 제공 <br>• 제안서 검토 시간 단축을 통해 컨설팅 본연의 업무에 집중할 수 있는 환경 조성 |

---

## 3. 시스템 아키텍처 (Architecture)

사용자의 질문 의도를 파악(Router)하고, 최적의 문서를 검색(Retriever)하여 답변을 생성(Generator)하는 구조입니다. <br>

<img width="4208" height="2004" alt="image" src="https://github.com/user-attachments/assets/6fd35353-7d88-464f-8d75-ff33fabc206b" />


## 4. 팀 소개 (Team)
> **"기본에 충실하며, 실전에서 통하는 서비스를 만들기 위해 끊임없이 노력했습니다."**

| 프로필 | 이름 / 연락처 | 상세 역할 분담 (Role) |
| :---: | :---: | :--- |
| <img src="https://github.com/user-attachments/assets/b9f1a52f-4304-496d-a19c-2d6b4775a5c3" width="100"> | **지동진**<br>_(PM / AI Lead)_<br>[![Github](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/Dongjin-1203)<br>![Gmail](https://img.shields.io/badge/Gmail-D14836?logo=gmail&logoColor=white) | • **프로젝트 총괄** : 전체 기획 및 일정 관리<br>• **Retrieval System** : Retriever 및 Query Router 설계 및 구현<br>• **Engineering** : 로컬 임베딩 모델 최적화, 동적 프롬프트 적용<br>• **DevOps** : Streamlit 대시보드 개발, 배포 환경 구축 및 시스템 통합 |
| <img src="https://avatars.githubusercontent.com/u/80089860?v=4.png" width="100"> | **김진욱**<br>_(Data Scientist)_<br>[![Github](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/Jinuk93)<br>![Gmail](https://img.shields.io/badge/Gmail-D14836?logo=gmail&logoColor=white) | • **Data Pipeline** : 데이터 전처리 및 관리 파이프라인 총괄<br>• **Chunking Strategy** : 문서 구조에 맞는 청킹 전략 수립 및 구현<br>• **Modeling** : 모델 Baseline 제공 및 양자화(Quantization) 최적화 |
| <img src="https://github.com/user-attachments/assets/4e635630-f00c-4026-bb1d-c73ec05f37c8" width="100"> | **이유노**<br>_(AI Engineer)_<br>[![Github](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/Leeyuno0419)<br>![Gmail](https://img.shields.io/badge/Gmail-D14836?logo=gmail&logoColor=white) | • **OpenAI API** : GPT 모델 연동 및 고도화 개발<br>• **Prompt Engineering** : API 모델에 최적화된 프롬프트 설계 및 테스트<br>• **Optimization** : 쿼리 최적화 및 응답 품질 개선 |
| <img src="https://github.com/user-attachments/assets/088a073c-cf1c-40a1-97fb-1d2c1f1b8794" width="100"> | **박지윤**<br>_(AI Engineer)_<br>[![Github](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/krapnuyij)<br>![Gmail](https://img.shields.io/badge/Gmail-D14836?logo=gmail&logoColor=white) | • **Local LLM** : HuggingFace 기반 로컬 모델(Llama-3 등) 개발<br>• **Prompt Engineering** : 로컬 모델 전용 프롬프트 최적화<br>• **Evaluation** : 모델 간 성능 비교 실험 및 평가 수행 |

---

## 5. 프로젝트 타임라인 (Timeline)
<img width="100%" alt="Timeline" src="https://github.com/user-attachments/assets/c06be17f-b82a-4ebc-87a3-45b23a42b5d1" />

---

## 6. 사용 방법 (Getting Started)

| 환경 구분 | 상세 내용 및 실행 방법 |
| :---: | :--- |
| **웹 서비스**<br>*(일반 사용자)* | **설치 없이 브라우저에서 바로 사용 가능합니다.**<br>**- [HuggingFace 데모 사이트 바로가기](https://huggingface.co/spaces/Dongjin1203/RFP_summary_chatbot)**<br>- **사용법** : 링크 접속 → 질문 입력 (예: "사업 기간이 12개월 이하인 사업 찾아줘") |
| **로컬 개발**<br>*(개발자)* | **내 컴퓨터에서 직접 코드를 실행하고 테스트합니다.**<br>- **필수 준비** : Python 3.12.3, Poetry, Git<br>- **데이터셋** : **[필수 데이터 다운로드 링크](https://drive.google.com/file/d/187QnN2VeCfa-nyFMcv8ZtBJP0JxTaY4U/view?usp=drive_link)** (다운로드 후 `data/` 폴더에 위치)<br>- **실행 커맨드** :<br>`poetry install` → `.env` 키 설정 → `python main.py` |


<br> **환경 변수 설정 (.env)**
프로젝트 루트에 `.env` 파일을 생성하고 아래 내용을 입력하세요.
```env
# 필수: OpenAI API (GPT 모델 사용 시)
OPENAI_API_KEY="sk-..."

# 선택: 모니터링 (LangSmith, WandB)
WANDB_API_KEY="..."
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY="..."
LANGCHAIN_PROJECT="입찰메이트"

# 선택: 로컬 모델 (GGUF) 사용 시
USE_MODEL_HUB=false
GGUF_MODEL_PATH="./models/Llama-3-Open-Ko-8B.Q4_K_M.gguf"
GGUF_N_CTX=4096
GGUF_N_GPU_LAYERS=35
```

<br> **설치 및 실행**
```powershell
# 1. 의존성 설치
cd Codeit-AI-1team-LLM-project
python -m poetry config virtualenvs.in-project true
python -m poetry env use 3.12.3
python -m poetry install
python -m poetry shell

# 2. 데이터 전처리 및 벡터 DB 구축 (최초 1회 실행)
python main.py --step all
# (옵션) 단계별 실행: python main.py --step preprocess / embed / vectordb

# 3. 챗봇 애플리케이션 실행
streamlit run src/visualization/chatbot_app.py

# 4. (선택) 실험 및 평가
python src/evaluation/run_experiment.py
```

---

## 7. 기술 스택 (Tech Stack)

| 분류 | 스택 & 라이브러리 |
| :--- | :--- |
| **언어** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) |
| **프레임워크** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) |
| **AI / LLM** | ![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white) ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white) ![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black) |
| **DB / Cloud** | ![ChromaDB](https://img.shields.io/badge/ChromaDB-cc5c29?style=flat-square) ![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4?style=flat-square&logo=google-cloud&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white) |
| **Tools** | ![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white) ![WandB](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=flat-square&logo=WeightsandBiases&logoColor=black) |
| **협업** | ![Notion](https://img.shields.io/badge/Notion-000000?style=flat-square&logo=notion&logoColor=white) ![Discord](https://img.shields.io/badge/Discord-5865F2?style=flat-square&logo=discord&logoColor=white) ![Google Drive](https://img.shields.io/badge/Google_Drive-4285F4?style=flat-square&logo=googledrive&logoColor=white) |

---

## 8. 관련 자료 및 문서 (Documents)

| 구분 | 자료명 | 링크 |
| :--- | :--- | :---: |
| **🔬 연구** | **프로젝트 후속 연구 (QLoRA RAG)** | [GitHub Link](https://github.com/Dongjin-1203/QLoRA_RAG_test) |
| **📑 보고서** | **프로젝트 최종 보고서 (PDF)** | [다운로드](https://drive.google.com/file/d/1p3HHeugJmaiJP4AQpxZZEzAiAngtaHr8/view?usp=sharing) |
| **📢 발표** | **최종 발표 자료 (PPT)** | [다운로드](https://drive.google.com/file/d/1QM88Ayztv5TNaxTXi0z1Xhy6ngHLLKUm/view?usp=sharing) |
| **📝 회고** | **지동진** - 개인 협업 일지 & 블로그 회고 | [Notion](https://www.notion.so/2a2e8d29749a80faa726fc13b879720d) / [blog](https://velog.io/@hambur1203/%EB%B6%80%ED%8A%B8%EC%BA%A0%ED%94%84-3%EC%A3%BC-RAG-%EC%B1%97%EB%B4%87-%EA%B0%9C%EB%B0%9C-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%ED%9A%8C%EA%B3%A0-%EC%B4%88%EC%8B%AC%EC%9D%B4-%EC%A4%91%EC%9A%94%ED%95%98%EB%8B%A4) |
| | **김진욱** - 개인 협업 일지 & 블로그 회고 | [Notion](https://www.notion.so/2a2e8d29749a812b96d9d8a847323ad6) / [blog](https://nugiwiki.tistory.com/tag/%EC%BD%94%EB%93%9C%EC%9E%87%20%EC%A4%91%EA%B8%89%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8) |
| | **이유노** - 개인 협업 일지 | [Notion](https://www.notion.so/2a2e8d29749a81dea0b5dec22b9d1663) |
| | **박지윤** - 개인 협업 일지 | [Notion](https://www.notion.so/2a2e8d29749a8186aff7e0c80534f18f) |

---

## 9. 프로젝트 구조 (Project Structure)
```bash
CODEIT-AI-1TEAM-LLM-PROJECT/
├── main.py                  # 🚀 실행 진입점
├── models/                  # GGUF 모델 (선택)
├── chroma_db/               # 벡터 데이터베이스
├── data/                    # 문서 및 벡터DB 저장 폴더
│   ├── files/               # 원본 RFP 문서
│   └── rag_chunks_final.csv # 전처리 완료된 RAG 용 데이터
├── src/
│   ├── loader/              # 문서 로딩 및 전처리
│   ├── router/              # 쿼리 라우팅
│   ├── prompt/              # 동적 프롬프트
│   ├── evaluation/          # LangSmith 평가
│   ├── embedding/           # 임베딩 생성
│   ├── retriever/           # 문서 검색기
│   ├── generator/           # 응답 생성기
│   ├── visualization/       # UI 구성 (Streamlit)
│   └── utils/               # 공통 함수 모듈
└── README.md
```
