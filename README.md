# 🔍 TrademarkAI — ML Server

> AI 기반 상표 유사도 분석 서비스의 Python ML 분석 서버

---

## 📌 프로젝트 개요

변리사의 선행조사 시간을 줄이는 Mircrosoft Azure 기반 상표 검색 서비스 **TrademarkAI**의 ML 분석 서버입니다.  
상표 이미지와 텍스트를 입력받아 3단계 유사도 분석을 수행하고 결과를 백엔드로 반환합니다.

---

## 🏗️ 시스템 아키텍처

```
Backend (Spring Boot)
  ↕
[ML] Python ML Server
  ↕
Azure AI Services
  ├── Azure Custom Vision  — 비엔나코드 도형 태그 분류 (1차)
  ├── GPT-4o               — 니스/유사군 코드 추출, 식별력 점수 계산 (1·3차)
  ├── Azure AI Vision      — 이미지 벡터 임베딩 유사도 계산 (2차)
  └── AI Vision OCR        — 상표 이미지 내 텍스트 추출 (2차)
```

---

## 🔄 3단계 분석 플로우

### 1차 검사 — 후보군 필터링
- **GPT-4o** + KIPRIS 비엔나 코드 자료집 기반으로 입력 상표의 니스 분류·유사군 코드 추출
- **Azure Custom Vision** 으로 비엔나코드(도형) 태그 분류 (threshold=0.3, 누락 최소화)
- 동일 상품군 + 유사 도형 코드 상표만 2차로 전달

### 2차 검사 — 이미지·텍스트 유사도
- **Azure AI Vision OCR** 로 상표 이미지 내 텍스트 추출
- **Azure AI Vision 임베딩** 으로 이미지 벡터화
- `0.5 × OCR 텍스트 코사인 유사도 + 0.5 × 이미지 임베딩 코사인 유사도` 계산

### 3차 및 리포트
- **GPT-4o (few-shot prompting)** 으로 상표명 식별력 점수 산출
  - 일반 명칭, 설명적 표현, 흔한 단어 조합 등 고려
- **GPT-4o mini (zero-shot prompting)** 으로 유사 후보 종합 분석 리포트 생성

---

## 🛠️ 기술 스택

- **Language**: Python
- **AI/ML**: Azure Custom Vision, Azure AI Vision, OpenAI GPT-4o / GPT-4o mini
- **데이터**: AI Hub 상표 이미지 데이터셋 (약 1,300장, NICE 25·41·42류)

---

## 📊 학습 데이터

| 항목 | 내용 |
|------|------|
| 출처 | AI Hub 상표 이미지 데이터셋 |
| 규모 | 약 1,300장 (텍스트 + 이미지 혼합 상표) |
| 라벨 형식 | JSON (비엔나코드, 니스 분류 포함) |
| 적용 상품군 | NICE 25류(의류), 41류(교육), 42류(과학·IT 서비스) |

---

## 💻 로컬 실행 방법

```bash
# 패키지 설치
pip install -r requirements.txt

# 서버 실행
python main.py
```

> Azure Custom Vision, OpenAI API 키 등 환경 변수 설정이 필요합니다.

---

## 📎 관련 레포지토리

- [FE](https://github.com/Trademark-analysis/FE) — React + TypeScript 프론트엔드
- [BE](https://github.com/Trademark-analysis/BE) — Spring Boot 백엔드 서버
