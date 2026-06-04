"""
유사군 코드 분류 스크립트
- 서비스/상품 설명 입력 → Azure OpenAI → 니스분류 + 유사군 코드 반환
- 대상: 25류(의류/신발/모자), 41류(교육/연예/스포츠), 42류(과학/기술/IT)
"""

from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

# ============================================================
# 🔧 여기만 수정하면 돼!
# ============================================================
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
API_VERSION = "2024-02-01"
# ============================================================

SYSTEM_PROMPT = """너는 한국 특허청 상표 유사군 코드 분류 전문가야.
사용자가 서비스 또는 상품을 설명하면, 아래 유사군 코드 목록을 참고해서
해당하는 니스분류(25류/41류/42류)와 유사군 코드를 추천해줘.

[25류 - 의류/신발/모자 유사군 코드]
- 가죽반코트/가죽제 롱코트/모직코트/바바리코트/패딩코트 등 코트류: G450101
- 긴소매셔츠/반소매셔츠/티셔츠/맨투맨/속옷/내의 등 셔츠·내의류: G4503
- 구두/가죽신발/골프신발/레이스부츠/신발용깔창 등 신발류: G270101
- 럭비셔츠/요가복/수영모/사이클복/스키팬츠 등 전문스포츠의류: G430301
- 스포츠용 낙하복/유니폼(운동용): G430305, G450101
- 수면안대/여성용 모피목도리: G450401
- 의류용 후드/두건: G450501
- 방수피복: G4513
- 혁대/허리띠: G4508

[41류 - 교육/연예/스포츠업 유사군 코드]
- 경마공연업/연예물 공연업/연예물 제작배급업/연예인을 위한 모델업: S110101
- 비디오카메라 대여업/영사기 부속품임대업: S110102
- 녹음음반임대업: S110103
- 박물관 운영업: S120902
- 과학학원/영어어학원 운영업: S120903
- 교육지도업(온·오프라인): S120903, S120904, S120905, S120907, S120999, S121001
- 연수회준비 및 진행업: S120906
- 미용학원 운영업: S120907
- 학교기숙사업/아동 가정방문교육업: S120999
- 골프연습장/골프장/수영장/볼링장/헬스클럽 등 스포츠시설 운영업: S121001
- DVD방/PC게임방/놀이방/온라인게임/인터넷게임센터 운영업: S121002
- 놀이공원/수상휴양지 운영업: S121003
- 보도업: S1234

[42류 - 과학/기술/IT서비스업 유사군 코드]
- 과학 기계기구 대여업/금속 구조측정 조사업/제품품질관리업: S174299
- 기술과제연구 조사업: S174201, S174202, S174203, S174204, S174205, S174206, S174299
- 자전거설계 및 디자인업: S121302

반드시 아래 JSON 형식으로만 응답해. 다른 텍스트는 절대 포함하지 마:
{
  "nice_class": 41,
  "class_name": "교육/연예/스포츠업",
  "similar_group_codes": [
    {"code": "S120903", "description": "과학학원/영어어학원 운영업"},
    {"code": "S120907", "description": "미용학원 운영업"}
  ],
  "matched_products": ["어학원 운영업", "미용학원 운영업"],
  "reason": "온라인 영어 교육 서비스이므로 41류 교육업에 해당합니다."
}
"""

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=API_VERSION
)

def classify_similar_group(service_description: str) -> dict:
    """
    서비스/상품 설명을 입력받아 유사군 코드를 반환
    
    Args:
        service_description: 서비스 또는 상품 설명 (예: "온라인 영어 교육 플랫폼 운영")
    
    Returns:
        dict: nice_class, similar_group_codes, reason 포함
    """
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": service_description}
        ],
        temperature=0.1  # 일관된 분류를 위해 낮게 설정
    )
    
    result_text = response.choices[0].message.content.strip()
    
    # JSON 파싱
    result_text = result_text.replace("```json", "").replace("```", "").strip()
    return json.loads(result_text)


if __name__ == "__main__":
    # 테스트
    test_cases = [
        "온라인 영어 교육 플랫폼 운영 서비스",
        "스포츠 의류 및 운동복 제조 판매",
        "모바일 앱 소프트웨어 개발 서비스"
    ]
    
    for desc in test_cases:
        print(f"\n입력: {desc}")
        result = classify_similar_group(desc)
        print(f"니스분류: {result['nice_class']}류 ({result['class_name']})")
        print(f"유사군 코드: {result['similar_group_codes']}")
        print(f"이유: {result['reason']}")
