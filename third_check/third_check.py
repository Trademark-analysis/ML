import os
import json
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI


class ThirdChecker:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    def check(
        self,
        user_mark_text: str,
        user_nice_codes: List[str],
        user_goods_description: Optional[str] = None,
        similar_candidates_summary: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        user_mark_text:
            사용자가 출원하려는 상표명

        user_nice_codes:
            사용자 상표의 NICE 코드

        user_goods_description:
            사용자가 입력한 상품/서비스 설명 (서비스 정보 + 유사군 코드 기반)

        similar_candidates_summary:
            유사 후보 요약 정보
        """

        prompt = self._build_prompt(
            user_mark_text=user_mark_text,
            user_nice_codes=user_nice_codes,
            user_goods_description=user_goods_description,
            similar_candidates_summary=similar_candidates_summary,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "너는 상표 출원 전 사전 검토를 돕는 AI다. "
                            "법적 확정 판단이 아니라 식별력과 브랜딩 위험도를 분석한다. "
                            "반드시 JSON 형식으로만 답한다."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            parsed = json.loads(content)

            return self._normalize_result(parsed)

        except Exception as e:
            return {
                "status": "failed",
                "distinctiveness": {
                    "score": 0,
                    "level": "UNKNOWN",
                    "reasons": [f"LLM 식별력 분석 실패: {str(e)}"],
                },
                "branding_risk": {
                    "score": 0,
                    "level": "UNKNOWN",
                    "reasons": [],
                },
                "overall": {
                    "score": 0,
                    "level": "UNKNOWN",
                    "summary": "LLM 분석에 실패했습니다.",
                },
                "suggestions": [],
            }

    def _build_prompt(
        self,
        user_mark_text: str,
        user_nice_codes: List[str],
        user_goods_description: Optional[str],
        similar_candidates_summary: Optional[List[Dict[str, Any]]],
    ) -> str:

        return f"""
다음 상표를 출원 전 사전 검토 관점에서 분석해줘.

주의:
- 법적 확정 판단을 하지 마.
- 등록 가능/불가능을 단정하지 마.
- "위험 가능성", "검토 필요", "식별력이 약할 수 있음"처럼 표현해.
- 데이터셋 내 유사 후보는 참고 정보일 뿐, 선출원 확정 판단에 사용하지 마.

[사용자 상표명]
{user_mark_text}

[사용자 NICE 코드]
{user_nice_codes}

[상품/서비스 설명]
{user_goods_description or "제공되지 않음"}

[유사 후보 요약]
{similar_candidates_summary or "제공되지 않음"}

아래 JSON 형식으로만 답해.

{{
  "distinctiveness": {{
    "score": 0,
    "level": "LOW | MEDIUM | HIGH",
    "reasons": ["식별력 관련 이유"]
  }},
  "branding_risk": {{
    "score": 0,
    "level": "LOW | MEDIUM | HIGH",
    "reasons": ["브랜딩 위험 관련 이유"]
  }},
  "overall": {{
    "score": 0,
    "level": "LOW | MEDIUM | HIGH",
    "summary": "전체 요약"
  }},
  "suggestions": [
    "개선 제안 1",
    "개선 제안 2"
  ]
}}

점수 기준:
- score는 0~100 사이 정수
- 점수가 높을수록 위험도가 높음

식별력 판단 기준:
- 보통명칭이면 위험 높음
- 상품/서비스를 직접 설명하면 위험 높음
- 지역명 + 업종명 조합이면 위험 높음
- 흔한 홍보어만 있으면 위험 중간
- 조어, 은유적 표현, 독창적 결합이면 위험 낮음

브랜딩 위험 판단 기준:
- 너무 흔한 단어 조합이면 위험 높음
- 검색/기억/차별화가 어려우면 위험 높음
- 발음과 의미가 명확하면 위험 낮음
- 업종과 연결성은 있으나 직접 설명적이지 않으면 긍정적으로 평가
"""

    def _normalize_result(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        distinctiveness = parsed.get("distinctiveness", {})
        branding_risk = parsed.get("branding_risk", {})
        overall = parsed.get("overall", {})

        return {
            "status": "success",
            "distinctiveness": {
                "score": self._clamp_score(distinctiveness.get("score", 0)),
                "level": distinctiveness.get("level"),
                "reasons": distinctiveness.get("reasons", []),
            },
            "branding_risk": {
                "score": self._clamp_score(branding_risk.get("score", 0)),
                "level": branding_risk.get("level"),
                "reasons": branding_risk.get("reasons", []),
            },
            "overall": {
                "score": self._clamp_score(overall.get("score", 0)),
                "level": overall.get("level"),
                "summary": overall.get("summary", ""),
            },
            "suggestions": parsed.get("suggestions", []),
        }

    def _clamp_score(self, score: Any) -> int:
        try:
            score = int(score)
        except Exception:
            return 0

        return max(0, min(score, 100))