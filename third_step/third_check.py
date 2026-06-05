import os
import json
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from first_step.viennacode_1_2 import run_first_pipeline, VIENNA_RESULT_JSON_PATH
from second_step.ocr_embedding import run_second_checker

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
        input_data: Optional[Dict[str, Any]] = None,
        first_result: Optional[Dict[str, Any]] = None,
        second_result: Optional[Dict[str, Any]] = None,
        run_full_pipeline: bool = True,
    ) -> Dict[str, Any]:
        try: 
            if run_full_pipeline:
                if not input_data:
                    raise ValueError("run_full_pipeline=True일 때는 input_data가 필요합니다.")

                # 1차 검사 실행
                first_result = run_first_pipeline(input_data)

                if not first_result or not first_result.get("saved_top100"):
                    raise ValueError("1차 검사 결과 saved_top100이 없습니다.")

                # 1차에서 저장한 top100.json을 2차 입력으로 사용
                second_result = run_second_checker(
                    input_json=str(VIENNA_RESULT_JSON_PATH),
                    save_result=True,
                )

            if not second_result:
                raise ValueError("2차 검사 결과가 없습니다.")

            trademark_name = second_result.get("trademark_name", "")
            query_ocr_text = second_result.get("query_ocr_text", "")
            service_description = second_result.get("service_description", "")
            nice_codes = second_result.get("nice_codes", [])
            vienna_codes = second_result.get("vienna_codes", [])
            similar_trademarks = second_result.get("similar_trademark", [])
            similar_group_codes = second_result.get("similar_group_codes", [])

            if not trademark_name:
                raise ValueError("2차 결과에 trademark_name 값이 없습니다.")
            
            distinctiveness_target_text = trademark_name.strip()

            prompt = self._build_prompt(
                trademark_name=trademark_name,
                distinctiveness_target_text=distinctiveness_target_text,
                service_description=service_description,
                nice_codes=nice_codes,
            )

            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "너는 상표 출원 전 사전 검토를 돕는 AI다. "
                            "오직 상표 문구의 식별력, 설명적 표현 여부, 보통명칭 여부를 분석한다. "
                            "법적 확정 판단은 하지 말고 반드시 JSON 형식으로만 답한다."
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

            distinctiveness = self._normalize_distinctiveness(parsed)

            # 2차 결과 기반 종합 유사도 계산
            similarity_assessment = self._calculate_similarity_assessment(
                similar_trademarks=similar_trademarks,
            )

            # 최종 리포트 생성
            final_report = self._build_final_report(
                trademark_name=trademark_name,
                query_ocr_text=query_ocr_text,
                service_description=service_description,
                nice_codes=nice_codes,
                similar_group_codes=similar_group_codes,
                vienna_codes=vienna_codes,
                distinctiveness=distinctiveness,
                similarity_assessment=similarity_assessment,
            )

            return {
                "status": "success",
                "input": {
                    "trademark_name": trademark_name,
                    "query_ocr_text": query_ocr_text,
                    "distinctiveness_target_text": distinctiveness_target_text,
                    "service_description": service_description,
                    "nice_codes": nice_codes,
                    "similar_group_codes": similar_group_codes,
                    "vienna_codes": vienna_codes,
                },
                "distinctiveness": distinctiveness,
                "similarity_assessment": similarity_assessment,
                "final_report": final_report,
                "similar_trademark": similar_trademarks,
            }

        except Exception as e:
            return {
                "status": "failed",
                "message": f"3차 검사 실패: {str(e)}",
            }

    def _build_prompt(
        self,
        trademark_name: str,
        distinctiveness_target_text: str,
        service_description: Optional[str],
        nice_codes: List[str],
    ) -> str:

        return f"""
다음 상표 문구의 식별력을 출원 전 사전 검토 관점에서 분석해줘.

주의:
- 법적 확정 판단을 하지 마.
- 등록 가능/불가능을 단정하지 마.
- 브랜딩, 사업모델, 마케팅 전략은 평가하지 마.
- 상품/서비스 설명과 비교했을 때 상표 문구가 설명적인지 판단해.
- "식별력이 약할 수 있음", "검토 필요", "설명적 표현 가능성"처럼 표현해.
- 반드시 JSON 형식으로만 답해.

[사용자가 입력한 상표명]
{trademark_name}

[실제 식별력 검사 대상 문구]
{distinctiveness_target_text}

[상품/서비스 설명]
{service_description or "제공되지 않음"}

[NICE 코드]
{nice_codes}

아래 JSON 형식으로만 답해.

{{
  "score": 0,
  "level": "LOW | MEDIUM | HIGH",
  "reasons": [
    "식별력 판단 이유 1",
    "식별력 판단 이유 2"
  ],
  "suggestions": [
    "개선 제안 1",
    "개선 제안 2"
  ]
}}

점수 기준:
- score는 0~100 사이 정수
- 점수가 높을수록 식별력 위험이 높음
- LOW: 식별력 위험 낮음
- MEDIUM: 식별력 위험 중간
- HIGH: 식별력 위험 높음

식별력 판단 기준:
- 보통명칭이면 위험 높음
- 상품/서비스를 직접 설명하면 위험 높음
- 지역명 + 업종명 조합이면 위험 높음
- 흔한 홍보어, 품질 표시, 기능 설명이면 위험 중간 이상
- 조어, 임의어, 은유적 표현, 독창적 결합이면 위험 낮음
"""

    def _normalize_distinctiveness(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "score": self._clamp_score(parsed.get("score", 0)),
            "level": self._normalize_level(parsed.get("level")),
            "reasons": parsed.get("reasons", []),
            "suggestions": parsed.get("suggestions", []),
        }

    def _calculate_similarity_assessment(
        self,
        similar_trademarks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        2차 검사 결과 기반 종합 유사도 계산.
        LLM이 아니라 코드로 계산해서 결과 일관성을 높임.
        """

        if not similar_trademarks:
            return {
                "overall_similarity_score": 0,
                "level": "LOW",
                "priority": "낮음",
                "risk_factors": ["유사 후보가 발견되지 않음"],
                "key_evidence": ["2차 검사 결과 유사 후보가 없습니다."],
                "top_matches": [],
            }

        top_matches = similar_trademarks[:3]
        top1 = similar_trademarks[0]

        top1_final_score = self._to_score_100(top1.get("final_score"))
        top1_image_score = self._to_score_100(top1.get("image_similarity"))
        top1_text_score = self._to_score_100(top1.get("text_similarity"))

        top3_scores = [
            self._to_score_100(item.get("final_score"))
            for item in top_matches
            if item.get("final_score") is not None
        ]

        top3_average = (
            sum(top3_scores) / len(top3_scores)
            if top3_scores
            else 0
        )

        has_matched_nice = any(
            item.get("matched_nice_codes")
            for item in similar_trademarks
        )

        has_matched_vienna = any(
            item.get("matched_vienna_codes")
            for item in similar_trademarks
        )

        # 기본 점수: Top1 70%, Top3 평균 30%
        overall_score = int((top1_final_score * 0.7) + (top3_average * 0.3))

        # 위험 요소 가산
        if has_matched_nice:
            overall_score += 8

        if has_matched_vienna:
            overall_score += 7

        if top1_text_score >= 80:
            overall_score += 5

        if top1_image_score >= 75:
            overall_score += 5

        overall_score = self._clamp_score(overall_score)

        level = self._score_to_level(overall_score)
        priority = self._score_to_priority(overall_score)

        risk_factors = self._extract_risk_factors(
            top1_text_score=top1_text_score,
            top1_image_score=top1_image_score,
            has_matched_nice=has_matched_nice,
            has_matched_vienna=has_matched_vienna,
        )

        key_evidence = self._build_key_evidence(
            top1=top1,
            top1_final_score=top1_final_score,
            top1_text_score=top1_text_score,
            top1_image_score=top1_image_score,
            has_matched_nice=has_matched_nice,
            has_matched_vienna=has_matched_vienna,
        )

        normalized_top_matches = []

        for item in top_matches:
            normalized_top_matches.append({
                "rank": item.get("rank"),
                "fileName": item.get("fileName"),
                "ocr_text": item.get("ocr_text", ""),
                "type": item.get("type"),
                "final_score": self._to_score_100(item.get("final_score")),
                "image_similarity": self._to_score_100(item.get("image_similarity")),
                "text_similarity": self._to_score_100(item.get("text_similarity")),
                "nice_codes": item.get("nice_codes", []),
                "matched_nice_codes": item.get("matched_nice_codes", []),
                "vienna_codes": item.get("vienna_codes", []),
                "matched_vienna_codes": item.get("matched_vienna_codes", []),
            })

        
        return {
            "overall_similarity_score": overall_score,
            "level": level,
            "priority": priority,
            "risk_factors": risk_factors,
            "key_evidence": key_evidence,
            "top_matches": normalized_top_matches,
        }

    def _extract_risk_factors(
        self,
        top1_text_score: int,
        top1_image_score: int,
        has_matched_nice: bool,
        has_matched_vienna: bool,
    ) -> List[str]:
        factors = []

        if top1_text_score >= 80:
            factors.append("문자 유사도 높음")
        elif top1_text_score >= 60:
            factors.append("문자 유사도 일부 존재")

        if has_matched_nice:
            factors.append("동일 또는 유사 NICE 분류 존재")

        if top1_image_score >= 75:
            factors.append("이미지 유사도 높음")
        elif top1_image_score >= 60:
            factors.append("이미지 유사도 일부 존재")

        if has_matched_vienna:
            factors.append("유사한 도형 구도 발견")

        if not factors:
            factors.append("뚜렷한 고위험 유사 요소는 제한적임")

        return factors

    def _build_key_evidence(
        self,
        top1: Dict[str, Any],
        top1_final_score: int,
        top1_text_score: int,
        top1_image_score: int,
        has_matched_nice: bool,
        has_matched_vienna: bool,
    ) -> List[str]:
        evidence = []

        evidence.append(
            f"가장 유사한 후보의 종합 유사도는 {top1_final_score}점입니다."
        )

        if top1_text_score > 0:
            evidence.append(
                f"가장 유사한 후보와의 문자 유사도는 {top1_text_score}점입니다."
            )

        if top1_image_score > 0:
            evidence.append(
                f"가장 유사한 후보와의 이미지 유사도는 {top1_image_score}점입니다."
            )

        if has_matched_nice:
            evidence.append(
                "사용자 상표와 유사 후보 사이에 동일하거나 유사한 NICE 분류가 확인되었습니다."
            )

        if has_matched_vienna:
            evidence.append(
                "비엔나 코드 기준에서 유사한 도형 요소가 확인되었습니다."
            )

        top1_ocr = top1.get("ocr_text", "")
        if top1_ocr:
            evidence.append(
                f"가장 유사한 후보의 OCR 텍스트는 '{top1_ocr}'입니다."
            )

        return evidence

    def _build_final_report(
        self,
        trademark_name: str,
        query_ocr_text: str,
        service_description: Optional[str],
        nice_codes: List[str],
        similar_group_codes: List[str],
        vienna_codes: List[str],
        distinctiveness: Dict[str, Any],
        similarity_assessment: Dict[str, Any],
    ) -> Dict[str, Any]:

        similarity_score = similarity_assessment.get("overall_similarity_score", 0)
        similarity_level = similarity_assessment.get("level", "LOW")
        priority = similarity_assessment.get("priority", "낮음")

        risk_factors = list(similarity_assessment.get("risk_factors", []))

        distinctiveness_level = distinctiveness.get("level")
        distinctiveness_score = distinctiveness.get("score", 0)

        if distinctiveness_level == "HIGH":
            risk_factors.append("상표 문구의 식별력 약화 가능성")
        elif distinctiveness_level == "MEDIUM":
            risk_factors.append("상표 문구의 설명적 표현 가능성")

        conclusion = self._build_conclusion(similarity_level)

        summary_report = (
            "본 분석 결과, 입력 상표는 일부 기존 상표와 문자 및 도형 요소에서 "
            "유사성이 확인되었다. 특히 동일하거나 유사한 니스류에 속한 등록 상표가 "
            "존재하므로 변리사의 추가 검토가 필요하다."
        )

        # 유사도가 낮고 위험 요소가 적은 경우 문장을 조금 완화
        if similarity_level == "LOW":
            summary_report = (
                "본 분석 결과, 입력 상표는 2차 검사 기준에서 기존 상표와의 고위험 유사성은 "
                "제한적으로 확인되었다. 다만 본 분석은 자동화된 사전 검토 결과이므로, "
                "최종 출원 전에는 변리사의 추가 검토가 필요하다."
            )

        return {
            "입력 상표명": trademark_name,
            "OCR 인식 문구": query_ocr_text,
            "상품/서비스 설명": service_description,
            "NICE 분류": nice_codes,
            "유사군 코드": similar_group_codes,
            "비엔나 코드": vienna_codes,

            "최종 결론": conclusion,
            "종합유사도": similarity_score,
            "우선순위": priority,

            "주요 위험 요소": risk_factors,

            "주요 근거": similarity_assessment.get("key_evidence", []),

            "식별력 검사": {
                "score": distinctiveness_score,
                "level": distinctiveness_level,
                "reasons": distinctiveness.get("reasons", []),
                "suggestions": distinctiveness.get("suggestions", []),
            },

            "분석 요약 리포트": summary_report,

            "주의 문구": (
                "본 결과는 상표 출원 전 사전 검토를 돕기 위한 참고용 분석이며, "
                "법적 등록 가능성, 거절 가능성, 침해 여부를 확정적으로 판단하지 않습니다."
            ),
        }

    def _build_conclusion(self, similarity_level: str) -> str:
        if similarity_level == "HIGH":
            return "유사성이 높음"
        if similarity_level == "MEDIUM":
            return "유사성이 일부 확인됨"
        return "유사성이 낮음"

    def _to_score_100(self, value: Any) -> int:
        """
        cosine similarity 값이 0~1이면 0~100으로 변환.
        이미 0~100이면 그대로 보정.
        """

        if value is None:
            return 0

        try:
            value = float(value)
        except Exception:
            return 0
        
        if 0 <= value <= 1:
            value = value * 100

        return self._clamp_score(value)


    def _clamp_score(self, score: Any) -> int:
        try:
            score = int(round(float(score)))
        except Exception:
            return 0

        return max(0, min(score, 100))
    
    def _normalize_level(self, level: Any) -> str:
        if not level:
            return "UNKNOWN"

        level = str(level).upper().strip()

        if level in ["LOW", "MEDIUM", "HIGH"]:
            return level

        return "UNKNOWN"

    def _score_to_level(self, score: int) -> str:
        if score >= 75:
            return "HIGH"
        if score >= 50:
            return "MEDIUM"
        return "LOW"
    
    def _score_to_priority(self, score: int) -> str:
        if score >= 85:
            return "매우 높음"
        if score >= 75:
            return "높음"
        if score >= 50:
            return "중간"
        return "낮음"
    
if __name__ == "__main__":
    checker = ThirdChecker()

    result = checker.check(
        input_data={
            "image_path": "../test/2022_79241717.jpg",
            "trademark_name": "STARBOX COFFEE",
            "service_description": "온라인 영어 교육 서비스",
            # "selected_nice_classes": [41],
        },
        run_full_pipeline=True,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))