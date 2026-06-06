from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles 
import os
import shutil
from typing import List, Any
import re
import json
from pydantic import BaseModel
from ML.first_step.nicecode_1_1 import (
    classify_first_step,
    classify_second_step,
)

# 3차 검사 통합 클래스 임포트
from ML.third_step.third_check import ThirdChecker

app = FastAPI(title="Trademark Analysis Integrated ML Server")

# 1. 404 에러 원천 차단: 브라우저가 접근할 수 있는 공식 static 폴더 지정
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# 2. FastAPI에 /static 경로로 들어오는 요청을 static 폴더와 매핑
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ocr_embedding.py가 참조하는 파이프라인 연산용 임시 폴더 유지
UPLOAD_DIR = "./test" 

class ClassificationRequest(BaseModel):
    trademarkName: str = ""
    serviceDescription: str


class SimilarGroupRequest(BaseModel):
    trademarkName: str = ""
    serviceDescription: str
    selectedNiceClasses: List[int]

def parse_selected_nice_classes(value: Any) -> List[int]:
    """
    BE에서 넘어온 selectedNiceClasses를 int 리스트로 변환한다.

    처리 가능:
    - ["제25류", "제42류"]
    - ['["제25류","제42류"]']
    - ["25", "42"]
    - [25, 42]
    """

    if value is None:
        return []

    raw_items = []

    if isinstance(value, list):
        for item in value:
            if item is None:
                continue

            if isinstance(item, str):
                text = item.strip()

                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        raw_items.extend(parsed)
                    else:
                        raw_items.append(parsed)
                except json.JSONDecodeError:
                    raw_items.append(text)
            else:
                raw_items.append(item)

    elif isinstance(value, str):
        text = value.strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                raw_items.extend(parsed)
            else:
                raw_items.append(parsed)
        except json.JSONDecodeError:
            raw_items.append(text)

    else:
        raw_items.append(value)

    selected = []

    for item in raw_items:
        nums = re.findall(r"\d+", str(item))

        for num in nums:
            nice_class = int(num)

            # 현재 ML 코드가 허용하는 니스분류만 통과
            if nice_class in {25, 41, 42} and nice_class not in selected:
                selected.append(nice_class)

    return selected


@app.post("/api/ml/classification")
async def recommend_classification(request: ClassificationRequest):
    try:
        result = classify_first_step(
            image="classification-only",
            service_description=request.serviceDescription,
        )

        candidates = result.get("nice_class_candidates", [])

        response = []

        for item in candidates:
            nice_class = item.get("nice_class")
            class_name = item.get("class_name", "")

            response.append({
                "code": f"제{nice_class}류",
                "value": nice_class,
                "description": class_name,
                "confidence": item.get("confidence", "")
            })

        return JSONResponse(status_code=200, content=response)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "message": f"니스분류 추천 실패: {str(e)}"
            }
        )
@app.post("/api/ml/similar-groups")
async def recommend_similar_groups(request: SimilarGroupRequest):
    try:
        result = classify_second_step(
            image="classification-only",
            service_description=request.serviceDescription,
            selected_nice_classes=request.selectedNiceClasses,
        )

        return JSONResponse(
            status_code=200,
            content={
                "trademarkName": request.trademarkName,
                "nice_codes": result.get("nice_codes", []),
                "similar_group_codes": result.get("similar_group_codes", [])
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "message": f"유사군 코드 추천 실패: {str(e)}"
            }
        )

@app.post("/api/ml/analyze")
async def analyze_trademark(
    trademarkName: str = Form(...),          
    serviceDescription: str = Form(...),     
    selectedNiceClasses: List[str] = Form(...), 
    image: UploadFile = File(...)            
):
    print(f"=== [BE -> ML] 분석 요청 수신: {trademarkName} ===")
    
    # 파이프라인용 폴더 생성 및 이미지 저장
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    temp_image_path = os.path.join(UPLOAD_DIR, image.filename)
    
    # 3. [사진 깨짐 해결]: 브라우저 서빙용 static 폴더에도 파일을 카피해서 영구 보존
    static_image_path = os.path.join(STATIC_DIR, image.filename)
    
    # 멀티파트 파일 읽기 수순
    file_content = await image.read()
    
    # 연산용 test 폴더에 쓰기
    with open(temp_image_path, "wb") as buffer:
        buffer.write(file_content)
        
    # 서빙용 static 폴더에 쓰기 (404 방지)
    with open(static_image_path, "wb") as buffer:
        buffer.write(file_content)
        
    try:
        # 4. [유사 후보 0건 해결]: "제30류" 형태로 들어온 배열에서 숫자만 쏙 추출 ("제30류" -> "30")
        cleaned_nice_classes = parse_selected_nice_classes(selectedNiceClasses)
                
        print(f"🔍 원본 니스 코드: {selectedNiceClasses} 정제된 니스 코드: {cleaned_nice_classes}")

        # 3차 검사기 인스턴스 생성 및 파이프라인 가동
        checker = ThirdChecker()
        
        # 정제된 니스분류 데이터(cleaned_nice_classes)를 인풋 데이터에 주입
        input_data = {
            "image_path": temp_image_path,
            "trademark_name": trademarkName,
            "service_description": serviceDescription,
            "selected_nice_classes": cleaned_nice_classes  # 정제된 숫자 리스트 주입
        }
        
        print("▶ 1차 -> 2차 -> 3차 통합 상표 분석 파이프라인 가동...")
        final_result = checker.check(input_data=input_data, run_full_pipeline=True)
        
        print("=== 1, 2, 3차 통합 분석 완료 및 결과 반환 ===")
        if final_result.get("status") != "success":
            return JSONResponse(status_code=500, content=final_result)

        user_image_url = f"http://localhost:8000/static/{image.filename}"

        if final_result.get("status") != "success":
            return JSONResponse(status_code=500, content=final_result)

        input_data_for_front = final_result.get("input", {})
        input_data_for_front["imageUrl"] = user_image_url

        response = {
            "status": "success",
            "input": input_data_for_front,

            "final_report": final_result.get("final_report", {}),
            "similarity_assessment": final_result.get("similarity_assessment", {}),
            "distinctiveness": final_result.get("distinctiveness", {}),
            "similar_trademark": final_result.get("similar_trademark", []),
            "distinctiveness_score": final_result.get("final_report", {})
                .get("식별력 검사", {})
                .get("score"),
            "report": final_result.get("final_report", {}),
            "candidates": final_result.get("similar_trademark", []),
            "imageUrl": user_image_url,
        }

        return JSONResponse(status_code=200, content=response)

    except Exception as e:
        print(f"파이프라인 구동 중 예외 터짐: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"status": "failed", "message": f"ML 서버 에러: {str(e)}"}
        )
        
    finally:
        # 5. 자원 관리: 연산용 임시 test 폴더 파일만 지움
        # static 폴더의 이미지는 브라우저 렌더링용으로 살아있기 때문에 404가 나지 않음
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)