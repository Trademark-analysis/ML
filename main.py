import json
from third_step.third_check import ThirdChecker

checker = ThirdChecker()

result = checker.check(
    input_data={
        "image_path": "test/2022_79241717.jpg",
        "trademark_name": "STAR ENGLISH",
        "service_description": "온라인 영어 교육 서비스",
        "selected_nice_classes": [41],
    },
    run_full_pipeline=True,
)

print(json.dumps(result, ensure_ascii=False, indent=2))