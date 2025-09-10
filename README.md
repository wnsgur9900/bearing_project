# Bearing Fault Analyzer (Web Demo)

## 1. 설치
```bash
conda create -n bearing_web python=3.11 -y
conda activate bearing_web
pip install -r requirements.txt
```

## 2. 모델 파일

`models/best_resnet50_integrated.pth`를 배치하세요.

## 3. 실행
```bash
python app.py
```

브라우저에서 http://localhost:5000 접속.

## 4. 업로드

- .wav 오디오(자동 리샘플링) 또는 .csv(마지막 1~2 컬럼을 진동 신호로 사용)
- SR 입력 비워두면 기본 25600

## 5. 기능

- 진동 신호 업로드 (.wav, .csv)
- 멜 스펙트로그램 생성
- ResNet50 기반 결함 분류 (Normal/Fault)
- 확률 및 시각화 결과 제공
