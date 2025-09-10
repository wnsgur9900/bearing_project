import os
import numpy as np
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from werkzeug.utils import secure_filename
import torch

from signal_utils import load_signal_any, to_mel_spectrogram, save_spectrogram_png, compute_qc_metrics, qc_warnings, detailed_frequency_analysis
from inference import load_model, predict_from_mel, predict_windows

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
SPEC_DIR = os.path.join(BASE_DIR, "static", "spectrograms")
SAMPLES_DIR = os.path.join(BASE_DIR, "static", "samples")
HISTORY_DIR = os.path.join(BASE_DIR, "history")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_resnet50_integrated.pth")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SPEC_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "bearing-demo-secret"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(MODEL_PATH, device)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")
    if not file or file.filename == "":
        flash("파일을 선택해 주세요 (.wav 또는 .csv)")
        return redirect(url_for("index"))

    try:
        sr = int(request.form.get("sr", 25600))
    except ValueError:
        sr = 25600

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    try:
        # 1) 신호 로딩
        y, used_sr = load_signal_any(save_path, target_sr=sr)
        
        # 0) QC: 길이/품질 검사 (경고만, 차단하지 않음)
        qc = compute_qc_metrics(y, used_sr)

        # 1) 동적 윈도우 설정 (짧은 신호도 분석)
        #   - 길이 기준으로 0.2s ~ 1.0s 범위에서 자동 결정
        length_sec = qc.get("length_sec", 0.0)
        win_sec = max(0.2, min(1.0, float(length_sec)))  # clamp to [0.2, 1.0]
        # hop은 윈도의 절반(최소 0.1s)
        hop_sec = max(0.1, win_sec / 2.0)

        # 2) 윈도 앙상블 예측 (fallback_if_short=True)
        ens = predict_windows(
            model, device, y, used_sr, to_mel_spectrogram,
            win_sec=win_sec, hop_sec=hop_sec, decision_th=0.7,
            fallback_if_short=True
        )
        if not ens.get("enough", False):
            flash("윈도 생성 실패: 입력 신호가 너무 짧거나 손상되었습니다.")
            return redirect(url_for("index"))

        # QC 경고 생성 (Fault 확률 포함)
        mean_p = ens["mean_fault"]
        warns = qc_warnings(qc, fault_prob=mean_p)
        
        # 상세 주파수 분석 수행
        freq_analysis = detailed_frequency_analysis(y, used_sr)

        # 3) 단일 전체 스펙 이미지 저장 (시각화용)
        mel_db = to_mel_spectrogram(y, sr=used_sr)
        spec_path = save_spectrogram_png(mel_db, SPEC_DIR)
        spec_url = url_for("static", filename=f"spectrograms/{os.path.basename(spec_path)}")

        # 4) 최종 판단 규칙(기본안) + OOD 힌트
        over_th = ens["over_th_ratio"]
        if (mean_p > 0.60) or (over_th >= 0.40):
            final_pred = "Fault"
        elif (mean_p < 0.40) and (over_th < 0.20):
            final_pred = "Normal"
        else:
            final_pred = "Suspect"

        low_conf = (ens["mean_maxp"] < 0.60) or (ens["mean_entropy"] > 0.65)

        # 분석 결과를 세션에 저장 (내보내기용)
        analysis_result = {
            "filename": filename,
            "sr": used_sr,
            "pred": final_pred,
            "p_fault": round(mean_p * 100, 2),
            "p_normal": round((1 - mean_p) * 100, 2),
            "over_th": round(over_th * 100, 1),
            "n_windows": ens["n_windows"],
            "mean_maxp": round(ens["mean_maxp"], 3),
            "mean_entropy": round(ens["mean_entropy"], 3),
            "low_conf": low_conf,
            "qc": qc,
            "warns": warns,
            "spec_url": spec_url,
            "short_mode": ens["short_mode"],
            "win_used": ens["win_sec_used"],
            "hop_used": ens["hop_sec_used"],
            "freq_analysis": freq_analysis,
            "timestamp": datetime.now().isoformat()
        }
        
        # 세션에 저장
        from flask import session
        session['latest_result'] = analysis_result
        
        # 히스토리에 저장
        save_to_history(analysis_result)

        return render_template(
            "result.html",
            filename=filename,
            sr=used_sr,
            pred=final_pred,
            p_fault=round(mean_p * 100, 2),
            p_normal=round((1 - mean_p) * 100, 2),
            over_th=round(over_th * 100, 1),
            n_windows=ens["n_windows"],
            mean_maxp=round(ens["mean_maxp"], 3),
            mean_entropy=round(ens["mean_entropy"], 3),
            low_conf=low_conf,
            qc=qc,
            warns=warns,
            spec_url=spec_url,
            short_mode=ens["short_mode"],
            win_used=ens["win_sec_used"],
            hop_used=ens["hop_sec_used"],
            freq_analysis=freq_analysis,
        )
    except Exception as e:
        flash(f"분석 실패: {e}")
        return redirect(url_for("index"))

@app.route("/manual")
def manual():
    return render_template("manual.html")

def create_sample_files():
    """샘플 데이터 파일들 생성"""
    import pandas as pd
    import librosa
    
    # 1. Normal bearing sample (CSV)
    normal_csv_path = os.path.join(SAMPLES_DIR, "normal_bearing_sample.csv")
    if not os.path.exists(normal_csv_path):
        # 정상 베어링의 진동 패턴 시뮬레이션
        t = np.linspace(0, 2, 25600*2)  # 2초, 25600Hz
        # 기본 회전 주파수와 하모닉스
        freq1, freq2 = 30, 60  # Hz
        signal = (0.5 * np.sin(2*np.pi*freq1*t) + 
                 0.3 * np.sin(2*np.pi*freq2*t) + 
                 0.1 * np.random.randn(len(t)))
        
        df = pd.DataFrame({
            'time': t,
            'vibration_x': signal + 0.05*np.random.randn(len(t)),
            'vibration_y': signal + 0.05*np.random.randn(len(t))
        })
        df.to_csv(normal_csv_path, index=False)
    
    # 2. Fault bearing sample (CSV)
    fault_csv_path = os.path.join(SAMPLES_DIR, "fault_bearing_sample.csv")
    if not os.path.exists(fault_csv_path):
        # 결함 베어링의 진동 패턴 시뮬레이션 (더 높은 주파수 성분과 충격)
        t = np.linspace(0, 2, 25600*2)
        freq1, freq2, freq3 = 30, 120, 240  # Hz
        # 충격성 신호 추가
        impulses = np.zeros_like(t)
        impulse_times = np.random.choice(len(t), size=20, replace=False)
        impulses[impulse_times] = np.random.uniform(2, 5, size=20)
        
        signal = (0.5 * np.sin(2*np.pi*freq1*t) + 
                 0.4 * np.sin(2*np.pi*freq2*t) + 
                 0.3 * np.sin(2*np.pi*freq3*t) + 
                 impulses +
                 0.2 * np.random.randn(len(t)))
        
        df = pd.DataFrame({
            'time': t,
            'vibration_x': signal + 0.1*np.random.randn(len(t)),
            'vibration_y': signal + 0.1*np.random.randn(len(t))
        })
        df.to_csv(fault_csv_path, index=False)

def save_to_history(result):
    """분석 결과를 히스토리에 저장"""
    history_file = os.path.join(HISTORY_DIR, "analysis_history.json")
    
    # 기존 히스토리 로드
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []
    
    # 새 결과 추가 (ID 부여)
    result['id'] = len(history) + 1
    history.append(result)
    
    # 최대 100개까지만 보관
    if len(history) > 100:
        history = history[-100:]
    
    # 히스토리 저장
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_history():
    """히스토리 로드"""
    history_file = os.path.join(HISTORY_DIR, "analysis_history.json")
    
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        return sorted(history, key=lambda x: x['timestamp'], reverse=True)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def get_history_item(item_id):
    """특정 히스토리 항목 조회"""
    history = load_history()
    for item in history:
        if item.get('id') == int(item_id):
            return item
    return None

@app.route("/samples")
def samples():
    """샘플 파일 목록 페이지"""
    create_sample_files()  # 샘플 파일이 없으면 생성
    
    samples = [
        {
            "name": "Normal Bearing Sample",
            "filename": "normal_bearing_sample.csv",
            "description": "정상 베어링의 진동 신호 샘플 데이터",
            "type": "CSV",
            "size": "약 1.2MB"
        },
        {
            "name": "Fault Bearing Sample", 
            "filename": "fault_bearing_sample.csv",
            "description": "결함 베어링의 진동 신호 샘플 데이터",
            "type": "CSV", 
            "size": "약 1.2MB"
        }
    ]
    
    return render_template("samples.html", samples=samples)

@app.route("/download_sample/<filename>")
def download_sample(filename):
    """샘플 파일 다운로드"""
    try:
        return send_file(
            os.path.join(SAMPLES_DIR, filename),
            as_attachment=True,
            download_name=filename
        )
    except FileNotFoundError:
        flash("요청한 샘플 파일을 찾을 수 없습니다.")
        return redirect(url_for("samples"))

@app.route("/history")
def history():
    """분석 히스토리 페이지"""
    history_list = load_history()
    return render_template("history.html", history=history_list)

@app.route("/history/<int:item_id>")
def history_detail(item_id):
    """히스토리 상세 보기"""
    item = get_history_item(item_id)
    if not item:
        flash("요청한 분석 결과를 찾을 수 없습니다.")
        return redirect(url_for("history"))
    
    # 세션에 저장 (내보내기 기능을 위해)
    from flask import session
    session['latest_result'] = item
    
    return render_template(
        "result.html",
        filename=item['filename'],
        sr=item['sr'],
        pred=item['pred'],
        p_fault=item['p_fault'],
        p_normal=item['p_normal'],
        over_th=item['over_th'],
        n_windows=item['n_windows'],
        mean_maxp=item['mean_maxp'],
        mean_entropy=item['mean_entropy'],
        low_conf=item['low_conf'],
        qc=item['qc'],
        warns=item['warns'],
        spec_url=item['spec_url'],
        short_mode=item['short_mode'],
        win_used=item['win_used'],
        hop_used=item['hop_used'],
        freq_analysis=item.get('freq_analysis', {}),
        from_history=True,
        history_timestamp=item['timestamp']
    )

@app.route("/history/delete/<int:item_id>")
def delete_history_item(item_id):
    """히스토리 항목 삭제"""
    history_file = os.path.join(HISTORY_DIR, "analysis_history.json")
    
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        # 해당 항목 제거
        history = [item for item in history if item.get('id') != item_id]
        
        # 저장
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        flash("히스토리 항목이 삭제되었습니다.")
    except Exception as e:
        flash(f"삭제 중 오류가 발생했습니다: {e}")
    
    return redirect(url_for("history"))


@app.route("/healthz")
def healthz():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)