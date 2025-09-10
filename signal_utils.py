import os
import uuid
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _detect_sep(sample_line: str) -> str:
    if ";" in sample_line:
        return ";"
    if "," in sample_line:
        return ","
    if "\t" in sample_line:
        return "\t"
    return ","  # fallback

def load_wav(path: str, target_sr: int = 25600):
    y, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y.astype(np.float32), sr

def load_csv_signal(path: str, target_sr: int = 25600):
    # 간단히 1채널 시그널로 사용 (마지막 2개 컬럼이면 평균)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sep = _detect_sep(f.readline())
    
    # 첫 번째 행이 헤더인지 확인
    df = pd.read_csv(path, sep=sep, header=None, engine="python")
    
    # 첫 번째 행이 숫자가 아닌 경우 헤더로 간주하고 제거
    try:
        # 첫 번째 행의 첫 번째 컬럼이 숫자로 변환 가능한지 확인
        pd.to_numeric(df.iloc[0, 0], errors='raise')
        # 숫자로 변환 가능하면 헤더가 없음
        data_start = 0
    except (ValueError, TypeError):
        # 숫자로 변환 불가능하면 헤더가 있음
        data_start = 1
        df = df.iloc[1:]  # 첫 번째 행(헤더) 제거
    
    if df.shape[1] >= 2:
        sig = df.iloc[:, -2:].astype(float).mean(axis=1).values
    else:
        sig = df.iloc[:, -1].astype(float).values
    return sig.astype(np.float32), target_sr

def to_mel_spectrogram(y: np.ndarray, sr: int = 25600, n_fft=1024, hop_length=512, n_mels=128):
    if y.size < n_fft:
        # zero-pad
        pad = n_fft - y.size
        y = np.pad(y, (0, pad), mode="constant")
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                         hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db  # (n_mels, time)

def save_spectrogram_png(mel_db: np.ndarray, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{uuid.uuid4().hex}.png"
    out_path = os.path.join(out_dir, fname)

    plt.figure(figsize=(3, 3))
    librosa.display.specshow(mel_db, y_axis='mel', x_axis='time', cmap="magma")
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    return out_path

def load_signal_any(path: str, target_sr: int = 25600):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        return load_wav(path, target_sr=target_sr)
    elif ext == ".csv":
        return load_csv_signal(path, target_sr=target_sr)
    else:
        # 시도: wav로 로드해보고 실패 시 csv
        try:
            return load_wav(path, target_sr=target_sr)
        except Exception:
            return load_csv_signal(path, target_sr=target_sr)

# QC 지표 및 윈도 함수들
def sliding_windows(y, sr, win_sec=1.0, hop_sec=0.5):
    """슬라이딩 윈도우 생성"""
    win = int(sr * win_sec)
    hop = int(sr * hop_sec)
    if len(y) < win:
        return []
    starts = np.arange(0, len(y) - win + 1, hop)
    return [(int(s), int(s + win)) for s in starts]

def _percentiles(y):
    """신호의 하위/상위 퍼센타일 계산"""
    if len(y) < 10:
        return (0.0, 0.0)
    return np.percentile(y, [0.1, 99.9])

def compute_qc_metrics(y, sr):
    """QC 지표 계산"""
    eps = 1e-12
    length_sec = float(len(y) / sr) if sr > 0 else 0.0
    rms = float(np.sqrt(np.mean(y**2))) if len(y) else 0.0
    mean = float(np.mean(y)) if len(y) else 0.0
    dc_offset = mean
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    crest = float(peak / (rms + eps)) if rms > 0 else 0.0
    lo, hi = _percentiles(y)
    eps_clip = max(1e-6, 0.001 * (hi - lo))
    clip_rate = float(((y >= hi - eps_clip) | (y <= lo + eps_clip)).mean()) if hi > lo else 0.0
    zcr = float(((y[:-1] * y[1:]) < 0).mean()) if len(y) > 1 else 0.0

    # 주파수 특징 일부
    try:
        centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    except Exception:
        centroid = 0.0
    try:
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))**2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        half = 0.4 * (sr / 2.0)
        low = S[(freqs <= half), :].sum()
        high = S[(freqs > half), :].sum()
        ber = float(high / (low + eps))
    except Exception:
        ber = 0.0

    return dict(
        length_sec=length_sec, rms=rms, dc_offset=dc_offset, crest_factor=crest,
        clipping_rate=clip_rate, zcr=zcr, spectral_centroid=centroid, band_energy_ratio=ber
    )

def qc_warnings(m, fault_prob=None):
    """QC 경고 생성"""
    warns = []
    # 심각
    if m['length_sec'] < 1.0: 
        warns.append(("error", "길이 1초 미만 (데이터 부족)"))
    if m['rms'] < 1e-4:      
        warns.append(("error", "신호 레벨 매우 낮음 (무음/센서 문제 의심)"))
    # 경고
    if abs(m['dc_offset']) > 0.1 * max(m['rms'], 1e-12):
        warns.append(("warn", "DC 오프셋 과다 (센서/프리앰프 오프셋 보정 필요)"))
    if m['clipping_rate'] > 0.01:
        warns.append(("warn", "클리핑/포화 의심 (스케일 조정 필요)"))
    # 참고
    if m['crest_factor'] > 15 or m['crest_factor'] < 1.2:
        warns.append(("info", f"Crest factor 비정상 ({m['crest_factor']:.1f})"))
    if m['zcr'] < 0.005:
        warns.append(("info", "Zero-crossing rate 낮음 (저주파 위주/정지 구간 가능)"))
    
    # 베어링 결함 징후 체크 (AI 예측 결과가 있을 때)
    if fault_prob is not None:
        if fault_prob > 0.8:
            warns.append(("error", f"베어링 결함 의심 높음 (Fault 확률: {fault_prob*100:.1f}%)"))
        elif fault_prob > 0.6:
            warns.append(("warn", f"베어링 결함 가능성 (Fault 확률: {fault_prob*100:.1f}%)"))
        elif fault_prob < 0.2:
            warns.append(("info", f"베어링 상태 양호 (Fault 확률: {fault_prob*100:.1f}%)"))
    
    return warns

def detailed_frequency_analysis(y, sr):
    """상세 주파수 분석 수행"""
    import scipy.signal as signal
    
    # 1. FFT 스펙트럼 계산
    fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    magnitude = np.abs(fft)
    power = magnitude ** 2
    
    # 2. 주요 주파수 피크 찾기
    peaks, properties = signal.find_peaks(
        magnitude, 
        height=np.max(magnitude) * 0.1,  # 최대값의 10% 이상
        distance=int(len(magnitude) * 0.01)  # 최소 거리
    )
    
    peak_freqs = freqs[peaks]
    peak_magnitudes = magnitude[peaks]
    
    # 상위 10개 피크만 선택
    if len(peaks) > 10:
        top_indices = np.argsort(peak_magnitudes)[-10:]
        peak_freqs = peak_freqs[top_indices]
        peak_magnitudes = peak_magnitudes[top_indices]
    
    # 3. 주파수 대역별 에너지 분석
    bands = {
        'Very Low (0-50 Hz)': (0, 50),
        'Low (50-200 Hz)': (50, 200),
        'Mid (200-1000 Hz)': (200, 1000),
        'High (1000-5000 Hz)': (1000, 5000),
        'Very High (5000+ Hz)': (5000, sr/2)
    }
    
    band_energies = {}
    total_energy = np.sum(power)
    
    for band_name, (f_low, f_high) in bands.items():
        mask = (freqs >= f_low) & (freqs <= f_high)
        band_energy = np.sum(power[mask])
        band_energies[band_name] = {
            'energy': float(band_energy),
            'percentage': float(band_energy / total_energy * 100) if total_energy > 0 else 0
        }
    
    # 4. 통계적 특성
    # 스펙트럼 중심 주파수
    spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
    
    # 스펙트럼 분산 (spread)
    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / np.sum(magnitude)) if np.sum(magnitude) > 0 else 0
    
    # 스펙트럼 비대칭도 (skewness)
    spectral_skewness = np.sum(((freqs - spectral_centroid) ** 3) * magnitude) / (np.sum(magnitude) * spectral_spread ** 3) if spectral_spread > 0 else 0
    
    # 스펙트럼 첨도 (kurtosis)
    spectral_kurtosis = np.sum(((freqs - spectral_centroid) ** 4) * magnitude) / (np.sum(magnitude) * spectral_spread ** 4) if spectral_spread > 0 else 0
    
    # 5. 하모닉 분석 (기본 주파수와 하모닉스 찾기)
    harmonics = []
    if len(peak_freqs) > 0:
        # 가장 강한 피크를 기본 주파수로 가정
        fundamental_idx = np.argmax(peak_magnitudes)
        fundamental_freq = peak_freqs[fundamental_idx]
        
        # 하모닉스 찾기 (2배, 3배, 4배, 5배)
        for harmonic_order in [2, 3, 4, 5]:
            target_freq = fundamental_freq * harmonic_order
            # 오차 범위 내에서 하모닉 찾기
            tolerance = fundamental_freq * 0.1  # 10% 오차 허용
            
            for i, freq in enumerate(peak_freqs):
                if abs(freq - target_freq) < tolerance:
                    harmonics.append({
                        'order': harmonic_order,
                        'frequency': float(freq),
                        'magnitude': float(peak_magnitudes[i])
                    })
                    break
    
    return {
        'spectrum': {
            'frequencies': freqs[:len(freqs)//10].tolist(),  # 데이터 크기 줄이기
            'magnitudes': magnitude[:len(magnitude)//10].tolist(),
            'power': power[:len(power)//10].tolist()
        },
        'peaks': {
            'frequencies': peak_freqs.tolist(),
            'magnitudes': peak_magnitudes.tolist()
        },
        'band_energies': band_energies,
        'statistics': {
            'spectral_centroid': float(spectral_centroid),
            'spectral_spread': float(spectral_spread),
            'spectral_skewness': float(spectral_skewness),
            'spectral_kurtosis': float(spectral_kurtosis)
        },
        'harmonics': harmonics
    }