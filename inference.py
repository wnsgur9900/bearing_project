
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from signal_utils import sliding_windows

# 학습과 동일한 transform
infer_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

def load_model(model_path: str, device: torch.device):
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model

def pil_from_mel(mel_db: np.ndarray) -> Image.Image:
    # mel_db를 0~255로 스케일 후 PIL 이미지(그레이스케일)로
    m = mel_db - mel_db.min()
    if m.max() > 0:
        m = m / m.max()
    m = (m * 255).astype(np.uint8)
    # (H, W) -> Gray 'L'
    return Image.fromarray(m, mode="L")

@torch.no_grad()
def predict_from_mel(model, device, mel_db: np.ndarray):
    pil_img = pil_from_mel(mel_db)  # Grayscale
    x = infer_transform(pil_img).unsqueeze(0).to(device)  # [1,1,224,224]
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()
    pred_idx = int(probs.argmax())
    classes = ["Normal", "Fault"]
    return {
        "pred_class": classes[pred_idx],
        "prob_normal": float(probs[0]),
        "prob_fault": float(probs[1]),
    }

def softmax_entropy_binary(p_fault):
    """이진 분류의 소프트맥스 엔트로피 계산"""
    p1 = np.clip(p_fault, 1e-9, 1 - 1e-9)
    p0 = 1.0 - p1
    # 엔트로피(비트) 표시를 원하면 / np.log(2) 추가
    return float(- (p0 * np.log(p0) + p1 * np.log(p1)))

@torch.no_grad()
def predict_windows(model, device, y, sr, to_mel_fn,
                    win_sec=1.0, hop_sec=0.5, decision_th=0.7,
                    fallback_if_short=True):
    """
    윈도 앙상블 예측.
    - 길이가 짧아 윈도 생성이 안 되면, fallback_if_short=True일 때 전체 신호 1윈도로 예측.
    - 반환 dict에 short_mode, win_sec_used, hop_sec_used 포함.
    """
    from signal_utils import sliding_windows  # keep local import to avoid cycles
    import numpy as np

    # 윈도 생성
    ws = sliding_windows(y, sr, win_sec=win_sec, hop_sec=hop_sec)
    short_mode = False
    if not ws:
        if fallback_if_short:
            ws = [(0, len(y))]  # 전체 신호 1윈도
            short_mode = True
        else:
            return {"enough": False}

    probs, maxps, ents = [], [], []

    def softmax_entropy_binary(p_fault):
        p1 = np.clip(p_fault, 1e-9, 1 - 1e-9)
        p0 = 1.0 - p1
        return float(- (p0 * np.log(p0) + p1 * np.log(p1)))

    for s, e in ws:
        mel_db = to_mel_fn(y[s:e], sr=sr)
        out = predict_from_mel(model, device, mel_db)
        p1 = out["prob_fault"]
        probs.append(p1)
        maxps.append(max(p1, 1.0 - p1))
        ents.append(softmax_entropy_binary(p1))

    probs = np.array(probs)
    maxps = np.array(maxps)
    ents  = np.array(ents)

    return {
        "enough": True,
        "short_mode": short_mode,
        "win_sec_used": float(win_sec),
        "hop_sec_used": float(hop_sec),
        "n_windows": len(ws),
        "mean_fault": float(probs.mean()),
        "std_fault": float(probs.std()),
        "over_th_ratio": float((probs >= decision_th).mean()),
        "mean_maxp": float(maxps.mean()),
        "mean_entropy": float(ents.mean()),
        "per_window_probs": probs.tolist(),
    }
