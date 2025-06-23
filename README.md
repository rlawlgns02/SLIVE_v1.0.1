# SLIVE_v1.0.1
데이터셋 구조 등록 &amp; 샘플 데이터 학습 하기

# Sign Language Translator

이 프로젝트는 웹캠을 이용해 손 동작(수어)을 인식하고, 이를 단어 또는 문장으로 번역하는 AI 기반 수어 번역기입니다.

## 주요 기능
- Mediapipe를 활용한 손 키포인트 추출
- LSTM 기반 수어 단어 분류 모델
- Seq2Seq 기반 문장 번역 모델
- 실시간 웹캠 수어 인식 및 음성 변환(TTS)
- Streamlit/Gradio를 통한 간단한 웹 UI

## 폴더 구조
```## 폴더 구조
SLIVE/
├── requirements.txt       # 프로젝트에 필요한 라이브러리 목록
├── 1_data/                # 데이터 관련 폴더
│   ├── New_sample/        # 새로운 샘플 데이터
│   │   ├── 원천데이터/    # 원본 데이터
│   │   └── LabelData/     # 라벨링된 데이터
│   ├── notebooks/         # 데이터 분석 및 실험 노트북
│   │   └── 디렉토리 생성용도.txt
│   ├── processed/         # 전처리된 데이터
│   │   ├── numpyView.py   # NPY 파일 확인 스크립트
│   │   ├── keypoints/     # 키포인트 데이터
│   │   └── sequence_data/ # 시퀀스 데이터
│   ├── raw/               # 원본 데이터 저장소
│   │   └── 디렉토리 생성용도.txt
│   └── utils/             # 데이터 처리 유틸리티
│       └── convert_json_to_sequence.py
├── 2_models/              # 모델 관련 폴더
│   ├── hand_tracking/     # 손 키포인트 추출 코드
│   ├── seq2seq_translator/ # 문장 번역 모델
│   │   └── seq2seq.py
│   ├── utils/             # 모델 관련 유틸리티
│   └── word_classifier/   # 단어 분류 모델
├── 3_app/                 # 실시간 추론 앱
│   ├── lstm_model.py      # LSTM 모델 코드
│   ├── realtime_infer.py  # 실시간 추론 코드
│   ├── __pycache__/       # 캐시 파일
│   └── ui_components/     # UI 관련 코드
├── 4_training/            # 모델 학습 코드
│   ├── lstm_model.py      # LSTM 모델 코드
│   ├── train_word_model.py # 단어 분류 모델 학습 코드
│   └── __pycache__/       # 캐시 파일
├── 5_checkpoints/         # 학습된 모델 저장소
│   ├── 디렉토리 생성용도.txt
│   └── word_model.pth     # 학습된 단어 분류 모델
├── 6_tests/               # 테스트 코드
│   └── 디렉토리 생성용도.txt
└── README.md              # 프로젝트 설명 파일
```
```

# Sign Language Translator

이 프로젝트는 웹캠을 이용해 손 동작(수어)을 인식하고, 이를 단어 또는 문장으로 번역하는 AI 기반 수어 번역기입니다.

---

## 코드 파일 목록

### 1_data\processed\numpyView.py
# npy파일을 컴퓨터에서 볼수 있도록 변환해 주는 코드
```python
import numpy as np
# NPY 파일에서 데이터 로드
loaded_array = np.load('1_data/processed/keypoints/')
print(loaded_array)
```

### 1_data\utils\convert_json_to_sequence.py
라벨링 데이터셋을
```python
import os
import json
import numpy as np

SRC_DIR = "1_data/New_sample/LabelData/REAL/WORD/01_real_word_keypoint"
DST_DIR = "1_data/processed/sequence_data"

os.makedirs(DST_DIR, exist_ok=True)

for label in os.listdir(SRC_DIR):
    label_dir = os.path.join(SRC_DIR, label)
    if not os.path.isdir(label_dir):
        continue
    save_label_dir = os.path.join(DST_DIR, label)
    os.makedirs(save_label_dir, exist_ok=True)
    for seq_folder in os.listdir(label_dir):
        seq_dir = os.path.join(label_dir, seq_folder)
        if not os.path.isdir(seq_dir):
            continue
        keypoints_seq = []
        files = sorted(os.listdir(seq_dir))
        for file in files:
            if not file.endswith(".json"):
                continue
            with open(os.path.join(seq_dir, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                # 예시: hand_left_keypoints_2d 또는 pose_keypoints_2d 등에서 21*3=63개 추출
                # 아래는 hand_left_keypoints_2d 사용 예시
                kps = data["people"]["hand_left_keypoints_2d"][:63]
                keypoints_seq.append(kps)
        if keypoints_seq:
            keypoints_seq = np.array(keypoints_seq)  # (프레임수, 63)
            np.save(os.path.join(save_label_dir, seq_folder + ".npy"), keypoints_seq)
```

### 2_models\hand_tracking\extract_keypoints.py
```python
import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=6)  # max_num_hands=2로 변경
mp_draw = mp.solutions.drawing_utils

SAVE_DIR = '1_data/processed/keypoints'
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
print("웹캠이 켜졌습니다. 손을 화면에 보여주세요.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):  # enumerate로 인덱스 추가
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.append([lm.x, lm.y, lm.z])
            keypoints = np.array(keypoints).flatten()

            filename = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_hand{idx}.npy"  # 손 인덱스 추가
            np.save(os.path.join(SAVE_DIR, filename), keypoints)

    cv2.imshow('Hand Keypoints', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 2_models\seq2seq_translator\seq2seq.py
```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        hidden, cell = self.encoder(src)
        outputs = []
        input = trg[:, 0].unsqueeze(1)
        for t in range(1, trg.size(1)):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs.append(output.unsqueeze(1))
            input = output.argmax(1).unsqueeze(1)
        return torch.cat(outputs, dim=1)
```

### 2_models\word_classifier\lstm_model.py
```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=100):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out
```

### 3_app\lstm_model.py
```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=100):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out
```

### 3_app\realtime_infer.py
```python
import cv2
import torch
import numpy as np
from lstm_model import LSTMClassifier
from gtts import gTTS
import os
import mediapipe as mp

model = LSTMClassifier()
model.load_state_dict(torch.load("5_checkpoints/word_model.pth"))
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print("웹캠을 켜고 수화를 해보세요. 'q' 누르면 종료")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.append([lm.x, lm.y, lm.z])
            keypoints = np.array(keypoints).flatten()
            x = torch.tensor(keypoints).float().view(1, 1, 63)
            pred = model(x)
            label = torch.argmax(pred, dim=1).item()
            word = f"단어{label}"
            cv2.putText(frame, word, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            tts = gTTS(text=word, lang='ko')
            tts.save("temp.mp3")
            os.system("start temp.mp3")

    cv2.imshow('Sign Language Translator', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 4_training\train_word_model.py
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from lstm_model import LSTMClassifier
import numpy as np
import os

class SignDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.labels = []
        self.label_map = {}
        for idx, label in enumerate(sorted(os.listdir(data_dir))):
            self.label_map[idx] = label
            for file in os.listdir(os.path.join(data_dir, label)):
                keypoints = np.load(os.path.join(data_dir, label, file))
                self.data.append(keypoints)
                self.labels.append(idx)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx]).float().view(1, -1, 63)
        y = torch.tensor(self.labels[idx]).long()
        return x, y

    def __len__(self):
        return len(self.data)

dataset = SignDataset("1_data/processed/sequence_data")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = LSTMClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for x, y in dataloader:
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"{epoch+1} epoch loss: {loss.item():.4f}")

torch.save(model.state_dict(), "5_checkpoints/word_model.pth")
```

---

## 실행 방법

1. **필수 라이브러리 설치**
   ```
   pip install -r requirements.txt
   ```

2. **데이터 전처리**
   - `1_data/utils/convert_json_to_sequence.py` 실행

3. **모델 학습**
   - 단어 분류: `python 4_training/train_word_model.py`
   - (필요시) 문장 번역: `2_models/seq2seq_translator/seq2seq.py` 참고

4. **실시간 추론**
   - `python 3_app/realtime_infer.py`

## 참고
- 손 키포인트 추출: Mediapipe
- 음성 변환: gTTS
- 모델: PyTorch 기반 LSTM/Seq2Seq

## 라이선스
본 프로젝트는 연구 및 교육 목적입니다.
