"""
asl_realtime.py

Usage:
    python asl_realtime.py collect   -> start data collection (saves dataset.csv)
    python asl_realtime.py train     -> trains model from dataset.csv (saves model.joblib)
    python asl_realtime.py predict   -> run realtime prediction using model.joblib

Requirements:
    pip install opencv-python mediapipe numpy pandas scikit-learn joblib

Notes:
 - Press letters A-Z while in collect mode to label samples.
 - Press SPACE to toggle recording ON/OFF while collecting.
 - Press 'q' to quit any mode.
 - J key collects samples for SPACE gesture.
 - J and Z are dynamic — a simple heuristic is included for them but expect lower reliability.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import sys
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib

ENABLE_DYNAMIC_JZ = True
DATA_CSV = "dataset.csv"
MODEL_FILE = "model.joblib"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_landmarks(hand_landmarks):
    """
    Convert mediapipe hand_landmarks to a normalized flat vector.
    Normalization: translate so wrist is (0,0), then divide by max distance among landmarks.
    Returns 63-length vector (21 points * 3 coords).
    """
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z])
    arr = np.array(coords)
    origin = arr[0].copy()
    arr -= origin
    max_val = np.max(np.abs(arr))
    if max_val > 0:
        arr /= max_val
    return arr.flatten()


def collect_mode():
    print("COLLECT MODE - Instructions:")
    print(" - Press letter key (a-z) to set current label.")
    print(" - Press SPACE to start/stop recording samples for current label.")
    print(" - Press 'q' to quit.")
    print(" - Press 'j' to collect SPACE samples.")
    print(" - Data saved to:", DATA_CSV)

    label = None
    recording = False
    rows = []
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        last_save_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera error")
                break
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(img_rgb)
            display = frame.copy()

            if res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(display, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            info = f"Label: {label} | Recording: {recording}"
            cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Collect - Press letter keys", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == 32:  # space toggles recording
                recording = not recording
                print("Recording:", recording)
                time.sleep(0.2)

            # set label if letter pressed
            if 97 <= key <= 122:  # a-z
                if chr(key).lower() == 'j':
                    label = "SPACE"
                    print("Current label set to SPACE (for blank/space gesture)")
                else:
                    label = chr(key).upper()
                    print("Current label set to", label)
                time.sleep(0.15)

            # capture samples
            if recording and res.multi_hand_landmarks and label is not None:
                vec = extract_landmarks(res.multi_hand_landmarks[0])
                rows.append(np.concatenate(([label], vec)))

                # flush data periodically
                if time.time() - last_save_time > 1.0 and len(rows) >= 20:
                    df_new = pd.DataFrame(rows)
                    cols = ["label"] + [f"f{i}" for i in range(vec.size)]
                    df_new.columns = cols
                    if not os.path.exists(DATA_CSV):
                        df_new.to_csv(DATA_CSV, index=False)
                    else:
                        df_new.to_csv(DATA_CSV, index=False, mode='a', header=False)
                    print(f"Saved {len(rows)} rows to {DATA_CSV}")
                    rows = []
                    last_save_time = time.time()

        # final save
        if rows:
            df_new = pd.DataFrame(rows)
            df_new.columns = ["label"] + [f"f{i}" for i in range(rows[0].size - 1)]
            if not os.path.exists(DATA_CSV):
                df_new.to_csv(DATA_CSV, index=False)
            else:
                df_new.to_csv(DATA_CSV, index=False, mode='a', header=False)
            print(f"Saved final {len(rows)} rows to {DATA_CSV}")

    cap.release()
    cv2.destroyAllWindows()


def train_mode():
    if not os.path.exists(DATA_CSV):
        print("No dataset found. Run collect mode first.")
        return
    df = pd.read_csv(DATA_CSV)
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    clf = KNeighborsClassifier(n_neighbors=5)
    print("Training KNN...")
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"Validation accuracy: {acc * 100:.2f}% (depends on your samples)")
    joblib.dump(clf, MODEL_FILE)
    print("Saved model to", MODEL_FILE)


def simple_dynamic_detection(recent_points):
    if len(recent_points) < 10:
        return None
    arr = np.array(recent_points)
    dx = arr[:, 0] - arr[0, 0]
    dy = arr[:, 1] - arr[0, 1]
    total_dist = np.linalg.norm(arr[-1] - arr[0])
    if total_dist < 60:
        return None
    dx_changes = np.sum(np.abs(np.diff(np.sign(np.diff(arr[:, 0])))) > 0)
    dy_changes = np.sum(np.abs(np.diff(np.sign(np.diff(arr[:, 1])))) > 0)
    curvature = np.std(dy) / (np.std(dx) + 1e-5)
    if dx_changes >= 1 and curvature > 1.5:
        return 'J'
    if dx_changes >= 2 and dy_changes >= 1 and curvature < 1.2:
        return 'Z'
    return None


def predict_mode():
    if not os.path.exists(MODEL_FILE):
        print("Model not found. Run train mode first.")
        return

    clf = joblib.load(MODEL_FILE)
    cap = cv2.VideoCapture(0)
    recent_index_tip = []

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        sentence = ""
        last_pred = ""
        stable_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera error.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(img_rgb)
            pred_text = ""

            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                vec = extract_landmarks(lm).reshape(1, -1)
                label = clf.predict(vec)[0]
                pred_text = label

                index_tip = lm.landmark[8]
                px = int(index_tip.x * w)
                py = int(index_tip.y * h)
                recent_index_tip.append((px, py))
                if len(recent_index_tip) > 15:
                    recent_index_tip.pop(0)

                if ENABLE_DYNAMIC_JZ:
                    dyn = simple_dynamic_detection(recent_index_tip)
                    if dyn is not None:
                        pred_text = dyn

                if pred_text == last_pred:
                    stable_count += 1
                else:
                    stable_count = 0
                last_pred = pred_text

                if stable_count >= 60:
                    if pred_text == "SPACE":
                        sentence += " "
                    else:
                        sentence += pred_text
                    stable_count = 0
                    print(f"Added letter: {pred_text}")

            else:
                recent_index_tip = []

            # Display
            display_pred = "(space)" if pred_text == "SPACE" else pred_text
            cv2.putText(frame, f"Pred: {display_pred}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
            cv2.putText(frame, f"Sentence: {sentence}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow("Predict (Press Q to quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                sentence = ""

    cap.release()
    cv2.destroyAllWindows()


def quick_help():
    print("ASL Real-time tool")
    print("Usage: python asl_realtime.py [collect|train|predict]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        quick_help()
        sys.exit(0)
    cmd = sys.argv[1].lower()
    if cmd == "collect":
        collect_mode()
    elif cmd == "train":
        train_mode()
    elif cmd == "predict":
        predict_mode()
    else:
        quick_help()