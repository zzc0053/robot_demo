import os
os.environ["QT_MAC_WANTS_LAYER"] = "1"

import time
import threading
import json
import socket
import tempfile

import cv2
import numpy as np

# ===== Voice (Whisper) deps =====
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

from gesture import HandGestureRecognizer
from camera import MacCameraCapture

# =========================
# Control / state
# =========================
control_enabled = False
stop_event = threading.Event()

# Latest camera frame
latest = {"frame": None, "ts": 0.0}
frame_lock = threading.Lock()

# UI shared (vision thread writes, main reads)
ui = {"frame": None, "text": "", "gesture": None, "stable": 0}
ui_lock = threading.Lock()

# Voice UI state (for display)
voice_state = {
    "status": "IDLE",   # IDLE / LISTEN / HEARD / CAND / NONE / DROPPED / SENT
    "text": "",
    "action_id": None,
    "ts": 0.0
}
voice_lock = threading.Lock()

# Candidates (latest only, no queue)
cand = {
    "vision": {"action_id": None, "payload": None, "ts": 0.0},
    "voice":  {"action_id": None, "payload": None, "ts": 0.0},
}
cand_lock = threading.Lock()

# Final command selected by arbiter -> sender sends it
current_cmd = None
current_payload = None
current_cmd_ts = 0.0
cmd_lock = threading.Lock()

# =========================
# Config
# =========================
GLOBAL_COOLDOWN_S = 5.0
ARBITER_PREFER = "voice"   # if both arrive, voice wins
STABLE_N = 30

# UDP target
JETSON_IP = "172.19.200.174"
JETSON_PORT = 5005

gesture = HandGestureRecognizer()

# Gesture -> action_id
GESTURE_TO_ACTION = {
    "damp":       (1, None),
    "index_up":   (2, None),  # stand_up
    "index_down": (3, None),  # stand_down
    "fist":       (4, None),  # hand_stand
    "open":       (0, None),  # no-op
    "yeah":       (0, None),  # no-op
}

def predict_gesture(frame):
    return gesture.detect(frame)

# =========================
# Voice: text -> action_id
# =========================
def parse_voice_to_action(text: str):
    if not text:
        return None
    t = text.strip().lower()

    if "stand up" in t or "standup" in t:
        return 2
    if "stand down" in t or "sit down" in t or "standdown" in t:
        return 3

    return None

# =========================
# Threads
# =========================
def camera_thread(camera: "MacCameraCapture"):
    while not stop_event.is_set():
        frame = camera.capture()
        if frame is None:
            time.sleep(0.01)
            continue
        with frame_lock:
            latest["frame"] = frame
            latest["ts"] = time.time()

def vision_thread(stable_N=STABLE_N):
    """
    Vision produces only the latest candidate (no queue).
    Global cooldown & dropping are handled by arbiter.
    """
    global control_enabled

    last_g = None
    same_cnt = 0

    while not stop_event.is_set():
        with frame_lock:
            frame = None if latest["frame"] is None else latest["frame"].copy()

        if frame is None:
            time.sleep(0.01)
            continue

        g = predict_gesture(frame)

        # Stable gate
        if g == last_g and g is not None:
            same_cnt += 1
        else:
            last_g = g
            same_cnt = 1

        # Collect voice info for UI
        with voice_lock:
            v_stat = voice_state["status"]
            v_txt  = voice_state["text"]
            v_act  = voice_state["action_id"]

        v_txt_short = (v_txt[:32] + "...") if len(v_txt) > 35 else v_txt

        status = "ON" if control_enabled else "OFF"
        line1 = f"gesture: {g}  stable: {same_cnt}/{stable_N}  control: {status}   (press 's' toggle, 'q' quit)"
        line2 = f"voice: {v_txt_short}  cmd: {v_act}  status: {v_stat}"
        txt = line1 + "\n" + line2

        with ui_lock:
            ui["frame"] = frame
            ui["text"] = txt
            ui["gesture"] = g
            ui["stable"] = same_cnt

        # Produce vision candidate only when control enabled + stable
        if not control_enabled:
            continue
        if g is None or g not in GESTURE_TO_ACTION:
            continue
        if same_cnt < stable_N:
            continue

        action_id, payload = GESTURE_TO_ACTION[g]
        if action_id == 0:
            continue

        now = time.time()
        with cand_lock:
            cand["vision"]["action_id"] = int(action_id)
            cand["vision"]["payload"] = payload
            cand["vision"]["ts"] = now

        print(f"[VISION] cand action_id={action_id} ts={now:.3f}")

def voice_thread():
    model = WhisperModel("small", device="cpu", compute_type="int8")
    sample_rate = 16000
    chunk_s = 2.0
    min_text_len = 3

    while not stop_event.is_set():
        text = ""
        try:
            with voice_lock:
                voice_state["status"] = "LISTEN"

            audio = sd.rec(int(chunk_s * sample_rate),
                           samplerate=sample_rate,
                           channels=1,
                           dtype="float32")
            sd.wait()
            audio = audio.reshape(-1)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                sf.write(f.name, audio, sample_rate)
                segments, _info = model.transcribe(
                    f.name,
                    language="en",
                    vad_filter=True
                )
                text = " ".join(seg.text for seg in segments).strip()

        except Exception as e:
            print(f"[VOICE][ERR] {e}")
            with voice_lock:
                voice_state["status"] = "IDLE"
            time.sleep(0.2)
            continue

        if not text or len(text) < min_text_len:
            with voice_lock:
                voice_state["status"] = "NONE"
                voice_state["text"] = ""
                voice_state["action_id"] = None
                voice_state["ts"] = time.time()
            continue

        print(f"[VOICE] '{text}'")

        action_id = parse_voice_to_action(text)
        now = time.time()

        with voice_lock:
            voice_state["text"] = text
            voice_state["action_id"] = action_id
            voice_state["ts"] = now
            voice_state["status"] = "CAND" if action_id is not None else "NONE"

        if action_id is None:
            continue

        if not control_enabled:
            continue

        with cand_lock:
            cand["voice"]["action_id"] = int(action_id)
            cand["voice"]["payload"] = None
            cand["voice"]["ts"] = now

        print(f"[VOICE] cand action_id={action_id} ts={now:.3f}")


def arbiter_thread(global_cooldown_s: float = GLOBAL_COOLDOWN_S, prefer: str = ARBITER_PREFER):
    global current_cmd, current_payload, current_cmd_ts
    global control_enabled

    last_fire_t = 0.0
    last_seen_ts = {"vision": 0.0, "voice": 0.0}

    while not stop_event.is_set():
        if not control_enabled:
            time.sleep(0.02)
            continue

        now = time.time()
        in_cooldown = (now - last_fire_t) < global_cooldown_s

        # Read candidates (and drop them if in cooldown)
        with cand_lock:
            v = cand["vision"].copy()
            a = cand["voice"].copy()

            if in_cooldown:
                # drop everything during cooldown
                cand["vision"]["action_id"] = None
                cand["voice"]["action_id"] = None

        if in_cooldown:
            # If a voice command was pending, mark as dropped for UI
            with voice_lock:
                if voice_state["status"] == "CAND":
                    voice_state["status"] = "DROPPED"
            time.sleep(0.02)
            continue

        v_new = v["action_id"] is not None and v["ts"] > last_seen_ts["vision"]
        a_new = a["action_id"] is not None and a["ts"] > last_seen_ts["voice"]

        if not v_new and not a_new:
            time.sleep(0.02)
            continue

        if v_new and a_new:
            pick = prefer
        else:
            pick = "voice" if a_new else "vision"

        chosen = a if pick == "voice" else v

        with cmd_lock:
            current_cmd = int(chosen["action_id"])
            current_payload = chosen["payload"]
            current_cmd_ts = float(chosen["ts"])

        # Mark used
        last_seen_ts[pick] = float(chosen["ts"])
        last_fire_t = now

        # Clear candidates (no stale resend)
        with cand_lock:
            cand["vision"]["action_id"] = None
            cand["voice"]["action_id"] = None

        if pick == "voice":
            with voice_lock:
                voice_state["status"] = "SENT"

        print(f"[ARBITER] pick={pick} SEND action_id={chosen['action_id']} cooldown_start={last_fire_t:.3f}")
        time.sleep(0.02)

def sender_thread(jetson_ip: str, jetson_port: int = 5005):
    global current_cmd, current_payload, current_cmd_ts

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    last_sent_ts = 0.0

    while not stop_event.is_set():
        with cmd_lock:
            action_id = current_cmd
            payload = current_payload
            ts = current_cmd_ts

        if action_id is None or ts <= last_sent_ts:
            time.sleep(0.02)
            continue

        msg = {"action_id": int(action_id), "payload": payload, "ts": float(ts)}
        data = (json.dumps(msg) + "\n").encode("utf-8")

        try:
            sock.sendto(data, (jetson_ip, jetson_port))
            print(f"[SEND] -> {jetson_ip}:{jetson_port} {msg}")
            last_sent_ts = ts
        except Exception as e:
            print(f"[SEND][ERR] {e}")

        with cmd_lock:
            current_cmd = None
            current_payload = None

        time.sleep(0.02)

# =========================
# Main (GUI must be main thread on macOS)
# =========================
def main():
    global control_enabled

    cam = MacCameraCapture(device_index=0, width=1280, height=720, fps=30)

    th_cam = threading.Thread(target=camera_thread, args=(cam,), daemon=True)
    th_vis = threading.Thread(target=vision_thread, daemon=True)
    th_voc = threading.Thread(target=voice_thread, daemon=True)
    th_arb = threading.Thread(target=arbiter_thread, daemon=True)
    th_send = threading.Thread(target=sender_thread, args=(JETSON_IP, JETSON_PORT), daemon=True)

    th_cam.start()
    th_vis.start()
    th_voc.start()
    th_arb.start()
    th_send.start()

    win = "Go2"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while not stop_event.is_set():
            with ui_lock:
                frame = None if ui["frame"] is None else ui["frame"].copy()
                text = ui["text"]

            if frame is None:
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(frame, "Loading camera...", (40, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Draw multi-line text (OpenCV doesn't handle '\n' automatically)
            lines = text.split("\n") if text else []
            y0 = 35
            for i, line in enumerate(lines[:2]):
                cv2.putText(frame, line, (10, y0 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                stop_event.set()
                break
            if key in (ord('s'), ord('S')):
                control_enabled = not control_enabled
                print(f"[SYSTEM] Gesture/Voice control = {control_enabled}")

            time.sleep(0.01)

    finally:
        stop_event.set()
        try:
            gesture.close()
        except Exception:
            pass
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
