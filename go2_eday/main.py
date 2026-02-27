import os
os.environ["QT_MAC_WANTS_LAYER"] = "1"

import time
import threading
import json
import socket
import tempfile

import cv2
import numpy as np

import sounddevice as sd
import soundfile as sf
from vosk import Model, KaldiRecognizer

from gesture import HandGestureRecognizer
from camera import MacCameraCapture

# =========================
# Control / state
# =========================
CONTROL_MODES = ["OFF", "VOICE", "GESTURE", "BOTH"]
control_mode = "OFF"          # replaces control_enabled
stop_event = threading.Event()

latest = {"frame": None, "ts": 0.0}
frame_lock = threading.Lock()

ui = {"frame": None, "text": "", "gesture": None, "stable": 0}
ui_lock = threading.Lock()

voice_state = {
    "status": "IDLE",
    "text": "",
    "action_id": None,
    "ts": 0.0
}
voice_lock = threading.Lock()

cand = {
    "vision": {"action_id": None, "payload": None, "ts": 0.0},
    "voice":  {"action_id": None, "payload": None, "ts": 0.0},
}
cand_lock = threading.Lock()

current_cmd = None
current_payload = None
current_cmd_ts = 0.0
cmd_lock = threading.Lock()

# =========================
# Config
# =========================
DEFAULT_COOLDOWN_S = 1.5
ARBITER_PREFER = "voice"
STABLE_N = 30

# JETSON_IP = "100.68.185.25"
JETSON_IP = "192.168.12.129"
JETSON_PORT = 5005

gesture = HandGestureRecognizer()

ACTION_DAMP         = 1
ACTION_STAND_UP     = 2
ACTION_STAND_DOWN   = 3
ACTION_HAND_STAND   = 4
ACTION_MOVE_FORWARD = 5
ACTION_MOVE_BACKWARD= 6
ACTION_MOVE_LEFT    = 7
ACTION_MOVE_RIGHT   = 8
ACTION_TURN_LEFT    = 9
ACTION_TURN_RIGHT   = 10
ACTION_JUMP         = 11
ACTION_HEART        = 12
ACTION_DANCE        = 13
ACTION_FLIP         = 20

ACTION_COOLDOWNS = {
    ACTION_DAMP: 1.0,
    ACTION_STAND_UP: 1.5,
    ACTION_STAND_DOWN: 1.5,
    ACTION_HAND_STAND: 4.0,
    ACTION_MOVE_FORWARD: 0.5,
    ACTION_MOVE_BACKWARD: 0.5,
    ACTION_MOVE_LEFT: 0.5,
    ACTION_MOVE_RIGHT: 0.5,
    ACTION_TURN_LEFT: 0.5,
    ACTION_TURN_RIGHT: 0.5,
    ACTION_JUMP: 2.0,
    ACTION_HEART: 3.0,
    ACTION_DANCE: 15.0,
    ACTION_FLIP: 3.0,
}

GESTURE_TO_ACTION = {
    "damp":       (1, None),
    "index_up":   (2, None),
    "index_down": (3, None),
    "fist":       (4, None),
    "open":       (12, None),
    "yeah":       (20, None),
}

# =========================
# Mode helpers
# =========================
def is_voice_enabled():
    return control_mode in ("VOICE", "BOTH")

def is_gesture_enabled():
    return control_mode in ("GESTURE", "BOTH")

def predict_gesture(frame):
    return gesture.detect(frame)

# =========================
# Voice: text -> action_id
# =========================
def parse_voice_to_action(text: str):
    if not text:
        return None
    t = text.strip().lower()
    if "fido" in t:
        if ("stand" in t and not "hand" in t) or "up" in t:      return 2
        elif "down" in t or "sit" in t:    return 3
        elif 'hand' in t and "stand" in t: return 4
        elif "forward" in t:               return 5
        elif "flip" in t:                  return 20
        elif "back" in t:                  return 6
        elif "turn left" in t:             return 9
        elif "turn right" in t:            return 10
        elif "left" in t:                  return 7
        elif "right" in t:                 return 8
        # elif "jump" in t:                  return 11
        elif "heart" in t:                 return 12
        elif "dance" in t or "party" in t or "disco" in t: return 13
    return None

# =========================
# Threads
# =========================
def camera_thread(camera):
    while not stop_event.is_set():
        frame = camera.capture()
        if frame is None:
            time.sleep(0.01)
            continue
        with frame_lock:
            latest["frame"] = frame
            latest["ts"] = time.time()


def vision_thread(stable_N=STABLE_N):
    last_g = None
    same_cnt = 0

    while not stop_event.is_set():
        with frame_lock:
            frame = None if latest["frame"] is None else latest["frame"].copy()

        if frame is None:
            time.sleep(0.01)
            continue

        g = predict_gesture(frame)

        if g == last_g and g is not None:
            same_cnt += 1
        else:
            last_g = g
            same_cnt = 1

        with voice_lock:
            v_stat = voice_state["status"]
            v_txt  = voice_state["text"]
            v_act  = voice_state["action_id"]

        v_txt_short = (v_txt[:32] + "...") if len(v_txt) > 35 else v_txt

        # ── CHANGED: show current mode instead of ON/OFF ──
        line1 = f"gesture: {g}  stable: {same_cnt}/{stable_N}  control: {control_mode}   (press 's' cycle, 'q' quit)"
        line2 = f"voice: {v_txt_short}  cmd: {v_act}  status: {v_stat}"
        txt = line1 + "\n" + line2

        with ui_lock:
            ui["frame"] = frame
            ui["text"] = txt
            ui["gesture"] = g
            ui["stable"] = same_cnt

        # ── CHANGED: guard on gesture mode ──
        if not is_gesture_enabled():
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
    model_path = "vosk-model-en-us-0.22-lgraph"
    model = Model(model_path)
    sample_rate = 16000

    grammar = json.dumps([
        "fido", "stand up", "stand down", "sit down", "flip",
        "move left", "move right", "move forward", "move back",
        "move backward", "turn left", "turn right",
        "heart", "dance", "party", "disco", "handstand"
    ])
    recognizer = KaldiRecognizer(model, sample_rate, grammar)

    def set_voice_ui(status, text="", action_id=None):
        now = time.time()
        with voice_lock:
            voice_state["status"] = status
            voice_state["text"] = text
            voice_state["action_id"] = action_id
            voice_state["ts"] = now

    set_voice_ui("IDLE", "", None)

    def audio_callback(indata, frames, time_info, status):
        if stop_event.is_set():
            return

        data = bytes(indata)

        if recognizer.AcceptWaveform(data):
            j = json.loads(recognizer.Result())
            text = j.get("text", "").strip()

            if not text:
                set_voice_ui("IDLE", "", None)
                return

            action_id = parse_voice_to_action(text)
            set_voice_ui("CAND" if action_id else "NONE", text, action_id)

            if action_id is None:
                return
            # ── CHANGED: guard on voice mode ──
            if not is_voice_enabled():
                return

            now = time.time()
            with cand_lock:
                cand["voice"]["action_id"] = int(action_id)
                cand["voice"]["payload"] = None
                cand["voice"]["ts"] = now

            print(f"[VOICE] '{text}'")
            print(f"[VOICE] cand action_id={action_id} ts={now:.3f}")

        else:
            pj = json.loads(recognizer.PartialResult())
            ptxt = pj.get("partial", "").strip()
            if ptxt:
                with voice_lock:
                    voice_state["status"] = "LISTEN"
                    voice_state["text"] = ptxt
                    voice_state["action_id"] = parse_voice_to_action(ptxt)
                    voice_state["ts"] = time.time()

    with sd.RawInputStream(
        samplerate=sample_rate,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback
    ):
        while not stop_event.is_set():
            time.sleep(0.05)


def arbiter_thread(prefer: str = ARBITER_PREFER):
    global current_cmd, current_payload, current_cmd_ts

    last_fire_t = 0.0
    current_cooldown_s = 0.0
    last_seen_ts = {"vision": 0.0, "voice": 0.0}

    while not stop_event.is_set():
        # ── CHANGED: OFF means neither input accepted ──
        if control_mode == "OFF":
            time.sleep(0.02)
            continue

        now = time.time()
        in_cooldown = (now - last_fire_t) < current_cooldown_s

        with cand_lock:
            v = cand["vision"].copy()
            a = cand["voice"].copy()
            if in_cooldown:
                cand["vision"]["action_id"] = None
                cand["voice"]["action_id"] = None

        if in_cooldown:
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

        last_seen_ts[pick] = float(chosen["ts"])
        last_fire_t = now
        current_cooldown_s = ACTION_COOLDOWNS.get(current_cmd, DEFAULT_COOLDOWN_S)

        with cand_lock:
            cand["vision"]["action_id"] = None
            cand["voice"]["action_id"] = None

        if pick == "voice":
            with voice_lock:
                voice_state["status"] = "SENT"

        print(f"[ARBITER] pick={pick} SEND action_id={chosen['action_id']} cooldown={current_cooldown_s}s")
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
# Main
# =========================
def main():
    global control_mode

    cam = MacCameraCapture(device_index=0, width=1280, height=720, fps=30)

    th_cam  = threading.Thread(target=camera_thread, args=(cam,), daemon=True)
    th_vis  = threading.Thread(target=vision_thread, daemon=True)
    th_voc  = threading.Thread(target=voice_thread, daemon=True)
    th_arb  = threading.Thread(target=arbiter_thread, daemon=True)
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
            # ── CHANGED: cycle through 4 modes ──
            if key in (ord('s'), ord('S')):
                idx = CONTROL_MODES.index(control_mode)
                control_mode = CONTROL_MODES[(idx + 1) % len(CONTROL_MODES)]
                print(f"[SYSTEM] Control mode = {control_mode}")

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
