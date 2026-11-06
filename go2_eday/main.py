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

# ── NEW: Push-to-Talk State ──
ptt_active = False
ptt_last_pressed = 0.0
ptt_required = True   # True = hold space to talk, False = always listening ('p' or on-screen button toggles)
ptt_lock = threading.Lock()

# ── Camera state (can be turned off to save power) ──
# The camera device is opened/released on the main thread only (macOS AVFoundation
# prefers that); camera_thread just reads from whatever cam_ref holds.
cam_ref = {"cam": None, "on": False}   # "on" is a lock-free flag for UI/status reads
cam_lock = threading.Lock()
camera_toggle_requested = False

# On-screen buttons: name -> rect (x1, y1, x2, y2) in frame coords; set each draw
ui_buttons = {}


def toggle_ptt_mode():
    global ptt_required
    with ptt_lock:
        ptt_required = not ptt_required
        mode = "PUSH-TO-TALK" if ptt_required else "ALWAYS LISTENING"
    print(f"[SYSTEM] Voice input mode = {mode}")


def request_camera_toggle():
    # Actual open/close happens in the main loop (opening can block ~1s)
    global camera_toggle_requested
    camera_toggle_requested = True


def camera_is_on():
    # No lock: camera_thread holds cam_lock during every frame read
    return cam_ref["on"]


def open_camera():
    try:
        c = MacCameraCapture(device_index=0, width=1280, height=720, fps=30)
    except Exception as e:
        print(f"[CAMERA][ERR] {e}  (camera stays OFF)")
        return False
    with cam_lock:
        cam_ref["cam"] = c
        cam_ref["on"] = True
    print("[SYSTEM] Camera ON")
    return True


def close_camera():
    cam_ref["on"] = False   # flip first so the UI shows OFF immediately
    with cam_lock:
        c = cam_ref["cam"]
        cam_ref["cam"] = None
        if c is not None:
            c.stop()
        with frame_lock:
            latest["frame"] = None
    if c is not None:
        print("[SYSTEM] Camera OFF (power saving)")


BUTTON_ACTIONS = {
    "ptt": toggle_ptt_mode,
    "camera": request_camera_toggle,
}


def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    for name, (x1, y1, x2, y2) in list(ui_buttons.items()):
        if x1 <= x <= x2 and y1 <= y <= y2:
            BUTTON_ACTIONS[name]()
            return


# =========================
# Config
# =========================
DEFAULT_COOLDOWN_S = 1.5
ARBITER_PREFER = "voice"
STABLE_N = 30

CAMERA_ENABLED = True   # start with the camera on? False = start off to save power ('c' / button toggles)

JETSON_IP = "100.68.185.25"
# JETSON_IP = "192.168.12.129"
# JETSON_IP = "172.19.200.174"
JETSON_PORT = 5005

# Keepalive: stop the Go2 going limp/sleeping when idle. If no command has been
# sent for KEEPALIVE_IDLE_S, re-send the command that holds its current pose:
#   standing -> Stand Up,  lying/sitting -> Stand Down,  after Damp -> nothing.
KEEPALIVE_ENABLED = True
KEEPALIVE_IDLE_S = 5 * 60
KEEPALIVE_ACTION_FOR_POSE = {
    "standing": 2,   # ACTION_STAND_UP
    "lying":    3,   # ACTION_STAND_DOWN
}

def pose_after(action_id):
    """Pose the dog is left in after an action: 'standing', 'lying', or None (damped/unknown)."""
    if action_id == 1:   # ACTION_DAMP -> limp on purpose, don't wake it
        return None
    if action_id == 3:   # ACTION_STAND_DOWN (also "sit down")
        return "lying"
    return "standing"    # stand up and every motion/trick end standing

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
ACTION_TURN_AROUND  = 14
ACTION_FRONT_FLIP      = 15
ACTION_LEFT_FLIP       = 16
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
        elif "back flip" in t:                  return 20
        elif "front flip" in t:                 return 15
        elif "left flip" in t:                  return 16
        elif "back" in t:                  return 6
        elif "turn left" in t:             return 9
        elif "turn right" in t:            return 10
        elif "left" in t:                  return 7
        elif "right" in t:                 return 8
        # elif "jump" in t:                  return 11
        elif "heart" in t:                 return 12
        elif "turn around" in t:            return 14
        elif "dance" in t or "party" in t or "disco" in t: return 13
    return None

# =========================
# Threads
# =========================
def camera_thread():
    while not stop_event.is_set():
        with cam_lock:
            cam = cam_ref["cam"]
            frame = cam.capture() if cam is not None else None
            if frame is not None:
                with frame_lock:
                    latest["frame"] = frame
                    latest["ts"] = time.time()
        if frame is None:
            # Camera off -> idle slowly; camera on but no frame -> retry soon
            time.sleep(0.1 if cam is None else 0.01)


def vision_thread(stable_N=STABLE_N):
    last_g = None
    same_cnt = 0

    while not stop_event.is_set():
        with frame_lock:
            frame = None if latest["frame"] is None else latest["frame"].copy()

        if frame is None:
            # Camera off (or not ready): no gesture work, but keep the voice status updating
            g = None
            time.sleep(0.05)
        else:
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

        with ptt_lock:
            is_ptt = ptt_active
            need_ptt = ptt_required

        v_txt_short = (v_txt[:32] + "...") if len(v_txt) > 35 else v_txt

        if need_ptt:
            mic_txt = "PTT: HELD" if is_ptt else "PTT: hold [space]"
        else:
            mic_txt = "ALWAYS LISTENING"

        cam_txt = "ON" if camera_is_on() else "OFF"
        line1 = f"gesture: {g}  stable: {same_cnt}/{stable_N}  control: {control_mode}  camera: {cam_txt}"
        line2 = f"voice ({mic_txt}): {v_txt_short}  cmd: {v_act}  status: {v_stat}"
        line3 = "[space]: PTT  'p': PTT on/off  'c': camera on/off  's': cycle mode  'q': quit"
        txt = line1 + "\n" + line2 + "\n" + line3

        with ui_lock:
            ui["frame"] = frame
            ui["text"] = txt
            ui["gesture"] = g
            ui["stable"] = same_cnt

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
        "heart", "dance", "party", "disco", "handstand", "turn around",
        "back flip", "front flip", "left flip"
    ])
    recognizer = KaldiRecognizer(model, sample_rate, grammar)

    was_ptt = False  # Track previous PTT state to reset on release

    def set_voice_ui(status, text="", action_id=None):
        now = time.time()
        with voice_lock:
            voice_state["status"] = status
            voice_state["text"] = text
            voice_state["action_id"] = action_id
            voice_state["ts"] = now

    set_voice_ui("IDLE", "", None)

    def audio_callback(indata, frames, time_info, status):
        nonlocal was_ptt
        if stop_event.is_set():
            return

        with ptt_lock:
            listening = ptt_active or not ptt_required

        # Only process audio if PTT is held, or PTT mode is off (always listening)
        if not listening:
            if was_ptt:
                # User just released PTT. Clear leftover partial words.
                try:
                    recognizer.Reset()
                except Exception:
                    pass
                set_voice_ui("IDLE", "", None)
                was_ptt = False
            return
        
        was_ptt = True
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
        blocksize=12000,
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
    last_send_time = time.time()   # wall time of last packet (real or keepalive)
    pose = None                    # unknown at start -> no keepalive until a command is sent

    while not stop_event.is_set():
        with cmd_lock:
            action_id = current_cmd
            payload = current_payload
            ts = current_cmd_ts

        if action_id is None or ts <= last_sent_ts:
            # ── Keepalive: idle for KEEPALIVE_IDLE_S -> re-send the hold-pose command ──
            ka_action = KEEPALIVE_ACTION_FOR_POSE.get(pose)
            if KEEPALIVE_ENABLED and ka_action is not None and time.time() - last_send_time >= KEEPALIVE_IDLE_S:
                now = time.time()
                msg = {"action_id": ka_action, "payload": None, "ts": now, "keepalive": True}
                try:
                    sock.sendto((json.dumps(msg) + "\n").encode("utf-8"), (jetson_ip, jetson_port))
                    print(f"[KEEPALIVE] {pose}, idle {KEEPALIVE_IDLE_S}s -> {jetson_ip}:{jetson_port} {msg}")
                except Exception as e:
                    print(f"[KEEPALIVE][ERR] {e}")
                last_send_time = now   # wait another full interval either way
            time.sleep(0.02)
            continue

        msg = {"action_id": int(action_id), "payload": payload, "ts": float(ts)}
        data = (json.dumps(msg) + "\n").encode("utf-8")

        try:
            sock.sendto(data, (jetson_ip, jetson_port))
            print(f"[SEND] -> {jetson_ip}:{jetson_port} {msg}")
            last_sent_ts = ts
            last_send_time = time.time()
            pose = pose_after(int(action_id))
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
    global control_mode, ptt_active, ptt_last_pressed, camera_toggle_requested

    if CAMERA_ENABLED:
        open_camera()
    else:
        print("[SYSTEM] Camera OFF at startup (CAMERA_ENABLED = False)")

    th_cam  = threading.Thread(target=camera_thread, daemon=True)
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
    cv2.setMouseCallback(win, on_mouse)

    try:
        while not stop_event.is_set():  
            with ui_lock:
                frame = None if ui["frame"] is None else ui["frame"].copy()
                text = ui["text"]

            cam_on = camera_is_on()
            if frame is None:
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                msg = "Loading camera..." if cam_on else "Camera OFF (power saving) - press 'c' to turn on"
                cv2.putText(frame, msg, (40, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            lines = text.split("\n") if text else []
            y0 = 35
            for i, line in enumerate(lines[:3]):
                cv2.putText(frame, line, (10, y0 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # ── Clickable buttons (bottom-left) ──
            h = frame.shape[0]

            # PTT toggle
            with ptt_lock:
                need_ptt = ptt_required
                held = ptt_active
            bx1, by1, bx2, by2 = 10, h - 60, 330, h - 10
            ui_buttons["ptt"] = (bx1, by1, bx2, by2)
            if need_ptt:
                label = "PTT: ON  (click / 'p')"
                color = (0, 200, 255) if held else (0, 140, 255)   # orange; brighter while held
            else:
                label = "PTT: OFF (always on)"
                color = (0, 180, 0)                                  # green = mic always live
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 2)
            cv2.putText(frame, label, (bx1 + 12, by2 - 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Camera toggle
            cx1, cy1, cx2, cy2 = 340, h - 60, 660, h - 10
            ui_buttons["camera"] = (cx1, cy1, cx2, cy2)
            if cam_on:
                label, color = "CAMERA: ON  (click / 'c')", (0, 180, 0)
            else:
                label, color = "CAMERA: OFF (click / 'c')", (90, 90, 90)
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), color, -1)
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 255, 255), 2)
            cv2.putText(frame, label, (cx1 + 12, cy2 - 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            cv2.imshow(win, frame)

            # Camera off -> redraw ~30x/s instead of ~100x/s (still fast enough for PTT key repeat)
            key = cv2.waitKey(1 if cam_on else 30) & 0xFF
            
            # --- KEYBINDS ---
            if key in (ord('q'), ord('Q'), 27):
                stop_event.set()
                break
                
            if key in (ord('s'), ord('S')):
                idx = CONTROL_MODES.index(control_mode)
                control_mode = CONTROL_MODES[(idx + 1) % len(CONTROL_MODES)]
                print(f"[SYSTEM] Control mode = {control_mode}")

            if key in (ord('p'), ord('P')):
                toggle_ptt_mode()

            if key in (ord('c'), ord('C')):
                request_camera_toggle()

            # Open/close the camera here on the main thread (from 'c' or the button)
            if camera_toggle_requested:
                camera_toggle_requested = False
                if camera_is_on():
                    close_camera()
                else:
                    open_camera()

            # ── CHANGED: Evaluate Push-To-Talk Spacebar ──
            if key == 32:  # ASCII for Spacebar
                with ptt_lock:
                    ptt_active = True
                    ptt_last_pressed = time.time()
                    
            # Auto-deactivate PTT if spacebar hasn't been detected recently.
            # OS keyboard repeat has a ~500ms initial delay, so 0.6s bridges the gap.
            with ptt_lock:
                if ptt_active and (time.time() - ptt_last_pressed > 0.6):
                    ptt_active = False

            time.sleep(0.01)

    finally:
        stop_event.set()
        try:
            gesture.close()
        except Exception:
            pass
        close_camera()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
