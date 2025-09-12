from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import cv2
import math
import time
import numpy as np
from collections import deque, Counter
import mediapipe as mp

from camera import RealSenseCapture
from exe_thread import ArmExecutor

COOLDOWN = 4.0        # seconds, perform once each cooldown window
SMOOTH_WINDOW = 3     #
MODEL_COMPLEXITY = 0  # 0 fast，1 accurate
MAX_NUM_HANDS = 1
DET_CONF = 0.5
TRK_CONF = 0.5

HOLD_REPEAT = 0.5      # when keep the same gesture, 0.5s act once


def _to_np(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h, landmark.z], dtype=float)

def angle_deg(a, b, c):
    """ ∠ABC  """
    ba = a - b
    bc = c - b
    nba = ba / (np.linalg.norm(ba) + 1e-9)
    nbc = bc / (np.linalg.norm(bc) + 1e-9)
    cosang = np.clip(np.dot(nba, nbc), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def is_finger_extended(pts, triplet, wrist):
    """ strict version """
    mcp, pip, tip = (pts[i] for i in triplet)
    ang = angle_deg(mcp, pip, tip)
    straight = ang > 160.0
    d_tip = np.linalg.norm(tip[:2] - wrist[:2])
    d_pip = np.linalg.norm(pip[:2] - wrist[:2])
    farther = d_tip > d_pip * 1.05
    return straight and farther, ang

def is_index_pointing(pts, wrist):
    """ relaxed version"""
    mcp, pip, dip, tip = pts[5], pts[6], pts[7], pts[8]
    a1 = angle_deg(mcp, pip, dip)
    a2 = angle_deg(pip, dip, tip)
    mean_ang = (a1 + a2) / 2.0
    straight_enough = mean_ang > 150
    return straight_enough, mean_ang

def is_middle_pointing(pts, wrist):
    """ middle relaxed version"""
    mcp, pip, dip, tip = pts[9], pts[10], pts[11], pts[12]
    a1 = angle_deg(mcp, pip, dip)
    a2 = angle_deg(pip, dip, tip)
    mean_ang = (a1 + a2) / 2.0
    straight_enough = mean_ang > 150
    return straight_enough, mean_ang

def is_pinky_pointing(pts, wrist):
    """pinky relaxed version"""
    mcp, pip, dip, tip = pts[17], pts[18], pts[19], pts[20]
    a1 = angle_deg(mcp, pip, dip)
    a2 = angle_deg(pip, dip, tip)
    mean_ang = (a1 + a2) / 2.0
    straight_enough = mean_ang > 150
    return straight_enough, mean_ang

# —— Thumb: strict "UP" detection (to avoid sideways being considered as UP) ——
TH_UP_AZIM_TOL = 20.0   # Tolerance from vertical up (90°): ±20° => [70°,110°]
TH_LEN_RATIO   = 0.45   # Minimum ratio of |TIP-MCP| to palm width (5-17)
TH_IP_MIN      = 150.0  # Minimum thumb IP joint angle (larger means straighter)


def is_thumb_up_strict(pts):
    """Strict thumb-up: angle close to 90° + sufficiently long + relatively straight"""
    th_mcp, th_ip, th_tip = pts[2], pts[3], pts[4]
    v = th_tip - th_mcp
    ang2d = math.degrees(math.atan2(-v[1], v[0]))  # right=0°, up=90°
    up_ok = (90 - TH_UP_AZIM_TOL) <= ang2d <= (90 + TH_UP_AZIM_TOL)
    palm_width = np.linalg.norm(pts[5][:2] - pts[17][:2]) + 1e-9
    len_ratio  = np.linalg.norm(th_tip[:2] - th_mcp[:2]) / palm_width
    long_enough = len_ratio > TH_LEN_RATIO
    ip_ang = angle_deg(th_mcp, th_ip, th_tip)
    straight_enough = ip_ang > TH_IP_MIN
    ok = up_ok and long_enough and straight_enough
    dbg = {"ang2d": round(ang2d,1), "len_ratio": round(len_ratio,2), "ip_ang": round(ip_ang,1),
           "up_ok": bool(up_ok), "long_enough": bool(long_enough), "straight": bool(straight_enough)}
    return ok, dbg

# —— Thumb "inward curl/bend" (used for FIST) ——
TH_CURVE_ANG = 150.0   # Considered bent if IP joint angle < this value
TH_INWARD_K  = 0.98    # TIP is closer to the palm center (more inward than MCP)


def is_thumb_curled(pts):
    th_mcp, th_ip, th_tip = pts[2], pts[3], pts[4]
    ip_ang = angle_deg(th_mcp, th_ip, th_tip)
    palm_center = (pts[0][:2] + pts[5][:2] + pts[17][:2]) / 3.0
    tip_to_palm = np.linalg.norm(th_tip[:2] - palm_center)
    mcp_to_palm = np.linalg.norm(th_mcp[:2] - palm_center)
    inward = tip_to_palm < TH_INWARD_K * mcp_to_palm
    ok = (ip_ang < TH_CURVE_ANG) or inward
    dbg = {"ip_ang": round(ip_ang,1), "inward": bool(inward)}
    return ok, dbg

def pointing_direction(idx_mcp, idx_tip):
    v = idx_tip - idx_mcp
    dx, dy = v[0], v[1]
    ang = math.degrees(math.atan2(-dy, dx))  # 0:right, 90:up, -90:down, 180/-180:left
    if 45 <= ang < 135:
        return "POINT_UP", ang
    elif -135 < ang <= -45:
        return "POINT_DOWN", ang
    elif -45 <= ang < 45:
        return "POINT_LEFT", ang
    else:
        return "POINT_RIGHT", ang


TH_TRIPLET = (2, 3, 4)      # thumb：MCP=2, IP=3, TIP=4
ID_TRIPLET = (5, 6, 8)      # Index finger：MCP=5, PIP=6, TIP=8
MD_TRIPLET = (9, 10, 12)    # Middle finger
RG_TRIPLET = (13, 14, 16)   # Ring finger
PK_TRIPLET = (17, 18, 20)   # Pinky finger

def classify_gesture(pts):
    wrist = pts[0]

    # Index finger: lenient straight
    id_ext, id_ang = is_index_pointing(pts, wrist)

    # The other four fingers: strict straight
    md_ext, md_ang = is_finger_extended(pts, MD_TRIPLET, wrist)
    md_point_ext, md_point_ang = is_middle_pointing(pts, wrist)
    rg_ext, rg_ang = is_finger_extended(pts, RG_TRIPLET, wrist)
    pk_ext, pk_ang = is_finger_extended(pts, PK_TRIPLET, wrist)

    # Pinky: lenient straight for pointing (does not conflict with pk_ext; pk_ext is stricter)
    pk_point_ext, pk_point_ang = is_pinky_pointing(pts, wrist)

    # Thumb: strict UP + inward curl
    th_up,  th_up_dbg  = is_thumb_up_strict(pts)
    th_cur, th_cur_dbg = is_thumb_curled(pts)
    th_ext, th_ang = is_finger_extended(pts, TH_TRIPLET, wrist)

    # 1) THUMB_UP: thumb strictly pointing up, and the other four fingers are NOT extended
    if th_up and not id_ext and not md_ext and not rg_ext and not pk_ext:
        return "THUMB_UP", {"thumb_up": th_up_dbg}

    # 2) OPEN: all five fingers are straight
    if th_ext and id_ext and md_ext and rg_ext and pk_ext:
        return "OPEN", {"angles": [th_ang, id_ang, md_ang, rg_ang, pk_ang]}

    # 3) FIST: four fingers bent + thumb curled/inward
    four_bent = (not id_ext) and (not md_ext) and (not rg_ext) and (not pk_ext)
    if four_bent and th_cur:
        return "FIST", {"thumb_curled": th_cur_dbg}

    # 4) POINT_* (index pointing): only index extended (thumb free)
    if id_ext and not md_ext and not rg_ext and not pk_ext:
        label, ang = pointing_direction(pts[5], pts[8])
        return label, {"idx_angle": ang}

    # Index + middle both pointing up/down together -> MOVE_FWD / MOVE_BACK
    if id_ext and md_point_ext and not rg_ext and not pk_ext:
        dir1, ang1 = pointing_direction(pts[5], pts[8])   # index
        dir2, ang2 = pointing_direction(pts[9], pts[12])  # middle
        if dir1 in ("POINT_UP", "POINT_DOWN") and dir2 == dir1:
            if dir1 == "POINT_UP":
                return "MOVE_FWD", {"idx_ang": ang1, "mid_ang": ang2}
            else:
                return "MOVE_BACK", {"idx_ang": ang1, "mid_ang": ang2}

    # 5) PINKY_* (pinky pointing): only pinky extended (thumb free; other three not extended)
    if pk_point_ext and not id_ext and not md_ext and not rg_ext and not th_ext:
        dir_label, ang = pointing_direction(pts[17], pts[20])  # MCP=17, TIP=20
        dir_suffix = dir_label.split("_", 1)[1]  # take UP/DOWN/LEFT/RIGHT
        return f"PINKY_{dir_suffix}", {"pk_angle": ang}

    return "UNKNOWN", {"thumb_up": th_up_dbg, "thumb_curled": th_cur_dbg}



def run_action(label, details, state, arm_exec):
    now = time.time()

    if label == "OPEN" and (now - state.get("last_open_ts", 0.0) > 1.5):
        arm_exec.submit("GRIPPER_OPEN", sec=2.0)
        state["last_open_ts"] = now
        print("[ARM] Gripper OPEN")
        return
    if label == "FIST" and (now - state.get("last_open_ts", 0.0) > 1.5):
        arm_exec.submit("GRIPPER_CLOSE", sec=2.0)
        state["last_open_ts"] = now
        print("[ARM] Gripper CLOSE")
        return

    # —— Good job ——
    if label == "THUMB_UP":
        print("[SOCIAL] 👍 Great job!")
        return

    # —— Throttling when maintaining the same directional gesture ——
    def allow_repeat(time):
        return (now - state["last_repeat_ts"]) > time

    if label in ("MOVE_FWD", "MOVE_BACK") and allow_repeat(0.15):
        if label == "MOVE_FWD":
            arm_exec.submit("MOVE_FRONT")   # arm move front
            print("[ARM] Move X forward")
        else:
            arm_exec.submit("MOVE_BACK")    # arm move back
            print("[ARM] Move X backward")
        state["last_repeat_ts"] = now

    if label.startswith("POINT_") and allow_repeat(0.15):
        direction = label.split("_", 1)[1]  # UP/DOWN/LEFT/RIGHT
        if direction == "UP":
            arm_exec.submit("POINT_UP")
            print("[ARM] Move Z up")
        elif direction == "DOWN":
            arm_exec.submit("POINT_DOWN")
            print("[ARM] Move Z down")
        elif direction == "LEFT":
            arm_exec.submit("POINT_LEFT")
            print("[ARM] Move Wrist left")
        elif direction == "RIGHT":
            arm_exec.submit("POINT_RIGHT")
            print("[ARM] Move Wrist right")
        state["last_repeat_ts"] = now

    if label.startswith("PINKY_") and allow_repeat(0.5):
        if label == "PINKY_UP":
            arm_exec.submit("ROLL_UP")
        elif label == "PINKY_DOWN":
            arm_exec.submit("ROLL_DOWN")
        elif label == "PINKY_LEFT":
            arm_exec.submit("ROLL_LEFT")
        elif label == "PINKY_RIGHT":
            arm_exec.submit("ROLL_RIGHT")
        print(f"[ARM] Move {label}")
        state["last_repeat_ts"] = now

def _put_multiline(img, origin, lines, line_h=26, color=(255,255,255), scale=0.7, thickness=2):
    x, y = origin
    for i, text in enumerate(lines):
        cv2.putText(img, text, (x, y + i*line_h), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0,0,0), thickness+2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y + i*line_h), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, thickness, cv2.LINE_AA)

def draw_help_panel_topright(frame, show=True):
    if not show:
        return frame

    h, w = frame.shape[:2]
    pad = 10

    title_arm = "Arm Control"
    arm_lines = [
        "Index Finger - Move the arm Up / Down / Turn Left / Turn Right",
        "Index + Middle Fingers - Move the arm Forward / Backward",
    ]
    title_grip = "Gripper Control"
    grip_lines = [
        "Pinky Finger - Rotate the gripper Left / Right or Tilt Up / Down",
        "Open/Fist Hand - Open/Close the gripper",
    ]

    all_lines = [title_arm] + arm_lines + [""] + [title_grip] + grip_lines
    max_w = 0
    for t in all_lines:
        (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        max_w = max(max_w, tw)
    (tw_title, _), _ = cv2.getTextSize("Gripper Control", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    max_w = max(max_w, tw_title)

    gap_block = 5
    panel_h = 24 + len(arm_lines)*20 + gap_block + 24 + len(grip_lines)*20 + 2*pad
    panel_w = max_w + 2*pad

    x0 = w - panel_w - pad
    y0 = pad
    x1 = x0 + panel_w
    y1 = y0 + panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (40, 80, 130), -1)  # 蓝灰色
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    _put_multiline(frame, (x0 + pad, y0 + 18),[title_arm], line_h=23, color=(255,255,0), scale=0.7, thickness=1)
    _put_multiline(frame, (x0 + pad, y0 + 35), arm_lines, line_h=20, color=(255,255,255), scale=0.5, thickness=1)

    y_block2 = y0 + 35 + len(arm_lines)*20 + gap_block
    _put_multiline(frame, (x0 + pad, y_block2 + 18),[title_grip], line_h=24, color=(255,255,0), scale=0.7, thickness=1)
    _put_multiline(frame, (x0 + pad, y_block2 + 35),grip_lines, line_h=20, color=(255,255,255), scale=0.5, thickness=1)
    return frame


def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    cap = RealSenseCapture()

    bot = InterbotixManipulatorXS("vx300", "arm", "gripper")
    print("Robot initialized. Ready to move!")
    bot.arm.go_to_home_pose(moving_time=3.0, accel_time=1.0)

    arm_exec = ArmExecutor(bot)

    history = deque(maxlen=SMOOTH_WINDOW)
    last_stable = None
    last_trigger_time = 0.0

    state = {
        "last_repeat_ts": 0.0,
        "last_cmd_label": None,
        "last_open_ts": 0.0,
    }

    show_help = True # <- toggle help panel with 'h'

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=DET_CONF,
        min_tracking_confidence=TRK_CONF,
    ) as hands:
        while True:
            frame, _ = cap.capture(save=False)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = hands.process(rgb)
            rgb.flags.writeable = True

            h, w = frame.shape[:2]
            ui_label = "NO_HAND"

            if res.multi_hand_landmarks:
                hand_landmarks = res.multi_hand_landmarks[0]
                pts = np.stack([_to_np(lm, w, h) for lm in hand_landmarks.landmark], axis=0)

                now = time.time()
                label, details = classify_gesture(pts)

                history.append(label)
                c = Counter(history)
                stable_label, cnt = c.most_common(1)[0]
                stable_enough = cnt >= max(3, SMOOTH_WINDOW // 2 + 1)

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

                if stable_enough:
                    ui_label = stable_label

                    # If outside cooldown period AND stable label has changed -> trigger once (good for open/close actions)
                    if stable_label != last_stable and (now - last_trigger_time > COOLDOWN):
                        arm_exec.flush()
                        print(f"[{time.strftime('%H:%M:%S')}] Detected: {stable_label} | {details}")
                        last_stable = stable_label
                        last_trigger_time = now
                        run_action(stable_label, details, state, arm_exec)
                        state["last_cmd_label"] = stable_label

                    # If stable label hasn’t changed: for directional actions, continuously trigger at HOLD_REPEAT interval
                    elif stable_label == state.get("last_cmd_label"):
                        run_action(stable_label, details, state, arm_exec)
                        state["last_cmd_label"] = stable_label
                    else:
                        state["last_cmd_label"] = stable_label
                else:
                    ui_label = label

            cv2.rectangle(frame, (10, 10), (360, 80), (0, 0, 0), -1)
            cv2.putText(frame, f"{ui_label}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
            frame = draw_help_panel_topright(frame, show=show_help)
            cv2.namedWindow("GestureCam", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("GestureCam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN )
            cv2.imshow("GestureCam", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                show_help = not show_help

    cap.stop()
    arm_exec.stop()
    bot.arm.go_to_sleep_pose(moving_time=3.0, accel_time=1.0)
    time.sleep(1)
    bot.shutdown()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
