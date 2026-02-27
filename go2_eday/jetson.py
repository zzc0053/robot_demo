import json
import socket
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient


# action_id 定义（和 Mac 端保持一致）
ACTION_DAMP = 1
ACTION_STAND_UP = 2
ACTION_STAND_DOWN = 3
ACTION_HAND_STAND = 4
ACTION_MOVE = 10  # 可选：payload=(vx, vy, wz)


def make_sport_client(nic: str = "enp3s0") -> SportClient:
    # 2) init Unitree (你要求的方式)
    ChannelFactoryInitialize(0, nic)
    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()
    return sport_client


def do_action(sport: SportClient, action_id: int, payload):
    """
    payload:
      - None for discrete actions
      - (vx, vy, wz) for move
    """
    if action_id == ACTION_DAMP:
        sport.Damp()

    elif action_id == ACTION_STAND_UP:
        sport.StandUp()

    elif action_id == ACTION_STAND_DOWN:
        sport.StandDown()

    elif action_id == ACTION_HAND_STAND:
        sport.HandStand(True)
        time.sleep(4.0)
        sport.HandStand(False)

    elif action_id == ACTION_MOVE:
        if payload is None or len(payload) != 3:
            print("[JETSON] MOVE payload invalid, expected [vx, vy, wz]")
            return
        vx, vy, wz = float(payload[0]), float(payload[1]), float(payload[2])
        sport.Move(vx, vy, wz)

    else:
        print(f"[JETSON] Unknown action_id={action_id}")


def udp_server(host: str = "0.0.0.0", port: int = 5005, nic: str = "enp3s0"):
    sport = make_sport_client(nic)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    print(f"[JETSON] UDP listening on {host}:{port} (NIC={nic})")

    while True:
        data, addr = sock.recvfrom(8192)
        try:
            msg = json.loads(data.decode("utf-8").strip())
        except Exception as e:
            print(f"[JETSON] Bad packet from {addr}: {e}")
            continue

        action_id = msg.get("action_id", None)
        payload = msg.get("payload", None)
        ts = msg.get("ts", None)

        if action_id is None:
            print(f"[JETSON] Missing action_id from {addr}: {msg}")
            continue

        try:
            action_id = int(action_id)
        except Exception:
            print(f"[JETSON] action_id not int: {action_id}")
            continue

        print(f"[JETSON] RECV from {addr} action_id={action_id} payload={payload} ts={ts}")

        try:
            do_action(sport, action_id, payload)
        except Exception as e:
            print(f"[JETSON] Action failed: {e}")


if __name__ == "__main__":
    NIC = "enp3s0"
    udp_server(host="0.0.0.0", port=5005, nic=NIC)
