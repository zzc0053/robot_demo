import math
import threading, queue
import time

import numpy as np


class ArmExecutor:
    def __init__(self, bot, rate_limit=0.1):
        self.bot = bot
        self.q = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._preempt = threading.Event()
        self.rate_limit = rate_limit
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def _sleep_preemptible(self, seconds, slice_ms=10):
        t0 = time.time()
        while (time.time() - t0) < seconds:
            if self._stop.is_set() or self._preempt.is_set():
                break
            time.sleep(slice_ms / 1000.0)

    def _loop(self):
        last_cmd = None
        while not self._stop.is_set():
            try:
                cmd, args = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                with self.lock:
                    if cmd in ("GRIPPER_OPEN", "GRIPPER_CLOSE") and cmd == last_cmd:
                        time.sleep(self.rate_limit)
                        continue

                    if cmd == "GRIPPER_OPEN":
                        self.bot.gripper.release(args.get("sec", 2.0))
                    elif cmd == "GRIPPER_CLOSE":
                        self.bot.gripper.grasp(args.get("sec", 2.0))
                    elif cmd == "POINT_UP":
                        self.bot.arm.set_ee_cartesian_trajectory(x=0.0, y=0.0, z=0.03)
                    elif cmd == "POINT_DOWN":
                        self.bot.arm.set_ee_cartesian_trajectory(x=0.0, y=0.0, z=-0.03)
                    elif cmd == "POINT_LEFT":
                        jp = self.bot.arm.get_single_joint_command('waist')
                        self.bot.arm.set_single_joint_position('waist', jp-0.05,blocking=True)
                    elif cmd == "POINT_RIGHT":
                        jp = self.bot.arm.get_single_joint_command('waist')
                        self.bot.arm.set_single_joint_position('waist', jp+0.05, blocking=True)
                    elif cmd == "ROLL_UP":
                        self.bot.arm.set_ee_cartesian_trajectory(pitch=math.radians(-15.0))
                    elif cmd == "ROLL_DOWN":
                        self.bot.arm.set_ee_cartesian_trajectory(pitch=math.radians(+15.0))
                    elif cmd == "ROLL_LEFT":
                        self.bot.arm.set_ee_cartesian_trajectory(roll=math.radians(+20.0))
                    elif cmd == "ROLL_RIGHT":
                        self.bot.arm.set_ee_cartesian_trajectory(roll=math.radians(-20.0))
                    elif cmd == "MOVE_FRONT":
                        self.bot.arm.set_ee_cartesian_trajectory(x=+0.05)
                    elif cmd == "MOVE_BACK":
                        self.bot.arm.set_ee_cartesian_trajectory(x=-0.05)
            except Exception as e:
                print(f"[ArmExecutor] command {cmd} failed: {e}")
            finally:
                last_cmd = cmd
                time.sleep(self.rate_limit)

    def submit(self, cmd, **kwargs):
        self._preempt.set()
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        try:
            self.q.put_nowait((cmd, kwargs))
        except queue.Full:
            pass

    def flush(self):
        self._preempt.set()
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass

    def stop(self):
        self._stop.set()
        self._preempt.set()
        self.worker.join(timeout=1.0)
