import time

import pyrealsense2 as rs
import numpy as np
import cv2
import os

class RealSenseCapture:
    def __init__(self, width=1280, height=720, fps=30, save_dir='camera_image'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        self.profile = self.pipeline.start(config)
        time.sleep(1)
        self.align = rs.align(rs.stream.color)
        device = self.profile.get_device()
        self.color_sensor = device.first_color_sensor()
        self.depth_sensor = device.first_depth_sensor()

        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()

        # self.depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
        # self.depth_sensor.set_option(rs.option.gain, 16)
        # self.depth_sensor.set_option(rs.option.visual_preset, 3)

        if self.color_sensor:
            self.color_sensor.set_option(rs.option.enable_auto_exposure, 1)
        self.color_sensor.set_option(rs.option.white_balance, 6000)
        self.color_sensor.set_option(rs.option.saturation, 45)

    def capture(self, color_name='color.png', depth_name='depth.png', save=False):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        # depth_frame = aligned_frames.get_depth_frame()

        # if not color_frame or not depth_frame:
        #     print("Failed to get frames.")
        #     return None, None

        # depth_frame = self.depth_to_disparity.process(depth_frame)
        # depth_frame = self.spatial.process(depth_frame)
        # depth_frame = self.temporal.process(depth_frame)
        # # depth_frame = self.hole_filling.process(depth_frame)
        # depth_frame = self.disparity_to_depth.process(depth_frame)

        color_image = np.asanyarray(color_frame.get_data())
        # depth_image = np.asanyarray(depth_frame.get_data())

        if save:
            color_path = os.path.join(self.save_dir, color_name)
            depth_path = os.path.join(self.save_dir, depth_name)
            cv2.imwrite(color_path, color_image)
            # cv2.imwrite(depth_path, depth_image)
            print(f"Images saved:\n - RGB: {color_path}\n - Depth: {depth_path}")

        return color_image, None

    def stop(self):
        self.pipeline.stop()
        print("RealSense pipeline stopped.")

