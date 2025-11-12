import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import tf2_ros
from geometry_msgs.msg import TransformStamped
import sensor_msgs_py.point_cloud2 as pc2


class LidarCameraCalibration(Node):
    def __init__(self):
        super().__init__('lidar_camera_calibration')
        self.bridge = CvBridge()

        # 구독자
        self.create_subscription(PointCloud2, '/carla/hero/lidar', self.lidar_callback, 10)
        self.create_subscription(Image, '/carla/hero/rgb_front/image', self.image_callback, 10)

        # 오버레이 퍼블리셔 (이 부분 변수 저장 빠졌었음)
        self.overlay_pub = self.create_publisher(Image, '/carla/hero/calib_overlay', 10)

        # TF 브로드캐스터
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # Extrinsic (LiDAR -> Camera)
        # CARLA 좌표계 변환: Camera_X=-LiDAR_Y, Camera_Y=-LiDAR_Z, Camera_Z=LiDAR_X
        # Camera는 LiDAR 기준 (2.0, 0.0, -0.4) 위치
        self.T_lidar2cam = np.array([
            [0, -1,  0,  0.0],
            [0,  0, -1, -0.4],
            [1,  0,  0, -2.0],
            [0,  0,  0,  1.0]
        ])

        # Camera Intrinsic
        self.K = np.array([
            [512, 0, 512],
            [0, 512, 384],
            [0,   0,   1]
        ])

        # 정적 TF 퍼블리시
        self.publish_static_tf()

    def publish_static_tf(self):
        t = TransformStamped()
        t.header.frame_id = 'lidar'
        t.child_frame_id = 'rgb_front'
        t.transform.translation.x = -2.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.4
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)
        self.get_logger().info("Static TF (lidar→camera) published.")

    def lidar_callback(self, msg):
        points = []
        intensities = []
        for p in pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity")):
            # p가 tuple인지, 유효한 포인트인지 확인
            if p is None or len(p) < 4:
                continue
            x, y, z, intensity = float(p[0]), float(p[1]), float(p[2]), float(p[3])
            points.append([x, y, z, 1.0])
            intensities.append(intensity)

        if len(points) == 0:
            self.get_logger().warn("No valid LiDAR points received.")
            return

        pts_np = np.array(points).T  # shape (4, N)
        pts_cam = self.T_lidar2cam @ pts_np
        self.latest_pts_cam = pts_cam
        self.latest_intensities = np.array(intensities)
        self.get_logger().info(f"LiDAR points: {len(points)}, Cam Z>0: {np.sum(pts_cam[2, :] > 0)}")

    def image_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if hasattr(self, 'latest_pts_cam'):
            pts_cam = self.latest_pts_cam
            mask = pts_cam[2, :] > 0  # 카메라 앞쪽 포인트만 사용
            pts_cam = pts_cam[:, mask]
            intensities = self.latest_intensities[mask]

            self.get_logger().info(f"Points in front of camera: {pts_cam.shape[1]}")

            # 투영
            proj = self.K @ pts_cam[:3, :]
            u = (proj[0, :] / proj[2, :]).astype(int)
            v = (proj[1, :] / proj[2, :]).astype(int)

            valid = (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0])
            num_valid = np.sum(valid)
            self.get_logger().info(f"Valid projected points: {num_valid}")

            # 거리 기반 색상 매핑 (가까운 곳: 빨강, 먼 곳: 파랑)
            depths = pts_cam[2, valid]  # Z값(거리)
            min_depth, max_depth = 0.0, 50.0

            for i, (x, y) in enumerate(zip(u[valid], v[valid])):
                depth = depths[i]
                # 거리를 0-1로 정규화
                normalized = np.clip((depth - min_depth) / (max_depth - min_depth), 0, 1)
                # HSV 색상맵: 빨강(0) -> 노랑(60) -> 초록(120) -> 청록(180) -> 파랑(240)
                hue = int((1.0 - normalized) * 240)  # 가까울수록 빨강(0), 멀수록 파랑(240)
                hsv_color = np.uint8([[[hue, 255, 255]]])
                bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
                color = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))
                cv2.circle(img, (x, y), 4, color, -1)  # 반지름 2 -> 4로 증가
        else:
            self.get_logger().warn("No LiDAR data yet")

        img_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        self.overlay_pub.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraCalibration()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
