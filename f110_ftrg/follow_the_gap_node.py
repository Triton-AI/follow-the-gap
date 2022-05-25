import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import Joy


def pol2cart(r, theta):
    z = r * np.exp(1j * theta)
    x, y = z.real, z.imag
    return x, y


def cart2pol(x, y):
    r = np.linalg.norm(np.array([x, y]), axis=0)
    theta = np.arctan2(y, x)
    return r, theta


class FollowTheGap(Node):

    def __init__(self):
        super().__init__("follow_the_gap_node")
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/lidar/scan',
            self.lidar_cb,
            qos_profile_sensor_data
        )
        self.ackermann_pub = self.create_publisher(
            AckermannDriveStamped,
            'drive',
            10
        )
        self.joy_sub = self.create_subscription(
            Joy,
            '/joy',
            self.joy_cb,
            qos_profile_sensor_data
        )
        self.feature_pub = self.create_publisher(MarkerArray, "features", 10)
        self.angle_filter_min = self.declare_parameter(
            "angle_filter_min", -2.35619449).get_parameter_value().double_value  # -135 deg
        self.angle_filter_max = self.declare_parameter(
            "angle_filter_max", 2.35619449).get_parameter_value().double_value  # 135 deg
        self.min_gap_width = self.declare_parameter(
            "vehicle_width", 2.0).get_parameter_value().double_value
        self.turn_in_timing = self.declare_parameter(
            "turn_in_timing", 0.15).get_parameter_value().double_value  # magic number in rad
        self.wheel_base = self.declare_parameter(
            "wheel_base", 0.3).get_parameter_value().double_value  # 135 deg
        self.min_scan_range = self.declare_parameter(
            "min_scan_range", 0.2).get_parameter_value().double_value
        self.max_scan_range = self.declare_parameter(
            "max_scan_range", 10.0).get_parameter_value().double_value
        self.green_speed = self.declare_parameter(
            "green_speed", 2.0).get_parameter_value().double_value
        self.yellow_speed = self.declare_parameter(
            "yellow_speed", 1.0).get_parameter_value().double_value
        self.max_distance_from_right = self.declare_parameter(
            "max_distance_from_right", 1.0).get_parameter_value().double_value
        self.estop_button = self.declare_parameter(
            "estop_button", 1.0).get_parameter_value().integer_value

        self.ranges = None
        self.angles = None
        self.scan_cart = None
        self.pts = None
        self.output = AckermannDriveStamped()
        self.estop_set = False

    def joy_cb(self, msg: Joy):
        self.estop_set = (msg.buttons[self.estop_button] == 1.0)

    def lidar_cb(self, msg: LaserScan):
        # Convert scan to more convenient numpy arrays
        ranges = np.array(msg.ranges)
        angles = np.linspace(start=msg.angle_min,
                             stop=msg.angle_max, num=len(msg.ranges))
        angle_filter = (angles >= self.angle_filter_min) & (
            angles <= self.angle_filter_max)
        self.angles = angles[angle_filter]
        self.ranges = ranges[angle_filter]

        valid_range_filter = (ranges != 0.0) & (
            ranges < self.max_scan_range) & (ranges > self.min_scan_range)
        pt_x, pt_y = pol2cart(
            ranges[valid_range_filter], angles[valid_range_filter])
        self.pts = np.vstack([pt_x, pt_y]).T
        pts_shift = np.roll(self.pts, -1, axis=0)

        dists = np.linalg.norm(self.pts - pts_shift, axis=1)[:-1]

        candidates = dists >= self.min_gap_width
        num_candidates = np.sum(candidates)
        candidates_idx = np.nonzero(candidates)[0]

        if num_candidates == 0:
            # STOP
            self.output.drive.speed = 0.0
            self.output.drive.steering_angle = 0.0
        else:
            confirmed_candidates_mask = np.zeros(
                len(candidates_idx), dtype=np.bool)
            confirmed_candidates_front_mask = np.zeros(
                len(candidates_idx), dtype=np.bool)
            gap_in_front = False
            markers = MarkerArray()
            for i in range(len(candidates_idx)):
                confirmed_candidates_mask[i] = self.is_valid_candidate(
                    self.pts[candidates_idx[i]], pts_shift[candidates_idx[i]])
                if (confirmed_candidates_mask[i]):
                    rgb = (0.0, 1.0, 0.0)
                    if self.pts[candidates_idx[i]][0] > 0.0 or pts_shift[candidates_idx[i]][0] > 0.0:
                        confirmed_candidates_front_mask[i] = True
                        gap_in_front = True
                else:
                    rgb = (1.0, 0.0, 0.0)
                markers.markers.append(self.generate_marker(
                    i, self.pts[candidates_idx[i]], pts_shift[candidates_idx[i]], rgb))
            self.feature_pub.publish(markers)

            if gap_in_front:
                confirmed_candidates_mask = confirmed_candidates_mask & confirmed_candidates_front_mask
            candidates_idx = candidates_idx[confirmed_candidates_mask]
            num_candidates = len(np.nonzero(candidates_idx)[0])

            if num_candidates == 0:
                # Caution. Potential candidate. Drive slow and see what happens
                self.output.drive.speed = self.yellow_speed
                self.output.drive.steering_angle = 0.0
            else:
                # If there is gap in front, ignore gaps in the back

                # Follow the right most gap
                self.output.drive.speed = self.yellow_speed
                right_most_gap_idx = candidates_idx[0]
                # if np.linalg.norm(self.pts[right_most_gap_idx] - pts_shift[right_most_gap_idx]) < self.max_distance_from_right:
                lookahead_pt = (
                    self.pts[right_most_gap_idx] + pts_shift[right_most_gap_idx]) / 2.0
                self.output.drive.steering_angle = self.pure_pursuit(
                    lookahead_pt)

            self.output.header.stamp = self.get_clock().now().to_msg()
            if self.estop_set:
                self.output.drive.speed = 0.0
                self.output.drive.steering_angle = 0.0
            self.ackermann_pub.publish(self.output)

    def is_valid_candidate(self, pt_1: np.array, pt_2: np.array):
        # Check the projection of the gap is wide enough
        pts = np.vstack([pt_1, pt_2]).T
        dist = np.linalg.norm(pt_1 - pt_2)
        r, theta = cart2pol(pts[0], pts[1])
        score = np.abs(theta[1] - theta[0]) * self.min_gap_width * dist
        return score > self.turn_in_timing

    def pure_pursuit(self, pt: np.array):
        lookahead_angle = np.arctan2(pt[0], pt[1])
        lookahead_distance = np.linalg.norm(pt)
        return np.arctan(2 * self.wheel_base * np.sin(lookahead_angle) / lookahead_distance)

    def generate_marker(self, idx, pt_1: np.array, pt_2: np.array, rgb):
        lifetime = Duration()
        lifetime.nanosec = int(2e8)
        marker = Marker()
        marker.header.frame_id = "lidar_front"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.ARROW
        marker.ns = "feature"
        marker.id = idx
        marker.action = Marker.MODIFY
        marker.lifetime = lifetime
        point_1 = Point()
        point_1.x = pt_1[0]
        point_1.y = pt_1[1]
        point_2 = Point()
        point_2.x = pt_2[0]
        point_2.y = pt_2[1]
        marker.points.append(point_1)
        marker.points.append(point_2)
        marker.color.r = rgb[0]
        marker.color.g = rgb[1]
        marker.color.b = rgb[2]
        marker.color.a = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.0
        return marker


def main(args=None):
    rclpy.init(args=args)
    node = FollowTheGap()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
