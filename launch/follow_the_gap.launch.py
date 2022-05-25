from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    param_path = os.path.join(
        get_package_share_directory('f110_ftrg'),
        'param',
        'follow_the_gap.param.yaml')

    return LaunchDescription(
        [
            Node(
                package="f110_ftrg",
                executable="follow_the_gap_node",
                name="follow_the_gap_node",
                output="screen",
                parameters=[param_path],
                emulate_tty=True
            ),

            Node(
                package="vesc_ackermann"
            )
        ]
    )
