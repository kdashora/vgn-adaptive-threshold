import sys
from unittest.mock import MagicMock

sys.modules['rospy'] = MagicMock()
sys.modules['sensor_msgs'] = MagicMock()
sys.modules['sensor_msgs.msg'] = MagicMock()
sys.modules['visualization_msgs'] = MagicMock()
sys.modules['visualization_msgs.msg'] = MagicMock()
sys.modules['std_msgs'] = MagicMock()
sys.modules['std_msgs.msg'] = MagicMock()
sys.modules['rclpy'] = MagicMock()
sys.modules['rclpy.context'] = MagicMock()
sys.modules['rclpy.impl'] = MagicMock()
sys.modules['rclpy.impl.implementation_singleton'] = MagicMock()
sys.modules['tf2_ros'] = MagicMock()
sys.modules['tf2_py'] = MagicMock()
sys.modules['tf2_ros.buffer'] = MagicMock()
sys.modules['geometry_msgs'] = MagicMock()
sys.modules['geometry_msgs.msg'] = MagicMock()
