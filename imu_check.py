#from pyquaternion import Quaternion
import numpy as np
import time
import rospy
import roslib
from nav_msgs.msg import Odometry
import tf

T = np.array([
    [-0.00262866, -0.55954594,  0.82879517,  0.10278545],
    [-0.99999336,  0.00356416, -0.00076537,  0.00296669],
    [-0.00252569, -0.82879168, -0.55955159, -0.14354709],
    [ 0.        ,  0.        ,  0.        ,  1.        ]])

euler = tf.transformations.euler_from_matrix(T,axes='rxyz')
print(np.array(euler)/3.14*180)
quaternion = tf.transformations.quaternion_from_matrix(T)

rospy.init_node('tf_broadcaster')
br = tf.TransformBroadcaster()

br.sendTransform((1.0, 1.0, 1.0),(0,0,0,1),rospy.Time.now(),'IMU',"world")
br.sendTransform((0.10278545, 0.00296669, -0.14354709),quaternion,rospy.Time.now(),'Camera',"IMU")
rospy.sleep(1)