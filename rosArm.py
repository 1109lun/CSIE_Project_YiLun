import RobotControl_func_ros1
import rospy
import gripper
import digital_twins_data_gen as data

rospy.init_node('my_node_name')

robot = RobotControl_func_ros1.RobotControl_Func()
pos_home = [496.8341605883501, -98.10461371288869, 434.80804443359375, -179.68991088867188, -0.650927722454071, 44.84828567504883]
[x,y,z,u,v,w] = robot.get_TMPos()
#print([x,y,z,u,v,w])
g = gripper.Gripper(1)

def median_of_sorted_descending(arr):
    # Step 1: Create a sorted copy of the array
    sorted_arr = sorted(arr, reverse=True)
    
    # Step 2: Find the middle element(s) of the sorted array
    n = len(sorted_arr)
    mid = n // 2  # Integer division to find the middle index
    
    if n % 2 == 0:
        # If the array length is even, average the two middle elements
        median = (sorted_arr[mid - 1] + sorted_arr[mid]) / 2
    else:
        # If the array length is odd, return the middle element
        median = sorted_arr[mid]
    
    return median

global input_y 
global input_z 
global input_x 


def arm_move():
    input_y = data.x_wrist_data[: : 5]
    input_z= data.y_wrist_data[: : 5]
    input_x = data.z_wrist_data[: : 5]
    [x, y, z, u, v, w] = robot.get_TMPos()
    # robot.set_TMPos([x+(x_start-x_median) , y+(y_start-y_median), z+(z_start-z_median) , u , v , w])
    for i in range(len(input_x) - 1):
        [x, y, z, u, v, w] = robot.get_TMPos()
        #robot.set_TMPos([x - 500*(input_x[i + 1] - input_x[i]), y + 1.5*(input_y[i + 1] - input_y[i]), z-1.2*(input_z[i + 1] - input_z[i]), u, v, w])
        robot.set_TMPos([x, y + 1.5*(input_y[i + 1] - input_y[i]), z-1.2*(input_z[i + 1] - input_z[i]), u, v, w])
    print("move arm complete\n")


def gripper_grasp():
    input_y = data.x_wrist_data[: : 5]
    input_z= data.y_wrist_data[: : 5]
    input_x = data.z_wrist_data[: : 5]
    [x, y, z, u, v, w] = robot.get_TMPos()
    # robot.set_TMPos([x+(x_start-x_median) , y+(y_start-y_median), z+(z_start-z_median) , u , v , w])
    for i in range(len(input_x) - 1):
        [x, y, z, u, v, w] = robot.get_TMPos()
        robot.set_TMPos([x - 500*(input_x[i + 1] - input_x[i]), y + 1.5*(input_y[i + 1] - input_y[i]), z-1.2*(input_z[i + 1] - input_z[i]), u, v, w])
    g.gripper_off()

def gripper_free():
    input_y = data.x_wrist_data[: : 5]
    input_z= data.y_wrist_data[: : 5]
    input_x = data.z_wrist_data[: : 5]
    [x, y, z, u, v, w] = robot.get_TMPos()
    # robot.set_TMPos([x+(x_start-x_median) , y+(y_start-y_median), z+(z_start-z_median) , u , v , w])
    for i in range(len(input_x) - 1):
        [x, y, z, u, v, w] = robot.get_TMPos()
        robot.set_TMPos([x - 500*(input_x[i + 1] - input_x[i]), y + 1.5*(input_y[i + 1] - input_y[i]), z-1.2*(input_z[i + 1] - input_z[i]), u, v, w])
    g.gripper_on()

def home():
    robot.set_TMPos(pos_home)
    g.gripper_reset()

