import sys
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QWidget, QVBoxLayout, QLabel , QMessageBox, QGraphicsScene, QGraphicsView 
from PyQt5.QtGui import QPixmap , QFont
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer
import cv2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import numpy as np
#import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from scipy.interpolate import RegularGridInterpolator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import subprocess
import digital_twins_data_gen as data
import rosArm

sys.path.append("./digital_twins_data_gen")

app = QtWidgets.QApplication(sys.argv) 
window = QtWidgets.QMainWindow()
window.setWindowTitle("CSIE Subject")
window.setGeometry(300, 100, 1000, 800) 

#set up buttons
font = QFont()
font.setPointSize(16)
font20 = QFont()
font20.setPointSize(20)
start_realsense = QtWidgets.QPushButton("Start Realsense", window)
start_realsense.resize(150, 60)
start_realsense.move(20, 600)
start_realsense.setFont(font)
stop_realsense = QtWidgets.QPushButton("Stop Realsense", window)
stop_realsense.resize(150, 60)
stop_realsense.move(220, 600)
stop_realsense.setFont(font)
start_arm = QtWidgets.QPushButton("Start Arm", window)
start_arm.resize(150, 60)
start_arm.move(620, 600)
start_arm.setFont(font)
start_gripper = QtWidgets.QPushButton("Start Gripper", window)
start_gripper.resize(150, 60)
start_gripper.move(20, 700)
start_gripper.setFont(font)
free_gripper = QtWidgets.QPushButton("Free Gripper", window)
free_gripper.resize(150, 60)
free_gripper.move(220, 700)
free_gripper.setFont(font)
generate = QtWidgets.QPushButton("Generate", window)
generate.resize(150, 60)
generate.move(420, 600)
generate.setFont(font)
clear = QtWidgets.QPushButton("Clear", window)
clear.resize(150, 60)
clear.move(420, 700)
clear.setFont(font)
home = QtWidgets.QPushButton("Home", window)
home.resize(150, 60)
home.move(620, 700)
home.setFont(font)
grip = QtWidgets.QPushButton("Grip", window)
grip.resize(150, 60)
grip.move(820, 600)
grip.setFont(font)
free = QtWidgets.QPushButton("Free", window)
free.resize(150, 60)
free.move(820, 700)
free.setFont(font)

#set up labels
output = QtWidgets.QGroupBox("output" , window)
output.resize(500 , 500)
output.move(50 , 30)
output.setFont(font)
layout1 = QVBoxLayout(output)
result = QtWidgets.QLabel()
result.setScaledContents(True)
layout1.addWidget(result, alignment=Qt.AlignCenter)
smooth = QtWidgets.QCheckBox("smooth" , window)
smooth.resize(100 , 100)
smooth.move(450 , 50)
smooth.setFont(font)

introduction = QtWidgets.QGroupBox("introduction" , window)
introduction.resize(300 , 400)
introduction.move(600 , 30)
introduction.setFont(font)
layout2 = QVBoxLayout(introduction)
direction = QtWidgets.QLabel()
direction.setScaledContents(True)
layout2.addWidget(direction, alignment=Qt.AlignCenter)
pixmap2 = QPixmap("./direction_introduction.png")
direction.setPixmap(pixmap2)

state = QtWidgets.QLabel("State : " , window)
state.resize(250 , 200)
state.move(600 , 400)
state.setFont(font20)

global pixmap1
pixmap1 = None

def make():
    global pixmap1
    pixmap1 = QPixmap("./smoothed_output.png")#再改成正確檔名即可
    result.setPixmap(pixmap1)
    smooth.setChecked(True)
generate.clicked.connect(make)

def change_outputimage():
        global pixmap1
        if pixmap1 is not None:
                if smooth.isChecked():
                        pixmap1 = QPixmap("./smoothed_output.png")#再改成正確檔名即可 後存一張有smooth過的
                        result.setPixmap(pixmap1)
                else:
                        pixmap1 = QPixmap("./original_output.png")#先存一張原始數據的
                        result.setPixmap(pixmap1)
        else:
               pass
smooth.clicked.connect(change_outputimage)

# def run_another_script():
#     # 取得程式碼的目錄
#     script_dir = os.path.dirname(os.path.realpath(__file__))
#     
#     # 要執行的另一個 Python 檔案的路徑
#     script_path = os.path.join(script_dir, 'digital_twins_data_gen.py')
#     
#     # 使用 subprocess 模組執行另一個 Python 檔案
#     subprocess.run(['python3', script_path])


def stop_clicked():
        data.key = cv2.waitKey(5)
        data.key = 27
        data.data_process()


#start_gripper.clicked.connect(rosArm.gripper_move)

#figure = plt.figure()
#canvas = FigureCanvas(figure)
#ax = figure.add_subplot(111, projection='3d')
#
#layout = QVBoxLayout()
#layout.addWidget(canvas)
global images
images = []
timer = QTimer()

def clear_result():
    global pixmap1
    result.clear()
    pixmap1 = None
    smooth.setChecked(False)
    data.x_wrist_data = []
    data.y_wrist_data = []
    data.z_wrist_data = []

def start_display():
    global timer
    timer = QTimer()
    timer.timeout.connect(show_next_image)
    timer.start(1000) # 每隔一秒顯示一張圖片

def show_next_image():
    if images:
        image = images.pop(0)
        result.setPixmap(image)
       
        if not images:
            timer.stop()  # 如果所有圖片都顯示完畢，停止計時器
    else:
           print("No images to display.")

def generate_images():
        new_x_wrist_data = data.x_wrist_data[: : 5]
        new_y_wrist_data = data.y_wrist_data[: : 5]
        new_z_wrist_data = data.z_wrist_data[: : 5]
        for i in range(len(new_x_wrist_data)):
                  # 創建 3D 圖形
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                ax.scatter(data.x_wrist_data, data.y_wrist_data, data.z_wrist_data, c='blue', alpha=0.6)
                for j in range(len(data.x_wrist_data)):
                      if data.x_wrist_data[j] == new_x_wrist_data[i] and data.y_wrist_data[j] == new_y_wrist_data[i] and data.z_wrist_data[j] == new_z_wrist_data[i]:
                         ax.scatter(new_x_wrist_data[i], new_y_wrist_data[i], new_z_wrist_data[i], c='red', alpha=1.0, s=100)

                  # 設置坐標軸標籤
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                plt.savefig(f'new_wrist_data{i+1}.png')
                plt.close()
                pixmap = QPixmap(f'new_wrist_data{i+1}.png')
                images.append(pixmap)
        print("generate complete")

def changestate(value):
    if value == 1:
        state.setText("State : Start Realsense")
    elif value == 2:
        state.setText("State : Stop Realsense")
    elif value == 3:
        state.setText("State : Generate Images")
    elif value == 4:
        state.setText("State : Start Arm")
    elif value == 5:
        state.setText("State : Start Gripper")
    elif value == 6:
        state.setText("State : Free Gripper")
    elif value == 7:
        state.setText("State : Free")
    elif value == 8:
        state.setText("State : Grip")
    elif value == 9:
        state.setText("State : Clear Data")
    elif value == 10:
        state.setText("State : Go home")

start_realsense.clicked.connect(data.realsense)
start_realsense.clicked.connect(lambda: changestate(1))

stop_realsense.clicked.connect(stop_clicked)
stop_realsense.clicked.connect(lambda: changestate(2))

generate.clicked.connect(generate_images)
generate.clicked.connect(lambda: changestate(3))

start_arm.clicked.connect(rosArm.arm_move)
start_arm.clicked.connect(start_display)
start_arm.clicked.connect(lambda: changestate(4))

start_gripper.clicked.connect(rosArm.gripper_grasp)
start_gripper.clicked.connect(lambda: changestate(5))

free_gripper.clicked.connect(rosArm.gripper_free)
free_gripper.clicked.connect(lambda: changestate(6))

free.clicked.connect(rosArm.g.gripper_on)
free.clicked.connect(lambda: changestate(7))

grip.clicked.connect(rosArm.g.gripper_off)
grip.clicked.connect(lambda: changestate(8))

clear.clicked.connect(clear_result)
clear.clicked.connect(lambda: changestate(9))

home.clicked.connect(rosArm.home)
home.clicked.connect(lambda: changestate(10))

def plot_data(self, x_wrist_data, y_wrist_data, z_wrist_data):
        self.ax.clear()
        self.ax.scatter(x_wrist_data, y_wrist_data, z_wrist_data, color='blue', label='Mapped Data')

        for i in range(len(x_wrist_data) - 1):
            self.ax.plot([x_wrist_data[i], x_wrist_data[i+1]], 
                         [y_wrist_data[i], y_wrist_data[i+1]], 
                         [z_wrist_data[i], z_wrist_data[i+1]], 
                         color='red')

        self.ax.set_xlabel("X (pixel)")
        self.ax.set_ylabel("Y (pixel)")
        self.ax.set_zlabel("Z (meter)")
        self.ax.legend()
        self.canvas.draw()
#def run_other():
    
window.show()
sys.exit(app.exec_())
