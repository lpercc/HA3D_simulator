import sys
import os
import json
from ui.form import Ui_Form
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtChart import *
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt
from src.utils.get_info import *
from src.utils.drawImage import drawCentering
import math
import numpy as np
import random
#sys.path.append('../')

#from src.utils.get_info import print_file_and_line_quick

WIDTH = 800
HEIGHT = 600
VFOV = 60
HFOV = VFOV*WIDTH/HEIGHT
TEXT_COLOR = [230, 40, 40]
ANGLEDELTA = 5 * math.pi / 180
TARGET_FPS = 20  # 目标帧率
FRAME_DURATION = 1000 / TARGET_FPS  # 目标帧持续时间ms

class myMainWindow(Ui_Form,QMainWindow):
    def __init__(self, sim, datasetPath, scanIDList):
        super(Ui_Form, self).__init__()
        self.sim = sim
        self.datasetPath = datasetPath
        self.scanIDList = []
        self.dataset = read_VLN_data(datasetPath)
        for item in self.dataset: 
            if item['scan'] not in self.scanIDList:
                self.scanIDList.append(item['scan'])
            if len(self.scanIDList) == 90:
                break
        self.scanID = scanIDList[0]
        self.positionData = read_position_data(f'con/pos_info/{self.scanID}_pos_info.json')
        self.connectionData = read_connection_data(f'con/con_info/{self.scanID}_con_info.json')
        self.pathListOfBuilding = [item for item in self.dataset if item['scan'] == self.scanID]
        self.pathItem = self.pathListOfBuilding[0]
        self.isPathTracking = False
        self.agentState = {}
        self.sim.newEpisode([self.scanID], [self.pathItem['path'][0]], [0], [0])
        self.setupUi(self)
        # label
        self.labelPathHeading.setText(f"{self.pathItem['heading']:.3f}")
        self.labelPathDistance.setText(f"{self.pathItem['distance']:.2f}")
        self.labelViewpointID.setText('0')
        self.labelAgentHeading.setText('0')
        self.labelAgentElevation.setText('0')
        self.labelAgentLocationX.setText('0')
        self.labelAgentLocationY.setText('0')
        self.labelAgentLocationZ.setText('0')
        self.labelAgentVFOV.setText(str(VFOV))
        # lineEdit
        self.lineEditScanID.setText(self.scanID)
        self.lineEditPathID.setText(str(self.pathItem['path_id']))
        # listWidget
        self.listWidgetPathViewpoint.addItems(self.pathItem['path'])
        # textEdit
        self.textEditInstruction1.setText(self.pathItem['instructions'][0])
        self.textEditInstruction2.setText(self.pathItem['instructions'][1])
        self.textEditInstruction3.setText(self.pathItem['instructions'][2])
        # pushButton
        self.pushButtonSearchBuilding.setEnabled(True)
        self.pushButtonSearchBuilding.clicked.connect(self.searchBuilding)
        self.pushButtonPreviousBuilding.setEnabled(False)
        self.pushButtonPreviousBuilding.clicked.connect(self.previousBuilding)
        self.pushButtonNextBuilding.setEnabled(True)
        self.pushButtonNextBuilding.clicked.connect(self.nextBuilding)
        self.pushButtonSearchPath.setEnabled(True)
        self.pushButtonSearchPath.clicked.connect(self.searchPath)
        self.pushButtonPreviousPath.setEnabled(False)
        self.pushButtonPreviousPath.clicked.connect(self.previousPath)
        self.pushButtonNextPath.setEnabled(True)
        self.pushButtonNextPath.clicked.connect(self.nextPath)
        self.pushButtonRandomBeginning.setEnabled(False)
        self.pushButtonRandomBeginning.clicked.connect(self.randomBeginning)
        self.pushButtonPathBack.setEnabled(False)
        self.pushButtonPathBack.clicked.connect(self.pathBack)
        self.pushButtonPathSave.setEnabled(False)
        self.pushButtonPathSave.clicked.connect(self.pathSave)
        self.pushButtonGenerateHuman.setEnabled(False)
        self.pushButtonGenerateHuman.clicked.connect(self.generateHuman)
        self.pushButtonGenerateInstructions.setEnabled(False)
        self.pushButtonGenerateInstructions.clicked.connect(self.generateInstructions)
        # checkBox
        self.checkBoxCreatePath.setChecked(False)
        self.checkBoxCreatePath.stateChanged.connect(self.labelCreatePath)
        self.checkBoxUpgradePath.setChecked(False)
        self.checkBoxUpgradePath.stateChanged.connect(self.labelUpgradePath)

        self.Image_label.setScaledContents(True)  # 自适应QLabel大小
        # 创建 QTimer 对象
        self.timer = QTimer(self)
        # 设置定时器超时（触发）的间隔时间（毫秒）
        self.timer.setInterval(int(FRAME_DURATION))  # 1000毫秒 = 1秒
        # 将定时器的 timeout 信号连接到我们要循环执行的函数
        self.timer.timeout.connect(self.updateAgentState)
        # 启动定时器
        self.timer.start()

        # chart initial
        self.createCoordinates()
        self.createScatterSeries()
        self.updateMap()
    
    def updateBuilding(self):
        self.pathListOfBuilding = [item for item in self.dataset if item['scan'] == self.scanID]
        self.lineEditScanID.setText(self.scanID)
        self.pathItem = self.pathListOfBuilding[0]
        self.positionData = read_position_data(f'con/pos_info/{self.scanID}_pos_info.json')
        self.connectionData = read_connection_data(f'con/con_info/{self.scanID}_con_info.json')
        self.pushButtonPreviousPath.setEnabled(False)
        self.pushButtonNextPath.setEnabled(True)
        self.updatePath()

    def nextBuilding(self):
        index = self.scanIDList.index(self.scanID) + 1
        if index == len(self.scanIDList) - 1:
            self.pushButtonNextBuilding.setEnabled(False)
        else:
            self.pushButtonPreviousBuilding.setEnabled(True)
        self.scanID = self.scanIDList[index]
        self.updateBuilding()

    def previousBuilding(self):
        index = self.scanIDList.index(self.scanID) - 1
        if index == 0:
            self.pushButtonPreviousBuilding.setEnabled(False)
        else:
            self.pushButtonNextBuilding.setEnabled(True)
        self.scanID = self.scanIDList[index]
        self.updateBuilding()

    def searchBuilding(self):
        text = self.lineEditScanID.text()
        if text == self.scanID: 
            return
        for index, item in enumerate(self.scanIDList):
            if item == text:
                self.scanID = item
                break
        if index == 0:
            self.pushButtonPreviousBuilding.setEnabled(False)
        elif index == len(self.scanIDList) - 1:
            self.pushButtonNextBuilding.setEnabled(False)
        else:
            self.pushButtonPreviousBuilding.setEnabled(True)
            self.pushButtonNextBuilding.setEnabled(True)
        self.updateBuilding()

    def updatePath(self):
        self.pushButtonRandomBeginning.setEnabled(False)
        self.timer.stop()
        self.sim.newEpisode([self.scanID], [self.pathItem['path'][0]], [0], [0])
        self.timer.start()
        self.labelPathHeading.setText(f"{self.pathItem['heading']:.3f}")
        self.labelPathDistance.setText(f"{self.pathItem['distance']:.2f}")
        self.lineEditPathID.setText(str(self.pathItem['path_id']))
        self.updatePathViewpoints()
        self.updateInstructions()
        self.updateMap()

    def updatePathViewpoints(self):
        self.listWidgetPathViewpoint.clear()
        self.listWidgetPathViewpoint.addItems(self.pathItem['path'])

    def updateInstructions(self):
        self.textEditInstruction1.setText(self.pathItem['instructions'][0])
        self.textEditInstruction2.setText(self.pathItem['instructions'][1])
        self.textEditInstruction3.setText(self.pathItem['instructions'][2])

    def nextPath(self):
        index = self.pathListOfBuilding.index(self.pathItem) + 1
        if index >= len(self.pathListOfBuilding) - 1:
            self.pushButtonNextPath.setEnabled(False)
        else:
            self.pushButtonPreviousPath.setEnabled(True)
        self.pathItem = self.pathListOfBuilding[index]
        self.updatePath()

    def previousPath(self):
        index = self.pathListOfBuilding.index(self.pathItem) - 1
        if index == 0:
            self.pushButtonPreviousPath.setEnabled(False)
        else:
            self.pushButtonNextPath.setEnabled(True)
        self.pathItem = self.pathListOfBuilding[index]
        self.updatePath()

    def searchPath(self):
        text = int(self.lineEditPathID.text())
        if text == self.pathItem['path_id']: 
            return
        for index, item in enumerate(self.pathListOfBuilding):
            if item["path_id"] == text:
                self.pathItem = item
                break
        if index == 0:
            self.pushButtonPreviousPath.setEnabled(False)
        elif index == len(self.pathListOfBuilding) - 1:
            self.pushButtonNextPath.setEnabled(False)
        else:
            self.pushButtonPreviousPath.setEnabled(True)
            self.pushButtonNextPath.setEnabled(True)
        self.updatePath()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()  # 
        elif event.key() == Qt.Key_A:
            sim.makeAction([0], [-ANGLEDELTA], [0])
            print("Left")  #
        elif event.key() == Qt.Key_D:
            sim.makeAction([0], [ANGLEDELTA], [0])
            print("Right")  #
        elif event.key() == Qt.Key_W:
            sim.makeAction([0], [0], [ANGLEDELTA])
            print("Up")  #
        elif event.key() == Qt.Key_S:
            sim.makeAction([0], [0], [-ANGLEDELTA])
            print("Down")  #
        elif event.key() == Qt.Key_F:
            forwardIdx = forwardViewpointIdx(self.agentState.navigableLocations)
            sim.makeAction([forwardIdx], [0], [0])           
            #print(f"Forward {forwardIdx}")  #
            #print(sim.getHumanState())
            if forwardIdx != 0 and self.isPathTracking:
                self.updateAgentState()
                if self.agentState.location.viewpointId not in self.pathItem["path"]:
                    self.pathItem["path"].append(self.agentState.location.viewpointId)
                    self.updatePathViewpoints()
                    self.updateMap()
        elif event.key() == Qt.Key_B:
            sim.makeAction([0], [0], [0])
            print("Back")
        super().keyPressEvent(event)

    def updateAgentState(self):
        self.agentState = self.sim.getState()[0]
        rgb = np.array(self.agentState.rgb, copy=False)
        drawCentering(rgb, 20)
        # 将 NumPy 数组转换为 QImage 对象
        height, width, channel = rgb.shape
        bytesPerLine = 3 * width
        qImg = QImage(rgb.data, width, height, bytesPerLine, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qImg)
        self.Image_label.setPixmap(pix)
        self.labelViewpointID.setText(self.agentState.location.viewpointId)
        self.labelAgentHeading.setText(f"{self.agentState.heading:.3f}")
        self.labelAgentElevation.setText(f"{self.agentState.elevation:.3f}")
        self.labelAgentLocationX.setText(f"{self.agentState.location.x:.2f}")
        self.labelAgentLocationY.setText(f"{self.agentState.location.y:.2f}")
        self.labelAgentLocationZ.setText(f"{self.agentState.location.z:.2f}")
        self.upgradeScatterSeriesAgent()
    
    def updateMap(self):
        self.upgradeCoordinates()
        self.upgradeScatterSeries()
        #self.upgradeScatterSeriesAgent()

    def changePushButton(self, flag):
        self.pushButtonPreviousBuilding.setEnabled(flag)
        self.pushButtonSearchBuilding.setEnabled(flag)
        self.pushButtonNextBuilding.setEnabled(flag)
        self.pushButtonPreviousPath.setEnabled(flag)
        self.pushButtonSearchPath.setEnabled(flag)
        self.pushButtonNextPath.setEnabled(flag)

    def labelCreatePath(self,state):
        if self.checkBoxCreatePath.isChecked():
            self.pushButtonRandomBeginning.setEnabled(True)
            self.pushButtonPathBack.setEnabled(True)
            self.pushButtonPathSave.setEnabled(True)
            self.checkBoxUpgradePath.setChecked(False)
            self.pathItem = {
                        "distance": 0, 
                        "scan": self.scanID, 
                        "path_id": len(self.dataset)+10000, 
                        "path": [self.agentState.location.viewpointId], 
                        "heading": self.agentState.heading, 
                        "instructions": ["None","None","None"]
            }
            self.pushButtonNextPath.setEnabled(False)
            self.pushButtonPreviousPath.setEnabled(False)
            self.pushButtonSearchPath.setEnabled(False)
            self.isPathTracking = True
            self.updatePath()
        elif not(self.checkBoxCreatePath.isChecked() or self.checkBoxUpgradePath.isChecked()):
            self.pushButtonPathBack.setEnabled(False)
            self.pushButtonPathSave.setEnabled(False)
            self.pushButtonRandomBeginning.setEnabled(False)
            self.pathItem = self.pathListOfBuilding[-1]
            self.pushButtonNextPath.setEnabled(False)
            self.pushButtonPreviousPath.setEnabled(True)
            self.pushButtonSearchPath.setEnabled(True)
            self.isPathTracking = False
            self.updatePath()

    def labelUpgradePath(self):
        if self.checkBoxUpgradePath.isChecked():
            self.pushButtonRandomBeginning.setEnabled(False)
            self.pushButtonPathBack.setEnabled(True)
            self.pushButtonPathSave.setEnabled(True)
            self.checkBoxCreatePath.setChecked(False)
            self.isPathTracking = True
            self.timer.stop()
            self.sim.newEpisode([self.scanID], [self.pathItem['path'][-1]], [0], [0])
            self.timer.start()
        elif not(self.checkBoxCreatePath.isChecked() or self.checkBoxUpgradePath.isChecked()):
            self.pushButtonPathBack.setEnabled(False)
            self.pushButtonPathSave.setEnabled(False)
            self.pushButtonRandomBeginning.setEnabled(False)
            self.isPathTracking = False
            self.updatePath()

    def randomBeginning(self):
        if len(self.pathItem["path"]) <= 1:
            index = random.randint(0, len(self.positionData)-1)
            self.pathItem["path"][0] = list(self.positionData.keys())[index]
            self.updatePath()
    
    def pathBack(self):
        if len(self.pathItem["path"]) > 1:
            self.pathItem["path"].pop()
            self.timer.stop()
            self.sim.newEpisode([self.scanID], [self.pathItem['path'][-1]], [self.agentState.heading], [self.agentState.elevation])
            self.timer.start()
            self.updatePathViewpoints()
            self.updateMap()

    def pathSave(self):
        tempViewpoint = self.pathItem['path'][0]
        distance = 0
        for viewpoint in self.pathItem['path']:
            distance += compute_distance(tempViewpoint, viewpoint, self.positionData)
            tempViewpoint = viewpoint
        self.pathItem['distance'] = distance
        #self.pathItem['distance']compute_distance()
        for index, item in enumerate(self.dataset):
            if self.pathItem['path_id'] == item['path_id']:
                self.dataset[index] = self.pathItem
                self.pathListOfBuilding = [item for item in self.dataset if item['scan'] == self.scanID]
                break
        if self.pathItem['path_id'] not in [item['path_id'] for item in self.pathListOfBuilding]:
            self.pathListOfBuilding.append(self.pathItem)
            self.dataset.append(self.pathItem)
        with open(self.datasetPath, 'w') as f:
            json.dump(self.dataset, f, indent=4)
        self.checkBoxUpgradePath.setChecked(False)
        self.checkBoxCreatePath.setChecked(False)
        self.isPathTracking = False

    
    def createCoordinates(self):
        # 绘制坐标轴
        axis_x = QLineSeries(self.chart_1)
        self.chart_1.addSeries(axis_x)
        axis_y = QLineSeries(self.chart_1)
        self.chart_1.addSeries(axis_y)
        self.chart_1.legend().hide()

        # 创建并设置 X 轴
        self.axisX = QValueAxis()
        self.axisX.setLabelFormat("%d")  # 设置标签格式，这里是整数
        self.chart_1.setAxisX(self.axisX, axis_x)

        # 创建并设置 Y 轴
        self.axisY = QValueAxis()
        self.axisY.setLabelFormat("%d")
        self.chart_1.setAxisY(self.axisY, axis_y)

    def upgradeCoordinates(self):
        allPoints = get_unobstructed_points(self.pathItem['path'], self.connectionData)
        path_set = set(self.pathItem['path'])
        self.unobstructedPoints = [point for point in allPoints if point not in path_set]
        viewpointsPosition = [self.positionData[viewpointID] for viewpointID in allPoints]
        # 检查是否已经有自定义轴，如果没有，则创建
        if not hasattr(self, 'axisX'):
            self.axisX = QValueAxis()
            self.chart_1.setAxisX(self.axisX, self.scatterSeriesCommon)

        if not hasattr(self, 'axisY'):
            self.axisY = QValueAxis()
            self.chart_1.setAxisY(self.axisY, self.scatterSeriesCommon)        
        # 初始化最大值和最小值为第一个点的坐标
        min_x = max_x = viewpointsPosition[0][0]
        min_y = max_y = viewpointsPosition[0][1]

        # 遍历所有点，更新最大值（上取整）和最小值（下取整）,
        for x, y, _ in viewpointsPosition:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
        min_x = int(min_x-1)
        max_x = int(max_x+1)
        min_y = int(min_y-1)
        max_y = int(max_y+1)

        # 设置 X 轴
        self.axisX.setRange(min_x, max_x)  # 设置轴的范围
        self.axisX.setTickCount(max_x-min_x+1)   # 设置刻度数量，包括两端点
        # 设置 Y 轴
        self.axisY.setRange(min_y, max_y)
        self.axisY.setTickCount(max_y-min_y+1)

    def createScatterSeries(self):
        # 创建散点系列
        # 普通点
        self.scatterSeriesCommon = QScatterSeries()
        self.scatterSeriesCommon.setName("Normal Points")
        self.scatterSeriesCommon.setMarkerShape(QScatterSeries.MarkerShapeCircle)  # 设置散点形状为圆形
        self.scatterSeriesCommon.setMarkerSize(10)  # 设置散点大小
        self.scatterSeriesCommon.setColor(QColor(Qt.blue))  # 设置散点颜色为蓝色
        
        # 路径点
        self.scatterSeriesPath = QScatterSeries()
        self.scatterSeriesPath.setName("Path Points")
        self.scatterSeriesPath.setMarkerShape(QScatterSeries.MarkerShapeCircle)  # 设置散点形状为圆形
        self.scatterSeriesPath.setMarkerSize(15)  # 设置散点大小
        self.scatterSeriesPath.setColor(QColor(Qt.red))  # 设置散点颜色为red

        # 创建线系列
        self.lineSeries = QLineSeries()
        self.lineSeries.setColor(QColor(Qt.red))  # 设置线颜色为红色

        # Agent点
        self.scatterSeriesAgent = QScatterSeries()
        self.scatterSeriesAgent.setName("Clicked Points")
        self.scatterSeriesAgent.setMarkerShape(QScatterSeries.MarkerShapeRectangle)  # 设置散点形状为三角形
        self.scatterSeriesAgent.setMarkerSize(10)  # 设置散点大小
        self.scatterSeriesAgent.setColor(QColor(Qt.green))  # 设置散点颜色为绿色

        # 将散点系列添加到图表中
        self.chart_1.addSeries(self.scatterSeriesCommon)
        self.chart_1.addSeries(self.scatterSeriesPath)
        self.chart_1.addSeries(self.scatterSeriesAgent)

        # 关联坐标轴（如果已经创建了自定义坐标轴）
        self.chart_1.setAxisX(self.axisX, self.scatterSeriesCommon)
        self.chart_1.setAxisY(self.axisY, self.scatterSeriesCommon)
        self.chart_1.setAxisX(self.axisX, self.scatterSeriesPath)
        self.chart_1.setAxisY(self.axisY, self.scatterSeriesPath)
        self.chart_1.setAxisX(self.axisX, self.scatterSeriesAgent)
        self.chart_1.setAxisY(self.axisY, self.scatterSeriesAgent)
        
        # 将线系列添加到图表中
        self.chart_1.addSeries(self.lineSeries)

        # 关联线系列到坐标轴
        self.chart_1.setAxisX(self.axisX, self.lineSeries)
        self.chart_1.setAxisY(self.axisY, self.lineSeries)

        # 连接 clicked 信号到槽函数
        #self.scatterSeriesCommon.clicked.connect(self.onScatterClicked)
        #self.scatterSeriesPath.clicked.connect(self.onScatterClicked)

    def upgradeScatterSeries(self):
        # 确保散点系列存在
        if not hasattr(self, 'scatterSeriesCommon'):
            self.scatterSeriesCommon = QScatterSeries()
            self.chart_1.addSeries(self.scatterSeriesCommon)
        if not hasattr(self, 'scatterSeriesPath'):
            self.scatterSeries_h = QScatterSeries()
            self.chart_1.addSeries(self.scatterSeriesPath)

        # 清除现有数据点
        self.scatterSeriesCommon.clear()
        self.scatterSeriesPath.clear()
        self.lineSeries.clear()  # 清除现有线数据
            
        # 添加数据点
        for viewpointID in self.unobstructedPoints:
            point = self.positionData[viewpointID]
            self.scatterSeriesCommon.append(point[0], point[1])
        for viewpointID in self.pathItem['path']:
            point = self.positionData[viewpointID]
            self.scatterSeriesPath.append(point[0], point[1])
            self.lineSeries.append(point[0], point[1])  # 将点添加到线系列

    def upgradeScatterSeriesAgent(self):
        if not hasattr(self, 'scatterSeriesAgent'):
            self.scatterSeriesAgent = QScatterSeries()
            self.chart_1.addSeries(self.scatterSeriesAgent)
        self.scatterSeriesAgent.clear()
        point = self.positionData[self.agentState.location.viewpointId]
        self.scatterSeriesAgent.append(point[0], point[1])

    def generateHuman(self):
        pass

    def generateInstructions(self):
        pass



if __name__ == '__main__':
    simulatorDataPath = os.path.join(os.environ.get("HC3D_SIMULATOR_DTAT_PATH"), "data/v1/scans")
    datasetPath = os.path.join('./tasks/R2R/data', 'path.json')
    scanIDList = readScanIDList("./connectivity/scans.txt")
    app = QApplication(sys.argv)
    import HC3DSim
    import cv2
    sim = HC3DSim.HCSimulator()
    #sim.setRenderingEnabled(False)
    sim.setRealTimeRender(True)
    sim.setDatasetPath(simulatorDataPath)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDepthEnabled(True) # Turn on depth only after running ./scripts/depth_toSkybox.py (see README.md)
    sim.initialize() 
    mainWindow = myMainWindow(sim, datasetPath, scanIDList)
    mainWindow.show()
    sys.exit(app.exec_())