import sys
import os
import json
from form import Ui_Form
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtChart import *
from PyQt5.QtMultimediaWidgets import QVideoWidget
import random
import math
import imageio
HA3D_SIMULATOR_PATH = os.environ.get("HA3D_SIMULATOR_PATH")
sys.path.append(HA3D_SIMULATOR_PATH)
from src.utils.concat_skybox import *
DOWNSIZED_WIDTH = 512
DOWNSIZED_HEIGHT = 512

class MyMainWindow(Ui_Form, QMainWindow):
    def __init__(self, viewpointImageDir, modelVideoDir, scanList, humanMotionData, regionMotionData, motionText):
        super(Ui_Form, self).__init__()
        self.viewpointImageDir = viewpointImageDir
        self.modelVideoDir = modelVideoDir
        self.scanList = scanList
        self.humanMotionData = humanMotionData
        self.regionMotionData = regionMotionData
        self.motionText = motionText

        self.scanId = self.scanList[0]
        with open(f'con/pos_info/{self.scanId}_pos_info.json', 'r') as f:
            self.locationData = json.load(f)
        self.viewpointList = [key for key in self.locationData]
        self.viewpointId = self.viewpointList[0]
        self.location = self.locationData[self.viewpointId]
        _, self.humanCount = compute(self.locationData, 67)

        self.regionList = [region for region in self.regionMotionData]
        self.region = self.regionList[0]
        self.humanMotionList = self.regionMotionData[self.region]
        self.humanMotion = self.humanMotionList[0]
        self.humanMotionId = self.motionText.index(f"{self.region}:{self.humanMotion}")
        self.humanModelList = ['0', '1', '2']
        self.humanModelId = self.humanModelList[0]

        self.panoramaImagePath = os.path.join(viewpointImageDir, self.scanId, "matterport_skybox_images", f"{self.viewpointId}_skybox_small.jpg")
        self.feetImagePath = os.path.join(viewpointImageDir, self.scanId, "matterport_skybox_images", f"{self.viewpointId}_skybox5_sami.jpg")
        self.modelVideoPath = os.path.join(self.modelVideoDir, f"sample{self.humanMotionId:02d}_rep{int(self.humanModelId):02d}.mp4")
        if not os.path.exists(self.modelVideoPath):
            print("Error: Model video path does not exist.")
            return

        self.setupUi(self)
        self.initializeUiElements()
        self.showImage()

        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.videowidget_motion)

        self.createCoordinates()
        self.createScatterSeries()
 
    def initializeUiElements(self):
        self.textBrowser_scanID.setText(self.scanId)
        self.textBrowser_viewpointID.setText(self.viewpointId)
        self.textBrowser_location.setText(f"X:{self.location[0]} Y:{self.location[1]}")
        self.textBrowser_humanCount.setText(f"{self.humanCount}")

        self.comboBox_region.addItems(self.regionList)
        self.comboBox_humanMotion.addItems(self.humanMotionList)
        self.comboBox_3DmodelID.addItems(self.humanModelList)
        self.comboBox_region.activated[str].connect(self.changeRegion)

        self.pushButton_scanID_pre.setEnabled(False)
        self.pushButton_scanID_pre.clicked.connect(self.preScan)
        self.pushButton_scanID_pre.clicked.connect(self.showImage)
        self.pushButton_scanID_next.clicked.connect(self.nextScan)
        self.pushButton_scanID_next.clicked.connect(self.showImage)

        self.pushButton_viewpointID_pre.setEnabled(False)
        self.pushButton_viewpointID_pre.clicked.connect(self.preViewpoint)
        self.pushButton_viewpointID_pre.clicked.connect(self.showImage)
        self.pushButton_viewpointID_next.clicked.connect(self.nextViewpoint)
        self.pushButton_viewpointID_next.clicked.connect(self.showImage)

        self.pushButton_play.clicked.connect(self.playModelVideo)
        self.pushButton_stop.clicked.connect(self.stopModelVideo)
        self.pushButton_open.clicked.connect(self.openModelVideo)
        self.pushButton_play.setEnabled(False)
        self.pushButton_stop.setEnabled(False)
        self.pushButton_random_loc.clicked.connect(self.randomHumanLocation)
        self.pushButton_save.clicked.connect(self.save)
        
        self.label_frame1.setScaledContents(True)
        self.label_frame2.setScaledContents(True)

    def preScan(self):
        index = self.scanList.index(self.scanId) - 1
        self.pushButton_scanID_next.setEnabled(True)
        if index == 0:
            self.pushButton_scanID_pre.setEnabled(False)
        self.updateScan(index)

    def nextScan(self):
        index = self.scanList.index(self.scanId) + 1
        self.pushButton_scanID_pre.setEnabled(True)
        if index == len(self.scanList) - 1:
            self.pushButton_scanID_next.setEnabled(False)
        self.updateScan(index)

    def updateScan(self, index):
        self.scanId = self.scanList[index]
        with open(f'con/pos_info/{self.scanId}_pos_info.json', 'r') as f:
            self.locationData = json.load(f)
        self.viewpointList = [key for key in self.locationData]
        self.viewpointId = self.viewpointList[0]
        self.location = self.locationData[self.viewpointId]
        _, self.humanCount = compute(self.locationData, 67)
        self.textBrowser_scanID.setText(self.scanId)
        self.textBrowser_viewpointID.setText(self.viewpointId)
        self.textBrowser_location.setText(f"X:{self.location[0]} Y:{self.location[1]}")
        self.textBrowser_humanCount.setText(f"{self.humanCount}")
        self.updateCoordinates()
        self.updateScatterSeries()
        self.updateHumanMotion()

    def preViewpoint(self):
        index = self.viewpointList.index(self.viewpointId) - 1
        self.pushButton_viewpointID_next.setEnabled(True)
        if index == 0:
            self.pushButton_viewpointID_pre.setEnabled(False)
        self.updateViewpoint(index)

    def nextViewpoint(self):
        index = self.viewpointList.index(self.viewpointId) + 1
        self.pushButton_viewpointID_pre.setEnabled(True)
        if index == len(self.viewpointList) - 1:
            self.pushButton_viewpointID_next.setEnabled(False)
        self.updateViewpoint(index)

    def updateViewpoint(self, index):
        self.viewpointId = self.viewpointList[index]
        self.location = self.locationData[self.viewpointId]
        self.textBrowser_viewpointID.setText(self.viewpointId)
        self.textBrowser_location.setText(f"X:{self.location[0]} Y:{self.location[1]}")
        self.scatterSeriesClicked.clear()
        self.scatterSeriesClicked.append(self.location[0], self.location[1])
        self.updateHumanMotion()

    def save(self):
        if self.scanId not in self.humanMotionData:
            self.humanMotionData[self.scanId] = {}
        if self.viewpointId not in self.humanMotionData[self.scanId]:
            self.humanMotionData[self.scanId][self.viewpointId] = []
            self.humanMotionData[self.scanId][self.viewpointId].append(f"{self.comboBox_region.currentText()}:{self.comboBox_humanMotion.currentText()}")
            self.humanMotionData[self.scanId][self.viewpointId].append(int(self.comboBox_3DmodelID.currentText()))
            self.humanMotionData[self.scanId][self.viewpointId].append(0)
        else:
            self.humanMotionData[self.scanId][self.viewpointId][0] = f"{self.comboBox_region.currentText()}:{self.comboBox_humanMotion.currentText()}"
            self.humanMotionData[self.scanId][self.viewpointId][1] = int(self.comboBox_3DmodelID.currentText())
            self.humanMotionData[self.scanId][self.viewpointId][2] = 0
        with open("human_motion_text.json", 'w') as f:
            json.dump(self.humanMotionData, f, indent=4)
        self.updateScatterSeries()

    def randomHumanLocation(self):
        self.humanMotionData[self.scanId] = {}
        viewpointSet = set(self.viewpointList)
        for i in range(self.humanCount):
            excludedViewpoints = set(self.humanMotionData[self.scanId])
            candidateList = list(viewpointSet - excludedViewpoints)
            while True:
                if candidateList:
                    selectedViewpoint = random.choice(candidateList)
                if not self.humanMotionData[self.scanId]:
                    break
                else:
                    distanceList = []
                    for humanPoint in [self.locationData[viewpointId] for viewpointId in self.humanMotionData[self.scanId]]:
                        selectedPoint = self.locationData[selectedViewpoint]
                        distance = math.sqrt((humanPoint[0] - selectedPoint[0])**2 + 
                                            (humanPoint[1] - selectedPoint[1])**2 + 
                                            (humanPoint[2] - selectedPoint[2])**2)
                        distanceList.append(distance)
                    if sum(distanceList) > len(distanceList) * 3:
                        break
            self.humanMotionData[self.scanId][selectedViewpoint] = ["", 0, 0]
        self.updateScatterSeries()

    def changeRegion(self, region):
        self.comboBox_humanMotion.clear()
        self.humanMotionList = self.regionMotionData[region]
        self.comboBox_humanMotion.addItems(self.humanMotionList)

    def showImage(self):
        self.panoramaImagePath = os.path.join(self.viewpointImageDir, self.scanId, "matterport_skybox_images", f"{self.viewpointId}_skybox_small.jpg")
        if not os.path.exists(self.panoramaImagePath):
            self.label_frame1.setText("No Panorama Image!")
            self.label_frame2.setText("No skybox-feet Image!")
        else:  
            imageio.imwrite("panoramaImage.jpg", concat(self.panoramaImagePath, DOWNSIZED_WIDTH))
            pixPanorama = QPixmap("panoramaImage.jpg")
            self.label_frame1.setPixmap(pixPanorama)
            imageio.imwrite("feetImage.jpg", concat_feet(self.panoramaImagePath, DOWNSIZED_WIDTH))
            pixFeet = QPixmap("feetImage.jpg")
            self.label_frame2.setPixmap(pixFeet)

    def updateHumanMotion(self):
        if self.scanId not in self.humanMotionData:
            return
        if self.viewpointId in self.humanMotionData[self.scanId]:
            human = self.humanMotionData[self.scanId][self.viewpointId]
            if human == ["", 0, 0]:
                return
            region = human[0].split(":")[0]
            self.changeRegion(region)
            humanMotion = human[0].split(":")[1]
            humanModelId = str(human[1])
            self.comboBox_region.setCurrentText(region)
            self.comboBox_humanMotion.setCurrentText(humanMotion)
            self.comboBox_3DmodelID.setCurrentText(humanModelId)
            self.openModelVideo()

    def openModelVideo(self):
        self.region = self.comboBox_region.currentText()
        self.humanMotion = self.comboBox_humanMotion.currentText()
        self.humanMotionId = self.motionText.index(f"{self.region}:{self.humanMotion}")
        self.humanModelId = self.comboBox_3DmodelID.currentText()
        self.modelVideoPath = os.path.join(self.modelVideoDir, f"sample{self.humanMotionId:02d}_rep{int(self.humanModelId):02d}.mp4")
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.modelVideoPath)))
        self.pushButton_stop.setEnabled(True)
        self.pushButton_play.setEnabled(False)
        self.player.play()

    def playModelVideo(self):
        self.pushButton_stop.setEnabled(True)
        self.pushButton_play.setEnabled(False)
        self.player.play()

    def stopModelVideo(self):
        self.pushButton_stop.setEnabled(False)
        self.pushButton_play.setEnabled(True)
        self.player.pause()

    def createCoordinates(self):
        dataPoints = [self.locationData[viewpointId] for viewpointId in self.locationData]
        axisX = QLineSeries(self.chart_1)
        self.chart_1.addSeries(axisX)
        axisY = QLineSeries(self.chart_1)
        self.chart_1.addSeries(axisY)
        self.chart_1.legend().hide()

        minX = maxX = dataPoints[0][0]
        minY = maxY = dataPoints[0][1]

        for x, y, _ in dataPoints:
            minX = min(minX, x)
            maxX = max(maxX, x)
            minY = min(minY, y)
            maxY = max(maxY, y)
        minX = int(minX - 1)
        maxX = int(maxX + 1)
        minY = int(minY - 1)
        maxY = int(maxY + 1)

        self.axisX = QValueAxis()
        self.axisX.setRange(minX, maxX)
        self.axisX.setTickCount(maxX - minX + 1)
        self.axisX.setLabelFormat("%d")
        self.chart_1.setAxisX(self.axisX, axisX)

        self.axisY = QValueAxis()
        self.axisY.setRange(minY, maxY)
        self.axisY.setTickCount(maxY - minY + 1)
        self.axisY.setLabelFormat("%d")
        self.chart_1.setAxisY(self.axisY, axisY)

    def updateCoordinates(self):
        dataPoints = [self.locationData[viewpointId] for viewpointId in self.locationData]
        if not hasattr(self, 'axisX'):
            self.axisX = QValueAxis()
            self.chart_1.setAxisX(self.axisX, self.scatterSeries)
        if not hasattr(self, 'axisY'):
            self.axisY = QValueAxis()
            self.chart_1.setAxisY(self.axisY, self.scatterSeries)
        minX = maxX = dataPoints[0][0]
        minY = maxY = dataPoints[0][1]

        for x, y, _ in dataPoints:
            minX = min(minX, x)
            maxX = max(maxX, x)
            minY = min(minY, y)
            maxY = max(maxY, y)
        minX = int(minX - 1)
        maxX = int(maxX + 1)
        minY = int(minY - 1)
        maxY = int(maxY + 1)

        self.axisX.setRange(minX, maxX)
        self.axisX.setTickCount(maxX - minX + 1)
        self.axisY.setRange(minY, maxY)
        self.axisY.setTickCount(maxY - minY + 1)

    def createScatterSeries(self):
        self.scatterSeries = QScatterSeries()
        self.scatterSeries.setName("Normal Points")
        self.scatterSeries.setMarkerShape(QScatterSeries.MarkerShapeCircle)
        self.scatterSeries.setMarkerSize(10)
        self.scatterSeries.setColor(QColor(Qt.blue))

        self.scatterSeriesHuman = QScatterSeries()
        self.scatterSeriesHuman.setName("Human Points")
        self.scatterSeriesHuman.setMarkerShape(QScatterSeries.MarkerShapeCircle)
        self.scatterSeriesHuman.setMarkerSize(10)
        self.scatterSeriesHuman.setColor(QColor(Qt.red))

        self.scatterSeriesClicked = QScatterSeries()
        self.scatterSeriesClicked.setName("Clicked Points")
        self.scatterSeriesClicked.setMarkerShape(QScatterSeries.MarkerShapeCircle)
        self.scatterSeriesClicked.setMarkerSize(10)
        self.scatterSeriesClicked.setColor(QColor(Qt.green))

        for viewpointId in self.locationData:
            point = self.locationData[viewpointId]
            if self.scanId not in self.humanMotionData:
                self.humanMotionData[self.scanId] = {}
            if viewpointId == self.viewpointId:
                self.scatterSeriesClicked.append(point[0], point[1])
            if viewpointId in self.humanMotionData[self.scanId]:
                self.scatterSeriesHuman.append(point[0], point[1])
            else:
                self.scatterSeries.append(point[0], point[1])

        self.chart_1.addSeries(self.scatterSeries)
        self.chart_1.addSeries(self.scatterSeriesHuman)
        self.chart_1.addSeries(self.scatterSeriesClicked)

        self.chart_1.setAxisX(self.axisX, self.scatterSeries)
        self.chart_1.setAxisY(self.axisY, self.scatterSeries)
        self.chart_1.setAxisX(self.axisX, self.scatterSeriesHuman)
        self.chart_1.setAxisY(self.axisY, self.scatterSeriesHuman)
        self.chart_1.setAxisX(self.axisX, self.scatterSeriesClicked)
        self.chart_1.setAxisY(self.axisY, self.scatterSeriesClicked)

        self.scatterSeries.clicked.connect(self.onScatterClicked)
        self.scatterSeriesHuman.clicked.connect(self.onScatterClicked)

    def updateScatterSeries(self):
        if not hasattr(self, 'scatterSeries'):
            self.scatterSeries = QScatterSeries()
            self.chart_1.addSeries(self.scatterSeries)
        if not hasattr(self, 'scatterSeriesHuman'):
            self.scatterSeriesHuman = QScatterSeries()
            self.chart_1.addSeries(self.scatterSeriesHuman)
        if not hasattr(self, 'scatterSeriesClicked'):
            self.scatterSeriesClicked = QScatterSeries()
            self.chart_1.addSeries(self.scatterSeriesClicked)

        self.scatterSeries.clear()
        self.scatterSeriesHuman.clear()
        self.scatterSeriesClicked.clear()
        
        for viewpointId in self.locationData:
            point = self.locationData[viewpointId]
            if self.scanId not in self.humanMotionData:
                self.humanMotionData[self.scanId] = {}
            if viewpointId == self.viewpointId:
                self.scatterSeriesClicked.append(point[0], point[1])
            if viewpointId in self.humanMotionData[self.scanId]:
                self.scatterSeriesHuman.append(point[0], point[1])
            else:
                self.scatterSeries.append(point[0], point[1])

    def onScatterClicked(self, point: QPointF):
        for viewpointId in self.locationData:
            if [point.x(), point.y()] == self.locationData[viewpointId][:2]:
                break
        if self.viewpointId != viewpointId:
            self.viewpointId = viewpointId
            self.scatterSeriesClicked.clear()
            self.scatterSeriesClicked.append(point.x(), point.y())
            self.location = self.locationData[self.viewpointId]
            self.textBrowser_viewpointID.setText(self.viewpointId)
            self.textBrowser_location.setText(f"X:{self.location[0]} Y:{self.location[1]}")
            self.showImage()
            self.updateHumanMotion()

def compute(posData, averageArea):
    viewNum = len(posData)
    humanNum = (viewNum * 2 // averageArea) + 1
    return viewNum, humanNum

if __name__ == '__main__':
    app = QApplication(sys.argv)
    GRAPHS = 'connectivity/'
    dataDir = os.getenv("HA3D_SIMULATOR_DATA_PATH")
    with open(GRAPHS+'scans.txt') as f:
        scans = [scan.strip() for scan in f.readlines()]
    with open('human-viewpoint_annotation/human_motion_text.json', 'r') as f:
        humanMotionData = json.load(f)
    with open('human-viewpoint_annotation/human_motion_model/region_motion_text.json', 'r') as f:
        regionMotionData = json.load(f)
    viewpointImageDir = os.path.join(dataDir, "data/v1/scans")
    modelVideoDir = os.path.join(dataDir, "samples_humanml_trans_enc_512_000200000_seed10_HC-VLN_text_prompts")
    with open(os.path.join(modelVideoDir, "results.txt")) as f:
        motionText = [motion.strip() for motion in f.readlines()]
    mainWindow = MyMainWindow(viewpointImageDir, modelVideoDir, scans, humanMotionData, regionMotionData, motionText)
    mainWindow.show()
    sys.exit(app.exec_())