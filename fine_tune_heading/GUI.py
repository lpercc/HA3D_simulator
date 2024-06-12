import sys
import os
import json
import trimesh
from form1 import Ui_Form
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtChart import *
from PyQt5.QtMultimediaWidgets import QVideoWidget

DOWNSIZED_WIDTH = 512
DOWNSIZED_HEIGHT = 512

class MyMainWindow(Ui_Form, QMainWindow):
    def __init__(self, viewpointImageDir, videoOutputDir, motionModelDir, scanList, humanMotionData, agentHeadingData):
        super(Ui_Form, self).__init__()
        self.viewpointImageDir = viewpointImageDir
        self.videoOutputDir = videoOutputDir
        self.motionModelDir = motionModelDir
        self.scanList = scanList
        self.humanMotionData = humanMotionData
        self.agentHeadingData = agentHeadingData

        # Initialize parameters
        self.scanId = self.scanList[0]

        # Get all viewpoints information (positions) of the building scene
        with open(f'con/pos_info/{self.scanId}_pos_info.json', 'r') as f:
            self.locationData = json.load(f)

        # Get all viewpoint relationships of the building scene
        with open(f'con/con_info/{self.scanId}_con_info.json', 'r') as f:
            self.connectionData = json.load(f)

        # Output path
        self.videoOutputPath = os.path.join(self.videoOutputDir, f"{self.scanId}")
        if not os.path.exists(self.videoOutputPath):
            os.makedirs(self.videoOutputPath)

        # Human Information
        self.humanViewpointList = [humanViewpoint for humanViewpoint in self.humanMotionData[self.scanId]]
        self.humanViewpointId = self.humanViewpointList[0]
        self.region = self.humanMotionData[self.scanId][self.humanViewpointId][0].split(":")[0]
        self.humanMotion = self.humanMotionData[self.scanId][self.humanViewpointId][0].split(":")[1]
        self.humanModelId = str(self.humanMotionData[self.scanId][self.humanViewpointId][1])
        self.humanHeading = self.humanMotionData[self.scanId][self.humanViewpointId][2]
        self.humanLocation = self.locationData[self.humanViewpointId]
        self.motionPath = os.path.join(self.motionModelDir, self.humanMotionData[self.scanId][self.humanViewpointId][0].replace(' ', '_').replace('/', '_'), f"{self.humanModelId}_obj")
        self.mesh = trimesh.load(os.path.join(self.motionPath, "frame000.obj"))

        # Agent Information
        self.agentViewpointList = [num for num, val in self.connectionData.items() if self.humanViewpointId in val['visible']]
        self.agentViewpointId = self.agentViewpointList[0]
        self.agentHeading = self.agentHeadingData[self.scanId][self.agentViewpointId][0]
        self.agentLocation = self.locationData[self.agentViewpointId]

        # Get the panorama view storage path of the building scene
        self.panoramaImagePath = os.path.join(self.viewpointImageDir, self.scanId, "matterport_skybox_images", f"{self.humanViewpointId}_skybox_small.jpg")
        self.background = concat(self.panoramaImagePath, DOWNSIZED_WIDTH)
        self.outputFramePath = "./fine_tune_heading/adjust.jpg"
        self.outputVideoPath = os.path.join(self.videoOutputPath, f"{self.agentViewpointId}.mp4")

        # Initialize render
        self.renderer = getRenderer(int(self.background.shape[1] / 4), self.background.shape[0])
        self.updateFusion()

        # Initialize UI
        self.setupUi(self)

        # Scan Information textBrowser
        self.textBrowser_scanID.setText(self.scanId)
        # Scan Information buttons
        self.pushButton_scanPrevious.setEnabled(False)
        self.pushButton_scanPrevious.clicked.connect(self.scanPrevious)
        self.pushButton_scanNext.clicked.connect(self.scanNext)

        # Human Information textBrowser
        self.textBrowser_humanViewpointID.setText(self.humanViewpointId)
        self.textBrowser_region.setText(self.region)
        self.textBrowser_humanMotion.setText(self.humanMotion)
        self.textBrowser_humanHeading.setText(f"{self.humanHeading}")
        self.textBrowser_humanLocation.setText(f"X:{self.humanLocation[0]} Y:{self.humanLocation[1]} Z:{self.humanLocation[2]}")

        # Human Information buttons
        self.pushButton_humanPrevious.setEnabled(False)
        self.pushButton_humanPrevious.clicked.connect(self.humanPrevious)
        self.pushButton_humanNext.clicked.connect(self.humanNext)

        # Human Heading dial
        self.dial_humanHeading.setRange(0, 360)
        self.dial_humanHeading.setNotchesVisible(True)
        self.dial_humanHeading.setWrapping(True)
        self.dial_humanHeading.setSingleStep(5)
        self.dial_humanHeading.setValue(int(self.humanHeading))
        self.dial_humanHeading.valueChanged.connect(self.onDialHumanHeadingChanged)

        # Agent Information textBrowser
        self.textBrowser_agentViewpointID.setText(self.agentViewpointId)
        self.textBrowser_agentHeading.setText(f"{self.agentHeading}")
        self.textBrowser_agentLocation.setText(f"X:{self.agentLocation[0]} Y:{self.agentLocation[1]} Z:{self.agentLocation[2]}")

        # Agent Information buttons
        self.pushButton_agentPrevious.setEnabled(False)
        self.pushButton_agentPrevious.clicked.connect(self.agentPrevious)
        self.pushButton_agentNext.clicked.connect(self.agentNext)

        # Agent Heading dial
        self.dial_agentHeading.setRange(0, 360)
        self.dial_agentHeading.setNotchesVisible(True)
        self.dial_agentHeading.setWrapping(True)
        self.dial_agentHeading.setSingleStep(5)
        self.dial_agentHeading.setValue(int(self.agentHeading))
        self.dial_agentHeading.valueChanged.connect(self.onDialAgentHeadingChanged)

        # Panorama Frame Label
        pixPanorama = QPixmap(self.outputFramePath)
        self.label_frame1.setPixmap(pixPanorama)
        self.label_frame1.setScaledContents(True)  # Adjust QLabel size

        # Fusion preview
        self.pushButton_fusionPreview.clicked.connect(self.updateVideo)
        self.pushButton_play.clicked.connect(self.playFusionVideo)
        self.pushButton_stop.clicked.connect(self.stopFusionVideo)
        self.pushButton_play.setEnabled(False)
        self.pushButton_stop.setEnabled(False)

        # Video player
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.widget_fusionVideo)

        # Save button
        self.pushButton_save.clicked.connect(self.headingAngleSave)

    def scanPrevious(self):
        index = self.scanList.index(self.scanId) - 1
        self.pushButton_scanNext.setEnabled(True)
        if index == 0:
            self.pushButton_scanPrevious.setEnabled(False)
        self.scanId = self.scanList[index]
        self.updateScan()

    def scanNext(self):
        index = self.scanList.index(self.scanId) + 1
        self.pushButton_scanPrevious.setEnabled(True)
        if index == len(self.scanList) - 1:
            self.pushButton_scanNext.setEnabled(False)
        self.scanId = self.scanList[index]
        self.updateScan()

    def updateScan(self):
        self.textBrowser_scanID.setText(self.scanId)
        print(self.scanList.index(self.scanId) + 1, ":", self.scanId)

        # Get all viewpoints information (positions) of the building scene
        with open(f'con/pos_info/{self.scanId}_pos_info.json', 'r') as f:
            self.locationData = json.load(f)

        # Get all viewpoint relationships of the building scene
        with open(f'con/con_info/{self.scanId}_con_info.json', 'r') as f:
            self.connectionData = json.load(f)

        # Output path
        self.videoOutputPath = os.path.join(self.videoOutputDir, f"{self.scanId}")
        if not os.path.exists(self.videoOutputPath):
            os.makedirs(self.videoOutputPath)

        self.humanViewpointList = [humanViewpoint for humanViewpoint in self.humanMotionData[self.scanId]]
        self.humanViewpointId = self.humanViewpointList[0]
        # Get the panorama view storage path of the building scene
        self.panoramaImagePath = os.path.join(self.viewpointImageDir, self.scanId, "matterport_skybox_images", f"{self.humanViewpointId}_skybox_small.jpg")
        self.pushButton_agentPrevious.setEnabled(False)
        self.pushButton_agentNext.setEnabled(True)
        self.pushButton_humanNext.setEnabled(True)
        self.pushButton_humanPrevious.setEnabled(False)
        self.updateHuman()

    def humanPrevious(self):
        index = self.humanViewpointList.index(self.humanViewpointId) - 1
        self.pushButton_humanNext.setEnabled(True)
        if index == 0:
            self.pushButton_humanPrevious.setEnabled(False)
        self.humanViewpointId = self.humanViewpointList[index]
        self.updateHuman()

    def humanNext(self):
        index = self.humanViewpointList.index(self.humanViewpointId) + 1
        self.pushButton_humanPrevious.setEnabled(True)
        if index == len(self.humanViewpointList) - 1:
            self.pushButton_humanNext.setEnabled(False)
        try:
            self.humanViewpointId = self.humanViewpointList[index]
            self.updateHuman()
        except IndexError:
            self.pushButton_humanNext.setEnabled(False)

    def updateHuman(self):
        self.region = self.humanMotionData[self.scanId][self.humanViewpointId][0].split(":")[0]
        self.humanMotion = self.humanMotionData[self.scanId][self.humanViewpointId][0].split(":")[1]
        self.humanModelId = str(self.humanMotionData[self.scanId][self.humanViewpointId][1])
        self.humanHeading = self.humanMotionData[self.scanId][self.humanViewpointId][2]
        self.humanLocation = self.locationData[self.humanViewpointId]
        self.motionPath = os.path.join(self.motionModelDir, self.humanMotionData[self.scanId][self.humanViewpointId][0].replace(' ', '_').replace('/', '_'), f"{self.humanModelId}_obj")
        self.mesh = trimesh.load(os.path.join(self.motionPath, "frame000.obj"))
        self.textBrowser_humanViewpointID.setText(self.humanViewpointId)
        self.textBrowser_region.setText(self.region)
        self.textBrowser_humanMotion.setText(self.humanMotion)
        self.textBrowser_humanHeading.setText(f"{self.humanHeading}")
        self.textBrowser_humanLocation.setText(f"X:{self.humanLocation[0]} Y:{self.humanLocation[1]} Z:{self.humanLocation[2]}")
        self.agentViewpointList = [num for num, val in self.connectionData.items() if self.humanViewpointId in val['visible']]
        self.agentViewpointId = self.agentViewpointList[0]
        self.dial_humanHeading.setValue(int(self.humanHeading))
        self.pushButton_agentPrevious.setEnabled(False)
        self.pushButton_agentNext.setEnabled(True)
        self.updateAgent()

    def onDialHumanHeadingChanged(self, value):
        self.humanMotionData[self.scanId][self.humanViewpointId][2] = value
        self.humanHeading = self.humanMotionData[self.scanId][self.humanViewpointId][2]
        self.textBrowser_humanHeading.setText(f"{self.humanHeading}")
        self.updateImage()

    def agentPrevious(self):
        index = self.agentViewpointList.index(self.agentViewpointId) - 1
        self.pushButton_agentNext.setEnabled(True)
        if index == 0:
            self.pushButton_agentPrevious.setEnabled(False)
        self.agentViewpointId = self.agentViewpointList[index]
        self.updateAgent()

    def agentNext(self):
        index = self.agentViewpointList.index(self.agentViewpointId) + 1
        self.pushButton_agentPrevious.setEnabled(True)
        if index == len(self.agentViewpointList) - 1:
            self.pushButton_agentNext.setEnabled(False)
        try:
            self.agentViewpointId = self.agentViewpointList[index]
            self.updateAgent()
        except IndexError:
            self.pushButton_agentNext.setEnabled(False)

    def updateAgent(self):
        try:
            self.agentHeading = self.agentHeadingData[self.scanId][self.agentViewpointId][0]
        except KeyError:
            self.agentHeadingData[self.scanId][self.agentViewpointId] = [0]
            self.agentHeading = self.agentHeadingData[self.scanId][self.agentViewpointId][0]
        self.agentLocation = self.locationData[self.agentViewpointId]
        self.textBrowser_agentViewpointID.setText(self.agentViewpointId)
        self.textBrowser_agentHeading.setText(f"{self.agentHeading}")
        self.textBrowser_agentLocation.setText(f"X:{self.agentLocation[0]} Y:{self.agentLocation[1]} Z:{self.agentLocation[2]}")
        self.dial_agentHeading.setValue(int(self.agentHeading))
        self.panoramaImagePath = os.path.join(self.viewpointImageDir, self.scanId, "matterport_skybox_images", f"{self.agentViewpointId}_skybox_small.jpg")
        self.updateImage()

    def onDialAgentHeadingChanged(self, value):
        self.agentHeadingData[self.scanId][self.agentViewpointId][0] = value
        self.agentHeading = self.agentHeadingData[self.scanId][self.agentViewpointId][0]
        self.textBrowser_agentHeading.setText(f"{self.agentHeading}")
        self.updateImage()

    def updateImage(self):
        self.background = concat(self.panoramaImagePath, DOWNSIZED_WIDTH)
        self.updateFusion()
        pixPanorama = QPixmap(self.outputFramePath)
        self.label_frame1.setPixmap(pixPanorama)

    def updateVideo(self):
        self.outputVideoPath = os.path.join(self.videoOutputPath, f"{self.agentViewpointId}.mp4")
        meshes = []
        objFiles = [f for f in os.listdir(self.motionPath) if f.endswith('.obj')]
        sortedObjFiles = sorted(objFiles)
        for objFile in sortedObjFiles[:120]:
            objPath = os.path.join(self.motionPath, objFile)
            mesh = trimesh.load(objPath)
            meshes.append(mesh)
        print(self.outputVideoPath)
        renderVideo(meshes, 
                    self.background, 
                    self.agentLocation, 
                    self.agentHeading, 
                    self.humanLocation, 
                    self.humanHeading, 
                    self.renderer, 
                    self.outputVideoPath, 
                    self.agentViewpointId,
                    self.scanId,
                    self.humanViewpointId)

        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.outputVideoPath))) 
        self.pushButton_stop.setEnabled(True)
        self.pushButton_play.setEnabled(False)
        self.player.play()

    def playFusionVideo(self):
        self.pushButton_stop.setEnabled(True)
        self.pushButton_play.setEnabled(False)
        self.player.play()

    def stopFusionVideo(self):
        self.pushButton_stop.setEnabled(False)
        self.pushButton_play.setEnabled(True)
        self.player.pause()

    def headingAngleSave(self):
        with open("human-viewpoint_annotation/human_motion_text.json", 'w') as f:
            json.dump(self.humanMotionData, f, indent=4)

        with open("con/heading_info.json", 'w') as f:
            json.dump(self.agentHeadingData, f, indent=4)

    def updateFusion(self):
        renderFirstFrame(self.mesh.copy(), 
                         self.background, 
                         self.agentLocation, 
                         self.agentHeading, 
                         self.humanLocation, 
                         self.humanHeading, 
                         self.renderer, 
                         self.outputFramePath, 
                         self.agentViewpointId,
                         self.scanId,
                         self.humanViewpointId)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    GRAPHS = 'connectivity/'
    HA3D_SIMULATOR_PATH = os.environ.get("HA3D_SIMULATOR_PATH")
    sys.path.append(HA3D_SIMULATOR_PATH)
    dataDir = os.getenv("HA3D_SIMULATOR_DATA_PATH")
    # Read scan list
    with open(GRAPHS+'scans.txt') as f:
        scanList = [scan.strip() for scan in f.readlines()]
    with open('human-viewpoint_annotation/human_motion_text.json', 'r') as f:
        humanMotionData = json.load(f)
    # Read agent heading information
    with open("con/heading_info.json", 'r') as f:
        agentHeadingData = json.load(f)

    viewpointImageDir = os.path.join(dataDir, "data/v1/scans")
    motionModelDir = os.path.join(dataDir, "human_motion_meshes")
    videoOutputDir = os.path.join(HA3D_SIMULATOR_PATH, "fine_tune_heading/video")

    from src.render.renderer import getRenderer
    from src.render.rendermdm import renderFirstFrame, renderVideo
    from src.utils.concat_skybox import concat
    mainWindow = MyMainWindow(viewpointImageDir, videoOutputDir, motionModelDir, scanList, humanMotionData, agentHeadingData)
    mainWindow.show()
    sys.exit(app.exec_())
