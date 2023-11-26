import sys
import os
import json
import trimesh
import imageio.v2 as imageio
from src.render.renderer import get_renderer
from src.render.rendermdm import render_first_frame
from form1 import Ui_Form
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtChart import *
from PyQt5.QtMultimediaWidgets import QVideoWidget


class myMainWindow(Ui_Form,QMainWindow):
    def __init__(self,viewpoint_image_dir, video_output_dir, motion_model_dir, scan_list, human_motion_data, agent_heading_data):
        super(Ui_Form, self).__init__()
        self.viewpoint_image_dir = viewpoint_image_dir
        self.video_output_dir = video_output_dir
        self.motion_model_dir = motion_model_dir
        self.scan_list = scan_list
        self.human_motion_data = human_motion_data
        self.agent_heading_data = agent_heading_data
        
        # Initialize parameters
        self.scan_id = self.scan_list[0]
        
        ## 获取建筑场景所有视点信息（视点位置）
        with open('con/pos_info/{}_pos_info.json'.format(self.scan_id), 'r') as f:
            self.location_data = json.load(f)
        
        ## 获取建筑场景所有视点信息（视点之间的关系）
        with open('con/con_info/{}_con_info.json'.format(self.scan_id), 'r') as f:
            self.connection_data = json.load(f)
            #print(len(connection_data))
        
        ## 获取该建筑场景的全景视图存放路径
        self.panorama_image_path = os.path.join(self.viewpoint_image_dir, f"{self.scan_id}/matterport_panorama_images")
        
        ## 输出路径
        self.video_output_path = os.path.join(self.video_output_dir, f"{self.scan_id}/matterport_panorama_video")
        if not os.path.exists(self.video_output_path):
            os.makedirs(self.video_output_path)
        
        ## Human Information
        self.human_viewpoint_list = [human_viewpoint for human_viewpoint in self.human_motion_data[self.scan_id]]
        self.human_viewpoint_id = self.human_viewpoint_list[0]
        self.region = self.human_motion_data[self.scan_id][self.human_viewpoint_id][0].split(":")[0]
        self.human_motion = self.human_motion_data[self.scan_id][self.human_viewpoint_id][0].split(":")[1]
        self.human_heading = self.human_motion_data[self.scan_id][self.human_viewpoint_id][2]
        self.human_location = self.location_data[self.human_viewpoint_id]
        self.motion_path = os.path.join(self.motion_model_dir, self.scan_id, self.human_viewpoint_id+"_obj")
        self.mesh = trimesh.load(os.path.join(self.motion_path,"frame000.obj"))
        ## Agent Information
        self.agent_viewpoint_list = []
        for num, val in self.connection_data.items():
            if self.human_viewpoint_id in val['visible']:
                self.agent_viewpoint_list.append(num)
        self.agent_viewpoint_id = self.agent_viewpoint_list[0]
        self.agent_heading = self.agent_heading_data[self.scan_id][self.agent_viewpoint_id][0]
        self.agent_location = self.location_data[self.agent_viewpoint_id]
        self.background = imageio.imread(os.path.join(self.panorama_image_path,self.agent_viewpoint_id+'.jpg'))
        self.output_frame_path = "./adjust.jpg"
        
        # Initialize render
        self.renderer = get_renderer(self.background.shape[1], self.background.shape[0])
        self.updateFusion()
        
        # Initialize Ui
        self.setupUi(self)
        
        ## Scan Information textBrowser
        self.textBrowser_scanID.setText(self.scan_id)
        
        ## Human Information textBrowser
        self.textBrowser_humanViewpointID.setText(self.human_viewpoint_id)
        self.textBrowser_region.setText(self.region)
        self.textBrowser_humanMotion.setText(self.human_motion)
        self.textBrowser_humanHeading.setText(f"{self.human_heading}")
        self.textBrowser_humanLocation.setText(f"X:{self.human_location[0]} Y:{self.human_location[1]} Z:{self.human_location[2]}")

        ## Human Information button
        self.pushButton_humanPrevious.setEnabled(False)
        self.pushButton_humanPrevious.clicked.connect(self.humanPrevious)
        self.pushButton_humanNext.clicked.connect(self.humanNext)

        ## Human Heading dial
        self.dial_humanHeading.setRange(0, 360)
        self.dial_humanHeading.setNotchesVisible(True)
        self.dial_humanHeading.setWrapping(True)
        self.dial_humanHeading.setSingleStep(5)
        self.dial_humanHeading.setValue(int(self.human_heading))
        self.dial_humanHeading.valueChanged.connect(self.on_dial_humanHeading_changed)
        
        ## Agent Information textBrowser
        self.textBrowser_agentViewpointID.setText(self.agent_viewpoint_id)
        self.textBrowser_agentHeading.setText(f"{self.agent_heading}")
        self.textBrowser_agentLocation.setText(f"X:{self.agent_location[0]} Y:{self.agent_location[1]} Z:{self.agent_location[2]}")

        ## Agent Information button
        self.pushButton_agentPrevious.setEnabled(False)
        self.pushButton_agentPrevious.clicked.connect(self.agentPrevious)
        self.pushButton_agentNext.clicked.connect(self.agentNext)
        
        ## Agent Heading dial
        self.dial_agentHeading.setRange(0, 360)
        self.dial_agentHeading.setNotchesVisible(True)
        self.dial_agentHeading.setWrapping(True)
        self.dial_agentHeading.setSingleStep(5)
        self.dial_agentHeading.setValue(int(self.agent_heading))
        self.dial_agentHeading.valueChanged.connect(self.on_dial_agentHeading_changed)

        ## Panorama Frame Label
        pix_panorama = QPixmap(self.output_frame_path)
        self.label_frame1.setPixmap(pix_panorama)
        self.label_frame1.setScaledContents(True)  # 自适应QLabel大小

        ## save
        self.pushButton_save.clicked.connect(self.headingAngleSave)


    def humanPrevious(self):
        index = self.human_viewpoint_list.index(self.human_viewpoint_id) - 1
        self.pushButton_humanNext.setEnabled(True)
        if index == 0:
            self.pushButton_humanPrevious.setEnabled(False)
        self.human_viewpoint_id = self.human_viewpoint_list[index]
        self.updateHuman()

    def humanNext(self):
        index = self.human_viewpoint_list.index(self.human_viewpoint_id) + 1
        self.pushButton_humanPrevious.setEnabled(True)
        if index == len(self.human_viewpoint_list)-1:
            self.pushButton_humanNext.setEnabled(False)
        self.human_viewpoint_id = self.human_viewpoint_list[index]
        self.updateHuman()

    def updateHuman(self):
        self.region = self.human_motion_data[self.scan_id][self.human_viewpoint_id][0].split(":")[0]
        self.human_motion = self.human_motion_data[self.scan_id][self.human_viewpoint_id][0].split(":")[1]
        self.human_heading = self.human_motion_data[self.scan_id][self.human_viewpoint_id][2]
        self.human_location = self.location_data[self.human_viewpoint_id]
        self.motion_path = os.path.join(self.motion_model_dir, self.scan_id, self.human_viewpoint_id+"_obj")
        self.mesh = trimesh.load(os.path.join(self.motion_path,"frame000.obj"))
        self.textBrowser_humanViewpointID.setText(self.human_viewpoint_id)
        self.textBrowser_region.setText(self.region)
        self.textBrowser_humanMotion.setText(self.human_motion)
        self.textBrowser_humanHeading.setText(f"{self.human_heading}")
        self.textBrowser_humanLocation.setText(f"X:{self.human_location[0]} Y:{self.human_location[1]} Z:{self.human_location[2]}")
        self.agent_viewpoint_list = []
        for num, val in self.connection_data.items():
            if self.human_viewpoint_id in val['visible']:
                self.agent_viewpoint_list.append(num)
        self.agent_viewpoint_id = self.agent_viewpoint_list[0]
        self.updateAgent()

    def on_dial_humanHeading_changed(self, value):
        self.human_motion_data[self.scan_id][self.human_viewpoint_id][2] = value
        self.human_heading = self.human_motion_data[self.scan_id][self.human_viewpoint_id][2]
        self.textBrowser_humanHeading.setText(f"{self.human_heading}")
        self.updateImage()

    def agentPrevious(self):
        index = self.agent_viewpoint_list.index(self.agent_viewpoint_id) - 1
        self.pushButton_agentNext.setEnabled(True)
        if index == 0:
            self.pushButton_agentPrevious.setEnabled(False)
        self.agent_viewpoint_id = self.agent_viewpoint_list[index]
        self.updateAgent()

    def agentNext(self):
        index = self.agent_viewpoint_list.index(self.agent_viewpoint_id) + 1
        self.pushButton_agentPrevious.setEnabled(True)
        if index == len(self.agent_viewpoint_list)-1:
            self.pushButton_agentNext.setEnabled(False)
        self.agent_viewpoint_id = self.agent_viewpoint_list[index]
        self.updateAgent()

    def updateAgent(self):
        try:
            self.agent_heading = self.agent_heading_data[self.scan_id][self.agent_viewpoint_id][0]
        except KeyError:
            self.agent_heading_data[self.scan_id][self.agent_viewpoint_id] = [0]
            self.agent_heading = self.agent_heading_data[self.scan_id][self.agent_viewpoint_id][0]
        self.agent_location = self.location_data[self.agent_viewpoint_id]
        self.textBrowser_agentViewpointID.setText(self.agent_viewpoint_id)
        self.textBrowser_agentHeading.setText(f"{self.agent_heading}")
        self.textBrowser_agentLocation.setText(f"X:{self.agent_location[0]} Y:{self.agent_location[1]} Z:{self.agent_location[2]}")
        self.dial_agentHeading.setValue(int(self.agent_heading))
        self.updateImage()

    def on_dial_agentHeading_changed(self, value):
        self.agent_heading_data[self.scan_id][self.agent_viewpoint_id][0] = value
        self.agent_heading = self.agent_heading_data[self.scan_id][self.agent_viewpoint_id][0]
        self.textBrowser_agentHeading.setText(f"{self.agent_heading}")
        self.updateImage()

    def updateImage(self):
        self.background = imageio.imread(os.path.join(self.panorama_image_path,self.agent_viewpoint_id+'.jpg'))
        #self.renderer = get_renderer(self.background.shape[1], self.background.shape[0])
        self.updateFusion()
        pix_panorama = QPixmap(self.output_frame_path)
        self.label_frame1.setPixmap(pix_panorama)

    def headingAngleSave(self):
        with open("human_motion_text.json", 'w') as f:
            json.dump(self.human_motion_data, f, indent=4)

        with open("con/heading_info.json", 'w') as f:
            json.dump(self.agent_heading_data, f, indent=4)

    def updateFusion(self):
        render_first_frame(self.mesh.copy(), 
                            self.background, 
                            self.agent_location, 
                            self.agent_heading, 
                            self.human_location, 
                            self.human_heading, 
                            self.renderer, 
                            self.output_frame_path, 
                            self.agent_viewpoint_id,
                            self.scan_id,
                            self.human_viewpoint_id)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    GRAPHS = 'connectivity/'
    # 每个建筑场景编号
    with open(GRAPHS+'scans.txt') as f:
        scan_list = [scan.strip() for scan in f.readlines()]
    with open('human_motion_text.json', 'r') as f:
        human_motion_data = json.load(f)
    # 每个建筑场景中的视点视角朝向
    with open("con/heading_info.json", 'r') as f:
        agent_heading_data = json.load(f)

    viewpoint_image_dir = os.path.join("./","data/v1/scans")
    motion_model_dir = os.path.join("./","human_motion_meshes")
    video_output_dir = os.path.join("./", "result/v1/scans")

    mainWindow = myMainWindow(viewpoint_image_dir, video_output_dir, motion_model_dir, scan_list, human_motion_data, agent_heading_data)
    mainWindow.show()
    sys.exit(app.exec_())