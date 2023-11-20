import sys
import os
import json
from form import Ui_Form
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtMultimediaWidgets import QVideoWidget
from compute_human_num import compute

class myMainWindow(Ui_Form,QMainWindow):
    def __init__(self,viewpoint_image_dir,model_video_dir, scan_list, human_motion_data, region_motion_data, motion_text):
        super(Ui_Form, self).__init__()
        self.viewpoint_image_dir = viewpoint_image_dir
        self.model_video_dir = model_video_dir
        self.scan_list = scan_list
        self.human_motion_data = human_motion_data
        self.region_motion_data = region_motion_data
        self.motion_text =  motion_text

        self.scan_id = self.scan_list[0]
        with open('con/pos_info/{}_pos_info.json'.format(self.scan_id), 'r') as f:
            self.location_data = json.load(f)
        self.viewpoint_list = [key for key in self.location_data]
        self.viewpoint_id = self.viewpoint_list[0]
        self.location = self.location_data[self.viewpoint_id]
        _,self.human_count = compute(self.location_data,67)

        self.region_list = [region for region in self.region_motion_data]
        self.region = self.region_list[0]
        self.human_motion_list = self.region_motion_data[self.region]
        self.human_motion = self.human_motion_list[0]
        self.human_motion_id = self.motion_text.index(f"{self.region}:{self.human_motion}")
        self.human_model_list = ['0','1','2']
        self.human_model_id = self.human_model_list[0]

        self.panorama_image_path = os.path.join(self.viewpoint_image_dir, self.scan_id, "matterport_panorama_images", self.viewpoint_id+'.jpg')
        self.feet_image_path = os.path.join(self.viewpoint_image_dir, self.scan_id, "matterport_skybox_images", self.viewpoint_id+'_skybox5_sami.jpg')
        self.model_video_path = os.path.join(self.model_video_dir, f"sample{self.human_motion_id:02d}_rep{int(self.human_model_id):02d}.mp4")
        if not os.path.exists(self.model_video_path):
            print("error")
            return
        
        self.setupUi(self)
        # textBrowser
        self.textBrowser_scanID.setText(self.scan_id)
        self.textBrowser_viewpointID.setText(self.viewpoint_id)
        self.textBrowser_location.setText(f"X:{self.location[0]} Y:{self.location[1]}")
        self.textBrowser_humanCount.setText(f"{self.human_count}")
        # comboBox initial
        self.comboBox_region.addItems(self.region_list)
        self.comboBox_humanMotion.addItems(self.human_motion_list)
        self.comboBox_3DmodelID.addItems(self.human_model_list)
        self.region = self.comboBox_region.currentText()
        self.human_motion = self.comboBox_humanMotion.currentText()
        self.human_model_id = self.comboBox_3DmodelID.currentText()
        self.comboBox_region.activated[str].connect(self.changeRegion)

        # button
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
        self.pushButton_save.clicked.connect(self.save)
        
        # frame initial
        pix_panorama = QPixmap(self.panorama_image_path)
        pix_feet = QPixmap(self.feet_image_path)
        self.label_frame1.setPixmap(pix_panorama)
        self.label_frame2.setPixmap(pix_feet)
        self.label_frame1.setScaledContents(True)  # 自适应QLabel大小
        self.label_frame2.setScaledContents(True)  # 自适应QLabel大小

        # video player
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.videowidget_motion)

 
    def preScan(self):
        index = self.scan_list.index(self.scan_id) - 1
        self.pushButton_scanID_next.setEnabled(True)
        # first scan
        if index == 0:
            self.pushButton_scanID_pre.setEnabled(False)
            
        self.scan_id = self.scan_list[index]
        with open('con/pos_info/{}_pos_info.json'.format(self.scan_id), 'r') as f:
            self.location_data = json.load(f)
        self.viewpoint_list = [key for key in self.location_data]
        self.viewpoint_id = self.viewpoint_list[0]
        self.location = self.location_data[self.viewpoint_id]
        _,self.human_count = compute(self.location_data,67)
        self.textBrowser_scanID.setText(self.scan_id)
        self.textBrowser_viewpointID.setText(self.viewpoint_id)
        self.textBrowser_location.setText(f"X:{self.location[0]} Y:{self.location[1]}")
        self.textBrowser_humanCount.setText(f"{self.human_count}")
    
    def nextScan(self):
        index = self.scan_list.index(self.scan_id) + 1
        self.pushButton_scanID_pre.setEnabled(True)
        # last scan
        if index == len(self.scan_list)-1:
            self.pushButton_scanID_next.setEnabled(False)
            
        self.scan_id = self.scan_list[index]
        with open('con/pos_info/{}_pos_info.json'.format(self.scan_id), 'r') as f:
            self.location_data = json.load(f)
        self.viewpoint_list = [key for key in self.location_data]
        self.viewpoint_id = self.viewpoint_list[0]
        self.location = self.location_data[self.viewpoint_id]
        _,self.human_count = compute(self.location_data,67)
        self.textBrowser_scanID.setText(self.scan_id)
        self.textBrowser_viewpointID.setText(self.viewpoint_id)
        self.textBrowser_location.setText(f"X:{self.location[0]} Y:{self.location[1]}")
        self.textBrowser_humanCount.setText(f"{self.human_count}")

    def preViewpoint(self):
        index = self.viewpoint_list.index(self.viewpoint_id) - 1
        self.pushButton_viewpointID_next.setEnabled(True)
        # first viewpoint
        if index == 0:
            self.pushButton_viewpointID_pre.setEnabled(False)
        self.viewpoint_id = self.viewpoint_list[index]
        self.location = self.location_data[self.viewpoint_id]
        self.textBrowser_viewpointID.setText(self.viewpoint_id)
        self.textBrowser_location.setText(f"X:{self.location[0]} Y:{self.location[1]}")
    
    def nextViewpoint(self):
        index = self.viewpoint_list.index(self.viewpoint_id) + 1
        self.pushButton_viewpointID_pre.setEnabled(True)
        # last viewpoint
        if index == len(self.viewpoint_list)-1:
            self.pushButton_viewpointID_next.setEnabled(False)
        self.viewpoint_id = self.viewpoint_list[index]
        self.location = self.location_data[self.viewpoint_id]
        self.textBrowser_viewpointID.setText(self.viewpoint_id)
        self.textBrowser_location.setText(f"X:{self.location[0]} Y:{self.location[1]}")
    
    def save(self):
        if self.viewpoint_id not in self.human_motion_data[self.scan_id]:
            self.human_motion_data[self.scan_id][self.viewpoint_id] = []
            self.human_motion_data[self.scan_id][self.viewpoint_id].append(self.comboBox_region.currentText() + ':' + self.comboBox_humanMotion.currentText())
            self.human_motion_data[self.scan_id][self.viewpoint_id].append(int(self.comboBox_3DmodelID.currentText()))
            self.human_motion_data[self.scan_id][self.viewpoint_id].append(0)
        else:
            # motion text
            self.human_motion_data[self.scan_id][self.viewpoint_id][0] = self.comboBox_region.currentText() + ':' + self.comboBox_humanMotion.currentText()
            # 3D model id
            self.human_motion_data[self.scan_id][self.viewpoint_id][1] = int(self.comboBox_3DmodelID.currentText())
            # model h angel
            self.human_motion_data[self.scan_id][self.viewpoint_id][2] = 0
        # save
        with open("human_motion_text.json", 'w') as f:
            json.dump(human_motion_data, f, indent=4)

    def changeRegion(self,region):
        #print("clear")
        self.comboBox_humanMotion.clear()
        self.human_motion_list = self.region_motion_data[region]
        self.comboBox_humanMotion.addItems(self.human_motion_list)
    
    def showImage(self):
        self.panorama_image_path = os.path.join(viewpoint_image_dir, self.scan_id, "matterport_panorama_images", self.viewpoint_id+'.jpg')
        self.feet_image_path = os.path.join(viewpoint_image_dir, self.scan_id, "matterport_skybox_images", self.viewpoint_id+'_skybox5_sami.jpg')
        if not os.path.exists(self.panorama_image_path):
            self.label_frame1.setText("No Panorama Image!")
        else:  
            pix_panorama = QPixmap(self.panorama_image_path)
            self.label_frame1.setPixmap(pix_panorama)
        
        if not os.path.exists(self.feet_image_path):
            self.label_frame2.setText("No skybox-feet Image!")
        else:
            pix_feet = QPixmap(self.feet_image_path)
            self.label_frame2.setPixmap(pix_feet)

    def openModelVideo(self):
        self.region = self.comboBox_region.currentText()
        self.human_motion = self.comboBox_humanMotion.currentText()
        self.human_motion_id = self.motion_text.index(f"{self.region}:{self.human_motion}")
        self.human_model_id = self.comboBox_3DmodelID.currentText()
        self.model_video_path = os.path.join(self.model_video_dir, f"sample{self.human_motion_id:02d}_rep{int(self.human_model_id):02d}.mp4")
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.model_video_path)))
        self.pushButton_stop.setEnabled(True)
        self.pushButton_play.setEnabled(False)
        self.player.play()

    def playModelVideo(self):
        self.pushButton_stop.setEnabled(True)
        self.pushButton_play.setEnabled(False)
        self.player.play()
        
        #print(self.player.availableMetaData())
 
    def stopModelVideo(self):
        self.pushButton_stop.setEnabled(False)
        self.pushButton_play.setEnabled(True)
        self.player.pause()
 
 
 
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     mainWindow = QMainWindow()
#     ui = Ui_MainWindow()
#     ui.setupUi(mainWindow)
#     mainWindow.show()
#     sys.exit(app.exec_())
 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    GRAPHS = 'connectivity/'
    # 每个建筑场景编号
    with open(GRAPHS+'scans.txt') as f:
        scans = [scan.strip() for scan in f.readlines()]
    with open('human_motion_text.json', 'r') as f:
        human_motion_data = json.load(f)
    with open('region_motion_text.json', 'r') as f:
        region_motion_data = json.load(f)
    viewpoint_image_dir = os.path.join("./","data/v1/scans")
    model_video_dir = os.path.join("/home/lmh/fsdownload","samples_humanml_trans_enc_512_000200000_seed10_HC-VLN_text_prompts")
    with open(os.path.join(model_video_dir, "results.txt")) as f:
        motion_text = [motion.strip() for motion in f.readlines()]
    mainWindow = myMainWindow(viewpoint_image_dir,model_video_dir, scans, human_motion_data, region_motion_data, motion_text)
    mainWindow.show()
    sys.exit(app.exec_())