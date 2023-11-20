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
    def __init__(self,viewpoint_image_dir,model_video_dir, scan_list, human_motion_data, region_motion_data):
        super(Ui_Form, self).__init__()
        
        self.scan_list = scan_list
        self.scan_id = self.scan_list[0]
        with open('con/pos_info/{}_pos_info.json'.format(self.scan_id), 'r') as f:
            self.location_data = json.load(f)
        self.viewpoint_list = [key for key in self.location_data]
        self.viewpoint_id = self.viewpoint_list[0]
        self.location = self.location_data[self.viewpoint_id]
        _,self.human_count = compute(self.location_data,67)

        self.region_motion_data = region_motion_data
        self.region_list = [region for region in self.region_motion_data]
        
        self.region = self.region_list[0]
        self.human_motion_list = self.region_motion_data[self.region]

        self.human_motion = self.human_motion_list[0]
        self.human_motion_id = self.human_motion_list.index(self.human_motion)
        self.human_model_list = ['0','1','2']
        self.human_model_id = self.human_model_list[0]

        self.panorama_image_path = os.path.join(viewpoint_image_dir, self.scan_id, "matterport_panorama_images", self.viewpoint_id+'.jpg')
        self.feet_image_path = os.path.join(viewpoint_image_dir, self.scan_id, "matterport_skybox_images", self.viewpoint_id+'_skybox5_sami.jpg')
        self.model_video_path = os.path.join(model_video_dir, f"sample{self.human_motion_id:02d}_rep{int(self.human_model_id):02d}.mp4")
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
        # 播放器
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.videowidget_motion)
 
    def showImage(self):
        self.frame_1
    def changeRegion(self,region):
        #print("clear")
        self.comboBox_humanMotion.clear()
        self.human_motion_list = self.region_motion_data[region]
        self.comboBox_humanMotion.addItems(self.human_motion_list)
    def openVideoFile(self):
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.model_video_path)))
        self.player.play()
        print(self.player.availableMetaData())
 
    def playVideo(self):
        self.player.play()
 
    def pauseVideo(self):
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
    model_video_dir = os.path.join("../fsdownload","samples_humanml_trans_enc_512_000200000_seed10_HC-VLN_text_prompts")
    mainWindow = myMainWindow(viewpoint_image_dir,model_video_dir, scans, human_motion_data, region_motion_data)
    mainWindow.show()
    sys.exit(app.exec_())