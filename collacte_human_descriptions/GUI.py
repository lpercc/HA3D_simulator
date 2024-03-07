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
from compute_human_num import compute
import random
import math

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
        self.pushButton_random_loc.clicked.connect(self.randomHumanLocation)
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

        # chart initial
        self.create_coordinates()
        self.create_scatter_series()
 
    def preScan(self):
        index = self.scan_list.index(self.scan_id) - 1
        print(f"Number {index} Scan")
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
        # upgrade chart
        self.update_coordinates()
        self.update_scatter_series()
        # update Human motion
        self.updateHumanMotion()
    
    def nextScan(self):
        index = self.scan_list.index(self.scan_id) + 1
        print(f"Number {index} Scan")
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
        # upgrade chart
        self.update_coordinates()
        self.update_scatter_series()
        # update Human motion
        self.updateHumanMotion()

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
        self.scatter_series_clicked.clear()
        self.scatter_series_clicked.append(self.location[0], self.location[1])
        # update Human motion
        self.updateHumanMotion()

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
        self.scatter_series_clicked.clear()
        self.scatter_series_clicked.append(self.location[0], self.location[1])
        # update Human motion
        self.updateHumanMotion()

    def save(self):
        if self.scan_id not in self.human_motion_data:
            self.human_motion_data[self.scan_id] = {}
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
        print(f"Number of annotated scan:{len(self.human_motion_data)}")
        # upgrade chart
        self.update_scatter_series()

    def randomHumanLocation(self):
        #清除现在的人物点
        self.human_motion_data[self.scan_id] = {}
        viewpoint_list = set(self.viewpoint_list)  # 转换为集合以提高效率
        for i in range(self.human_count):
            excluded_viewpoints = set(self.human_motion_data[self.scan_id])
            # 创建一个不包含排除元素的候选列表
            candidate_list = list(viewpoint_list - excluded_viewpoints)
            while True:
                # 随机选择一个元素
                if candidate_list:
                    selected_viewpoint = random.choice(candidate_list)
                    #print(selected_viewpoint)
                    
                if self.human_motion_data[self.scan_id] == {}:
                    break
                else:
                    # 每个人物的距离>3m
                    distance_list = []
                    for human_point in [self.location_data[viewpoint_id] for viewpoint_id in self.human_motion_data[self.scan_id]]:
                        selected_point = self.location_data[selected_viewpoint]
                        # 计算距离
                        distance = math.sqrt((human_point[0] - selected_point[0])**2 + 
                                            (human_point[1] - selected_point[1])**2 + 
                                            (human_point[2] - selected_point[2])**2)
                        distance_list.append(distance)
                    #print(distance_list,sum(distance_list) > len(distance_list) * 3)
                    #break
                    if sum(distance_list) > len(distance_list) * 3:
                        break
            #加入    
            self.human_motion_data[self.scan_id][selected_viewpoint] = ["",0,0]
        self.update_scatter_series()
    
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

    def updateHumanMotion(self):
        if self.scan_id not in self.human_motion_data: return
        if self.viewpoint_id in self.human_motion_data[self.scan_id]:
            human = self.human_motion_data[self.scan_id][self.viewpoint_id]
            print(human)
            if human == ["",0,0]: return
            region = human[0].split(":")[0]
            self.changeRegion(region)
            human_motion = human[0].split(":")[1]
            human_model_id = str(human[1])
            self.comboBox_region.setCurrentText(region)
            self.comboBox_humanMotion.setCurrentText(human_motion)
            self.comboBox_3DmodelID.setCurrentText(human_model_id)
            self.openModelVideo()
    
    def openModelVideo(self):
        self.region = self.comboBox_region.currentText()
        self.human_motion = self.comboBox_humanMotion.currentText()
        #print(self.human_motion)
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
 
    def create_coordinates(self):
        data_points = [self.location_data[viewpoint_id] for viewpoint_id in self.location_data]
        # 绘制坐标轴
        axis_x = QLineSeries(self.chart_1)
        self.chart_1.addSeries(axis_x)
        axis_y = QLineSeries(self.chart_1)
        self.chart_1.addSeries(axis_y)
        self.chart_1.legend().hide()
        
        # 初始化最大值和最小值为第一个点的坐标
        min_x = max_x = data_points[0][0]
        min_y = max_y = data_points[0][1]

        # 遍历所有点，更新最大值（上取整）和最小值（下取整）,
        for x, y, _ in data_points:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
        min_x = int(min_x-1)
        max_x = int(max_x+1)
        min_y = int(min_y-1)
        max_y = int(max_y+1)
        # 创建并设置 X 轴
        self.axisX = QValueAxis()
        self.axisX.setRange(min_x, max_x)  # 设置轴的范围
        self.axisX.setTickCount(max_x-min_x+1)   # 设置刻度数量，包括两端点
        self.axisX.setLabelFormat("%d")  # 设置标签格式，这里是整数
        self.chart_1.setAxisX(self.axisX, axis_x)

        # 创建并设置 Y 轴
        self.axisY = QValueAxis()
        self.axisY.setRange(min_y, max_y)
        self.axisY.setTickCount(max_y-min_y+1)
        self.axisY.setLabelFormat("%d")
        self.chart_1.setAxisY(self.axisY, axis_y)

    def update_coordinates(self):
        data_points = [self.location_data[viewpoint_id] for viewpoint_id in self.location_data]
        # 检查是否已经有自定义轴，如果没有，则创建
        if not hasattr(self, 'axisX'):
            self.axisX = QValueAxis()
            self.chart_1.setAxisX(self.axisX, self.scatter_series)

        if not hasattr(self, 'axisY'):
            self.axisY = QValueAxis()
            self.chart_1.setAxisY(self.axisY, self.scatter_series)        
        # 初始化最大值和最小值为第一个点的坐标
        min_x = max_x = data_points[0][0]
        min_y = max_y = data_points[0][1]

        # 遍历所有点，更新最大值（上取整）和最小值（下取整）,
        for x, y, _ in data_points:
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

    def create_scatter_series(self):
        # 创建散点系列
        # 普通点
        self.scatter_series = QScatterSeries()
        self.scatter_series.setName("Normal Points")
        self.scatter_series.setMarkerShape(QScatterSeries.MarkerShapeCircle)  # 设置散点形状为圆形
        self.scatter_series.setMarkerSize(10)  # 设置散点大小
        self.scatter_series.setColor(QColor(Qt.blue))  # 设置散点颜色为蓝色
        # 人物点
        self.scatter_series_h = QScatterSeries()
        self.scatter_series_h.setName("Human Points")
        self.scatter_series_h.setMarkerShape(QScatterSeries.MarkerShapeCircle)  # 设置散点形状为圆形
        self.scatter_series_h.setMarkerSize(10)  # 设置散点大小
        self.scatter_series_h.setColor(QColor(Qt.red))  # 设置散点颜色为蓝色

        # 选中点
        self.scatter_series_clicked = QScatterSeries()
        self.scatter_series_clicked.setName("Clicked Points")
        self.scatter_series_clicked.setMarkerShape(QScatterSeries.MarkerShapeCircle)  # 设置散点形状为圆形
        self.scatter_series_clicked.setMarkerSize(10)  # 设置散点大小
        self.scatter_series_clicked.setColor(QColor(Qt.green))  # 设置散点颜色为绿色

        # 添加数据点
        for viewpoint_id in self.location_data:
            point = self.location_data[viewpoint_id]
            if self.scan_id not in self.human_motion_data:
                self.human_motion_data[self.scan_id] = {}
            if viewpoint_id == self.viewpoint_id:
                self.scatter_series_clicked.append(point[0], point[1])
            if viewpoint_id in self.human_motion_data[self.scan_id]:
                self.scatter_series_h.append(point[0], point[1])
            else:
                self.scatter_series.append(point[0], point[1])
            

        # 将散点系列添加到图表中
        self.chart_1.addSeries(self.scatter_series)
        self.chart_1.addSeries(self.scatter_series_h)
        self.chart_1.addSeries(self.scatter_series_clicked)

        # 关联坐标轴（如果已经创建了自定义坐标轴）
        self.chart_1.setAxisX(self.axisX, self.scatter_series)
        self.chart_1.setAxisY(self.axisY, self.scatter_series)
        self.chart_1.setAxisX(self.axisX, self.scatter_series_h)
        self.chart_1.setAxisY(self.axisY, self.scatter_series_h)
        self.chart_1.setAxisX(self.axisX, self.scatter_series_clicked)
        self.chart_1.setAxisY(self.axisY, self.scatter_series_clicked)
        # 连接 clicked 信号到槽函数
        self.scatter_series.clicked.connect(self.on_scatter_clicked)
        self.scatter_series_h.clicked.connect(self.on_scatter_clicked)

    def update_scatter_series(self):
        # 确保散点系列存在
        if not hasattr(self, 'scatter_series'):
            self.scatter_series = QScatterSeries()
            self.chart_1.addSeries(self.scatter_series)
        if not hasattr(self, 'scatter_series_h'):
            self.scatter_series_h = QScatterSeries()
            self.chart_1.addSeries(self.scatter_series_h)
        if not hasattr(self, 'scatter_series_clicked'):
            self.scatter_series_clicked = QScatterSeries()
            self.chart_1.addSeries(self.scatter_series_clicked)

        # 清除现有数据点
        self.scatter_series.clear()
        self.scatter_series_h.clear()
        self.scatter_series_clicked.clear()
        
        # 添加数据点
        for viewpoint_id in self.location_data:
            point = self.location_data[viewpoint_id]
            if self.scan_id not in self.human_motion_data:
                self.human_motion_data[self.scan_id] = {}
            if viewpoint_id == self.viewpoint_id:
                self.scatter_series_clicked.append(point[0], point[1])
            if viewpoint_id in self.human_motion_data[self.scan_id]:
                self.scatter_series_h.append(point[0], point[1])
            else:
                self.scatter_series.append(point[0], point[1])
        
    def on_scatter_clicked(self, point: QPointF):
        for viewpoint_id in self.location_data:
            if [point.x(), point.y()] == self.location_data[viewpoint_id][:2]:
                print(f"Clicked point: ({point.x()}, {point.y()}), ID:{viewpoint_id}")
                break
        if self.viewpoint_id != viewpoint_id:
            self.viewpoint_id = viewpoint_id
            self.scatter_series_clicked.clear()
            self.scatter_series_clicked.append(point.x(), point.y())
            self.location = self.location_data[self.viewpoint_id]
            self.textBrowser_viewpointID.setText(self.viewpoint_id)
            self.textBrowser_location.setText(f"X:{self.location[0]} Y:{self.location[1]}")
            self.showImage()
            self.updateHumanMotion()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    GRAPHS = 'connectivity/'
    basic_dir = "/media/lmh/backend"
    # 每个建筑场景编号
    with open(GRAPHS+'scans.txt') as f:
        scans = [scan.strip() for scan in f.readlines()]
    with open('human_motion_text.json', 'r') as f:
        human_motion_data = json.load(f)
    with open('region_motion_text.json', 'r') as f:
        region_motion_data = json.load(f)
    viewpoint_image_dir = os.path.join(basic_dir,"HC-VLN_dataset_/data/v1/scans")
    print(viewpoint_image_dir)
    model_video_dir = os.path.join(basic_dir,"samples_humanml_trans_enc_512_000200000_seed10_HC-VLN_text_prompts")
    with open(os.path.join(model_video_dir, "results.txt")) as f:
        motion_text = [motion.strip() for motion in f.readlines()]
    mainWindow = myMainWindow(viewpoint_image_dir,model_video_dir, scans, human_motion_data, region_motion_data, motion_text)
    mainWindow.show()
    sys.exit(app.exec_())