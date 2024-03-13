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
from data_annotation import *

class myMainWindow(Ui_Form,QMainWindow):
    def __init__(self, HC_data_path,basic_data_dir):
        super(Ui_Form, self).__init__()
        self.HC_data_path = HC_data_path
        self.basic_data_dir =  os.path.join(basic_data_dir, "data/v1/scans")
        self.hc_data, self.path_max = read_HC_data(HC_data_path)

        self.setupUi(self)

        # button
        self.Previous_pushButton.setEnabled(True)
        self.Previous_pushButton.clicked.connect(self.prePath)
        self.Previous_pushButton.clicked.connect(self.upgradeVideo)
        self.Next_pushButton.setEnabled(True)
        self.Next_pushButton.clicked.connect(self.nextPath)
        self.Next_pushButton.clicked.connect(self.upgradeVideo)

        self.Insert_pushButton.setEnabled(True)
        self.Insert_pushButton.clicked.connect(self.insertPoint)
        self.Delete_pushButton.setEnabled(True)
        self.Delete_pushButton.clicked.connect(self.deletePoint)

        self.Save_pushButton.clicked.connect(self.save)



        # Panorama Frame Label
        self.Image_label.setScaledContents(True)  # 自适应QLabel大小

        # chart initial
        self.create_coordinates()
        self.create_scatter_series()
        # video player
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.pointView_widget)
        self.player.mediaStatusChanged.connect(self.check_media_status)
        # Path initial
        self.index = 0
        self.upgradePath(index=self.index, target_path_id=6567)

    def prePath(self):
        self.index = self.index - 1
        self.upgradePath(self.index)
        if self.path_id == 1: 
            self.Previous_pushButton.setEnabled(False)
        self.Next_pushButton.setEnabled(True)
        return
        
    def nextPath(self):
        self.index = self.index + 1
        self.upgradePath(self.index)
        if self.path_id == self.path_max: 
            self.Next_pushButton.setEnabled(False)
        self.Previous_pushButton.setEnabled(True)
        return
    
    def upgradeViewpoint(self, target_viewpoint_id):
        self.viewpoint_id = target_viewpoint_id
        self.path_unobstructed_points_ = get_unobstructed_points_(self.viewpoint_id, self.connection_data)
        self.ViewpointID_textBrowser.setText(self.viewpoint_id)
        self.upgradeVideo()

    def upgradePath(self, index=None, target_path_id=None):
        if target_path_id is not None:
            # 使用列表推导式查找索引
            indexes = [i for i, d in enumerate(self.hc_data) if d.get('path_id') == target_path_id]
            # 检查是否找到了索引，并取出第一个（如果存在的话）
            index = indexes[0] if indexes else None
        self.scan_id = self.hc_data[index]["scan"]
        self.path_id = self.hc_data[index]["path_id"]
        self.distance = self.hc_data[index]["distance"]
        self.heading = self.hc_data[index]["heading"]
        self.instructions = self.hc_data[index]["instructions"]
        self.path = self.hc_data[index]["path"]
        self.viewpoint_id = self.path[0]
        self.connection_data = read_connection_data('con/con_info/{}_con_info.json'.format(self.scan_id))
        self.position_data = read_position_data('con/pos_info/{}_pos_info.json'.format(self.scan_id))
        self.path_unobstructed_points = get_unobstructed_points(self.path, self.connection_data)
        self.path_unobstructed_points_ = get_unobstructed_points_(self.viewpoint_id, self.connection_data)
        
        # textBrowser
        self.ScanID_textBrowser.setText(self.scan_id)
        self.PathID_textBrowser.setText(str(self.path_id))
        self.Distance_textBrowser.setText(str(self.distance))
        self.Heading_textBrowser.setText(str(self.heading))
        self.ViewpointID_textBrowser.setText(self.viewpoint_id)

        # textEdit
        self.Instruction1_textEdit.setText(self.instructions[0])
        self.Instruction2_textEdit.setText(self.instructions[1])
        self.Instruction3_textEdit.setText(self.instructions[2])
        # upgrade chart
        self.upgrade_coordinates()
        self.upgrade_scatter_series()
        # upgrade video
        self.upgradeVideo()
    
    def upgradeVideo(self):
        video_path = os.path.join(self.basic_data_dir, self.scan_id, "matterport_panorama_video",f"{self.viewpoint_id}.mp4")
        print(video_path)
        if os.path.exists(video_path):
            self.Image_label.lower()  # 将 label 降到底层
            self.pointView_widget.raise_()  # 将 video_widget 提升到顶层
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
            self.player.play()
        else:
            self.pointView_widget.lower()  # 将 label 提升到顶层
            self.Image_label.raise_()  # 将 video_widget 降到底层
            self.upgradeImage()
    
    def check_media_status(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.player.play()  # 重新开始播放
    
    def upgradeImage(self):
        image_path = os.path.join(self.basic_data_dir, self.scan_id, "matterport_panorama_images", f"{self.viewpoint_id}.jpg")
        print(image_path)
        if not os.path.exists(image_path):
            print("No image or video found!!")
            return
        pix_panorama = QPixmap(image_path)
        self.Image_label.setPixmap(pix_panorama)
        
    def create_coordinates(self):
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

    def upgrade_coordinates(self):
        all_points = self.path  + self.path_unobstructed_points
        data_points = [self.position_data[viewpoint_id] for viewpoint_id in all_points]
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
        
        # 路径点
        self.scatter_series_p = QScatterSeries()
        self.scatter_series_p.setName("Path Points")
        self.scatter_series_p.setMarkerShape(QScatterSeries.MarkerShapeCircle)  # 设置散点形状为圆形
        self.scatter_series_p.setMarkerSize(15)  # 设置散点大小
        self.scatter_series_p.setColor(QColor(Qt.red))  # 设置散点颜色为red

        # 创建线系列
        self.line_series = QLineSeries()
        self.line_series.setColor(QColor(Qt.red))  # 设置线颜色为红色

        # 选中点
        self.scatter_series_clicked = QScatterSeries()
        self.scatter_series_clicked.setName("Clicked Points")
        self.scatter_series_clicked.setMarkerShape(QScatterSeries.MarkerShapeCircle)  # 设置散点形状为圆形
        self.scatter_series_clicked.setMarkerSize(10)  # 设置散点大小
        self.scatter_series_clicked.setColor(QColor(Qt.green))  # 设置散点颜色为绿色

        # 将散点系列添加到图表中
        self.chart_1.addSeries(self.scatter_series)
        self.chart_1.addSeries(self.scatter_series_p)
        self.chart_1.addSeries(self.scatter_series_clicked)

        # 关联坐标轴（如果已经创建了自定义坐标轴）
        self.chart_1.setAxisX(self.axisX, self.scatter_series)
        self.chart_1.setAxisY(self.axisY, self.scatter_series)
        self.chart_1.setAxisX(self.axisX, self.scatter_series_p)
        self.chart_1.setAxisY(self.axisY, self.scatter_series_p)
        self.chart_1.setAxisX(self.axisX, self.scatter_series_clicked)
        self.chart_1.setAxisY(self.axisY, self.scatter_series_clicked)
        
        # 将线系列添加到图表中
        self.chart_1.addSeries(self.line_series)

        # 关联线系列到坐标轴
        self.chart_1.setAxisX(self.axisX, self.line_series)
        self.chart_1.setAxisY(self.axisY, self.line_series)

        # 连接 clicked 信号到槽函数
        self.scatter_series.clicked.connect(self.on_scatter_clicked)
        self.scatter_series_p.clicked.connect(self.on_scatter_clicked)

    def upgrade_scatter_series(self):
        # 确保散点系列存在
        if not hasattr(self, 'scatter_series'):
            self.scatter_series = QScatterSeries()
            self.chart_1.addSeries(self.scatter_series)
        if not hasattr(self, 'scatter_series_p'):
            self.scatter_series_h = QScatterSeries()
            self.chart_1.addSeries(self.scatter_series_p)
        if not hasattr(self, 'scatter_series_clicked'):
            self.scatter_series_clicked = QScatterSeries()
            self.chart_1.addSeries(self.scatter_series_clicked)

        # 清除现有数据点
        self.scatter_series.clear()
        self.scatter_series_p.clear()
        self.scatter_series_clicked.clear()
        self.line_series.clear()  # 清除现有线数据
            
        # 添加数据点
        for viewpoint_id in self.path_unobstructed_points:
            point = self.position_data[viewpoint_id]
            self.scatter_series.append(point[0], point[1])
        for viewpoint_id in self.path:
            point = self.position_data[viewpoint_id]
            self.scatter_series_p.append(point[0], point[1])
            self.line_series.append(point[0], point[1])  # 将点添加到线系列
        point = self.position_data[self.viewpoint_id]
        self.scatter_series_clicked.append(point[0], point[1])
            
    def on_scatter_clicked(self, point: QPointF):
        for viewpoint_id in self.position_data:
            if [point.x(), point.y()] == self.position_data[viewpoint_id][:2]:
                print(f"Clicked point: ({point.x()}, {point.y()}), ID:{viewpoint_id}")
                break
        if self.viewpoint_id != viewpoint_id:
            self.scatter_series_clicked.clear()
            self.scatter_series_clicked.append(point.x(), point.y())
            self.upgradeViewpoint(viewpoint_id)
            
    def insertPoint(self):
        inserted_point = self.viewpoint_id
        pos_point = self.path[-1]
        self.path = insert_point(pos_point, inserted_point, self.path)
        self.path_unobstructed_points = get_unobstructed_points(self.path, self.connection_data)
        # upgrade chart
        self.upgrade_coordinates()
        self.upgrade_scatter_series()
        return
    
    def deletePoint(self):
        self.path = delete_point(self.viewpoint_id, self.path)
        self.path_unobstructed_points = get_unobstructed_points(self.path, self.connection_data)
        # upgrade chart
        self.upgrade_coordinates()
        self.upgrade_scatter_series()
        return

    def save(self):
        # 使用列表推导式查找索引
        indexes = [i for i, d in enumerate(self.hc_data) if d.get('path_id') == self.path_id]
        # 检查是否找到了索引，并取出第一个（如果存在的话）
        index = indexes[0] if indexes else None
        self.hc_data[index]["instructions"] = self.instructions
        self.hc_data[index]["path"] = self.path
        save_HC_data(HC_data_path, self.hc_data)
        self.upgradePath(index)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    HC_data_path = "./HC-VLN/path.json"
    basic_data_dir = os.getenv('VLN_DATA_DIR')
    mainWindow = myMainWindow(HC_data_path,basic_data_dir)
    mainWindow.show()
    sys.exit(app.exec_())