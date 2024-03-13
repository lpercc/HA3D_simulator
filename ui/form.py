# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'form.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1080, 839)
        self.Image_label = QtWidgets.QLabel(Form)
        self.Image_label.setGeometry(QtCore.QRect(10, 10, 600, 450))
        self.Image_label.setObjectName("Image_label")
        self.chart_1 = QChart()
        self.chart_view = QChartView(self.chart_1, Form)
        self.chart_view.setGeometry(QtCore.QRect(620, 10, 450, 450))
        self.chart_view.setObjectName("chart_1")
        self.widget = QtWidgets.QWidget(Form)
        self.widget.setGeometry(QtCore.QRect(10, 470, 1061, 361))
        self.widget.setObjectName("widget")
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_18.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.labelScanID = QtWidgets.QLabel(self.widget)
        self.labelScanID.setObjectName("labelScanID")
        self.horizontalLayout.addWidget(self.labelScanID)
        self.lineEditScanID = QtWidgets.QLineEdit(self.widget)
        self.lineEditScanID.setObjectName("lineEditScanID")
        self.horizontalLayout.addWidget(self.lineEditScanID)
        self.pushButtonSearchBuilding = QtWidgets.QPushButton(self.widget)
        self.pushButtonSearchBuilding.setObjectName("pushButtonSearchBuilding")
        self.horizontalLayout.addWidget(self.pushButtonSearchBuilding)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.pushButtonPreviousBuilding = QtWidgets.QPushButton(self.widget)
        self.pushButtonPreviousBuilding.setObjectName("pushButtonPreviousBuilding")
        self.horizontalLayout_10.addWidget(self.pushButtonPreviousBuilding)
        self.pushButtonNextBuilding = QtWidgets.QPushButton(self.widget)
        self.pushButtonNextBuilding.setObjectName("pushButtonNextBuilding")
        self.horizontalLayout_10.addWidget(self.pushButtonNextBuilding)
        self.verticalLayout.addLayout(self.horizontalLayout_10)
        self.verticalLayout_3.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.PathID_label = QtWidgets.QLabel(self.widget)
        self.PathID_label.setObjectName("PathID_label")
        self.horizontalLayout_2.addWidget(self.PathID_label)
        self.lineEditPathID = QtWidgets.QLineEdit(self.widget)
        self.lineEditPathID.setObjectName("lineEditPathID")
        self.horizontalLayout_2.addWidget(self.lineEditPathID)
        self.pushButtonSearchPath = QtWidgets.QPushButton(self.widget)
        self.pushButtonSearchPath.setObjectName("pushButtonSearchPath")
        self.horizontalLayout_2.addWidget(self.pushButtonSearchPath)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.pushButtonPreviousPath = QtWidgets.QPushButton(self.widget)
        self.pushButtonPreviousPath.setObjectName("pushButtonPreviousPath")
        self.horizontalLayout_9.addWidget(self.pushButtonPreviousPath)
        self.pushButtonNextPath = QtWidgets.QPushButton(self.widget)
        self.pushButtonNextPath.setObjectName("pushButtonNextPath")
        self.horizontalLayout_9.addWidget(self.pushButtonNextPath)
        self.verticalLayout_2.addLayout(self.horizontalLayout_9)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.Heading_label = QtWidgets.QLabel(self.widget)
        self.Heading_label.setObjectName("Heading_label")
        self.horizontalLayout_4.addWidget(self.Heading_label)
        self.labelPathHeading = QtWidgets.QLabel(self.widget)
        self.labelPathHeading.setObjectName("labelPathHeading")
        self.horizontalLayout_4.addWidget(self.labelPathHeading)
        self.horizontalLayout_12.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.Distance_label = QtWidgets.QLabel(self.widget)
        self.Distance_label.setObjectName("Distance_label")
        self.horizontalLayout_3.addWidget(self.Distance_label)
        self.labelPathDistance = QtWidgets.QLabel(self.widget)
        self.labelPathDistance.setObjectName("labelPathDistance")
        self.horizontalLayout_3.addWidget(self.labelPathDistance)
        self.horizontalLayout_12.addLayout(self.horizontalLayout_3)
        self.verticalLayout_3.addLayout(self.horizontalLayout_12)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.horizontalLayout_11.addWidget(self.label)
        self.listWidgetPathViewpoint = QtWidgets.QListWidget(self.widget)
        self.listWidgetPathViewpoint.setEnabled(True)
        self.listWidgetPathViewpoint.setObjectName("listWidgetPathViewpoint")
        self.horizontalLayout_11.addWidget(self.listWidgetPathViewpoint)
        self.verticalLayout_3.addLayout(self.horizontalLayout_11)
        self.horizontalLayout_18.addLayout(self.verticalLayout_3)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.Instruction1_label = QtWidgets.QLabel(self.widget)
        self.Instruction1_label.setObjectName("Instruction1_label")
        self.horizontalLayout_5.addWidget(self.Instruction1_label)
        self.textEditInstruction1 = QtWidgets.QTextEdit(self.widget)
        self.textEditInstruction1.setMaximumSize(QtCore.QSize(16777215, 100))
        self.textEditInstruction1.setObjectName("textEditInstruction1")
        self.horizontalLayout_5.addWidget(self.textEditInstruction1)
        self.verticalLayout_4.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.Instruction2_label = QtWidgets.QLabel(self.widget)
        self.Instruction2_label.setObjectName("Instruction2_label")
        self.horizontalLayout_6.addWidget(self.Instruction2_label)
        self.textEditInstruction2 = QtWidgets.QTextEdit(self.widget)
        self.textEditInstruction2.setMaximumSize(QtCore.QSize(16777215, 100))
        self.textEditInstruction2.setObjectName("textEditInstruction2")
        self.horizontalLayout_6.addWidget(self.textEditInstruction2)
        self.verticalLayout_4.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.Instruction3_label = QtWidgets.QLabel(self.widget)
        self.Instruction3_label.setObjectName("Instruction3_label")
        self.horizontalLayout_7.addWidget(self.Instruction3_label)
        self.textEditInstruction3 = QtWidgets.QTextEdit(self.widget)
        self.textEditInstruction3.setMaximumSize(QtCore.QSize(16777215, 100))
        self.textEditInstruction3.setObjectName("textEditInstruction3")
        self.horizontalLayout_7.addWidget(self.textEditInstruction3)
        self.verticalLayout_4.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_18.addLayout(self.verticalLayout_4)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.ViewpointID_label = QtWidgets.QLabel(self.widget)
        self.ViewpointID_label.setObjectName("ViewpointID_label")
        self.horizontalLayout_14.addWidget(self.ViewpointID_label)
        self.labelViewpointID = QtWidgets.QLabel(self.widget)
        self.labelViewpointID.setObjectName("labelViewpointID")
        self.horizontalLayout_14.addWidget(self.labelViewpointID)
        self.verticalLayout_5.addLayout(self.horizontalLayout_14)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_8.addWidget(self.label_2)
        self.labelAgentHeading = QtWidgets.QLabel(self.widget)
        self.labelAgentHeading.setObjectName("labelAgentHeading")
        self.horizontalLayout_8.addWidget(self.labelAgentHeading)
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_8.addWidget(self.label_3)
        self.labelAgentElevation = QtWidgets.QLabel(self.widget)
        self.labelAgentElevation.setObjectName("labelAgentElevation")
        self.horizontalLayout_8.addWidget(self.labelAgentElevation)
        self.verticalLayout_5.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_13.addWidget(self.label_4)
        self.label_5 = QtWidgets.QLabel(self.widget)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_13.addWidget(self.label_5)
        self.labelAgentLocationX = QtWidgets.QLabel(self.widget)
        self.labelAgentLocationX.setMinimumSize(QtCore.QSize(20, 0))
        self.labelAgentLocationX.setObjectName("labelAgentLocationX")
        self.horizontalLayout_13.addWidget(self.labelAgentLocationX)
        self.label_6 = QtWidgets.QLabel(self.widget)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_13.addWidget(self.label_6)
        self.labelAgentLocationY = QtWidgets.QLabel(self.widget)
        self.labelAgentLocationY.setMinimumSize(QtCore.QSize(20, 0))
        self.labelAgentLocationY.setObjectName("labelAgentLocationY")
        self.horizontalLayout_13.addWidget(self.labelAgentLocationY)
        self.label_7 = QtWidgets.QLabel(self.widget)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_13.addWidget(self.label_7)
        self.labelAgentLocationZ = QtWidgets.QLabel(self.widget)
        self.labelAgentLocationZ.setMinimumSize(QtCore.QSize(20, 0))
        self.labelAgentLocationZ.setObjectName("labelAgentLocationZ")
        self.horizontalLayout_13.addWidget(self.labelAgentLocationZ)
        self.label_8 = QtWidgets.QLabel(self.widget)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_13.addWidget(self.label_8)
        self.labelAgentVFOV = QtWidgets.QLabel(self.widget)
        self.labelAgentVFOV.setObjectName("labelAgentVFOV")
        self.horizontalLayout_13.addWidget(self.labelAgentVFOV)
        self.verticalLayout_5.addLayout(self.horizontalLayout_13)
        self.verticalLayout_8.addLayout(self.verticalLayout_5)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_9 = QtWidgets.QLabel(self.widget)
        self.label_9.setObjectName("label_9")
        self.verticalLayout_6.addWidget(self.label_9)
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.checkBoxCreatePath = QtWidgets.QCheckBox(self.widget)
        self.checkBoxCreatePath.setObjectName("checkBoxCreatePath")
        self.horizontalLayout_15.addWidget(self.checkBoxCreatePath)
        self.checkBoxUpgradePath = QtWidgets.QCheckBox(self.widget)
        self.checkBoxUpgradePath.setObjectName("checkBoxUpgradePath")
        self.horizontalLayout_15.addWidget(self.checkBoxUpgradePath)
        self.verticalLayout_6.addLayout(self.horizontalLayout_15)
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.pushButtonRandomBeginning = QtWidgets.QPushButton(self.widget)
        self.pushButtonRandomBeginning.setObjectName("pushButtonRandomBeginning")
        self.horizontalLayout_16.addWidget(self.pushButtonRandomBeginning)
        self.verticalLayout_6.addLayout(self.horizontalLayout_16)
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.pushButtonPathBack = QtWidgets.QPushButton(self.widget)
        self.pushButtonPathBack.setObjectName("pushButtonPathBack")
        self.horizontalLayout_17.addWidget(self.pushButtonPathBack)
        self.pushButtonPathSave = QtWidgets.QPushButton(self.widget)
        self.pushButtonPathSave.setObjectName("pushButtonPathSave")
        self.horizontalLayout_17.addWidget(self.pushButtonPathSave)
        self.verticalLayout_6.addLayout(self.horizontalLayout_17)
        self.verticalLayout_8.addLayout(self.verticalLayout_6)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_10 = QtWidgets.QLabel(self.widget)
        self.label_10.setObjectName("label_10")
        self.verticalLayout_7.addWidget(self.label_10)
        self.pushButtonGenerateHuman = QtWidgets.QPushButton(self.widget)
        self.pushButtonGenerateHuman.setObjectName("pushButtonGenerateHuman")
        self.verticalLayout_7.addWidget(self.pushButtonGenerateHuman)
        self.pushButtonGenerateInstructions = QtWidgets.QPushButton(self.widget)
        self.pushButtonGenerateInstructions.setObjectName("pushButtonGenerateInstructions")
        self.verticalLayout_7.addWidget(self.pushButtonGenerateInstructions)
        self.verticalLayout_8.addLayout(self.verticalLayout_7)
        self.horizontalLayout_18.addLayout(self.verticalLayout_8)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.Image_label.setText(_translate("Form", "ImageLabel"))
        self.labelScanID.setText(_translate("Form", "Scan ID:"))
        self.pushButtonSearchBuilding.setText(_translate("Form", "OK"))
        self.pushButtonPreviousBuilding.setText(_translate("Form", "Previous Building"))
        self.pushButtonNextBuilding.setText(_translate("Form", "Next Building"))
        self.PathID_label.setText(_translate("Form", "Path ID:"))
        self.pushButtonSearchPath.setText(_translate("Form", "OK"))
        self.pushButtonPreviousPath.setText(_translate("Form", "Previous Path"))
        self.pushButtonNextPath.setText(_translate("Form", "Next Path"))
        self.Heading_label.setText(_translate("Form", "Heading:"))
        self.labelPathHeading.setText(_translate("Form", "TextLabel"))
        self.Distance_label.setText(_translate("Form", "Distance:"))
        self.labelPathDistance.setText(_translate("Form", "TextLabel"))
        self.label.setText(_translate("Form", "Path Viewpoints"))
        self.Instruction1_label.setText(_translate("Form", "Instruction 1:"))
        self.Instruction2_label.setText(_translate("Form", "Instruction 2:"))
        self.Instruction3_label.setText(_translate("Form", "Instruction 3:"))
        self.ViewpointID_label.setText(_translate("Form", "Agent Viewpoint ID:"))
        self.labelViewpointID.setText(_translate("Form", "12345"))
        self.label_2.setText(_translate("Form", "Agent Heading:"))
        self.labelAgentHeading.setText(_translate("Form", "0"))
        self.label_3.setText(_translate("Form", "Agent Elevation:"))
        self.labelAgentElevation.setText(_translate("Form", "0"))
        self.label_4.setText(_translate("Form", "Agent Location:"))
        self.label_5.setText(_translate("Form", "X:"))
        self.labelAgentLocationX.setText(_translate("Form", "0"))
        self.label_6.setText(_translate("Form", "Y:"))
        self.labelAgentLocationY.setText(_translate("Form", "0"))
        self.label_7.setText(_translate("Form", "Z:"))
        self.labelAgentLocationZ.setText(_translate("Form", "0"))
        self.label_8.setText(_translate("Form", "Agent VFOV:"))
        self.labelAgentVFOV.setText(_translate("Form", "60"))
        self.label_9.setText(_translate("Form", "Dataset Label"))
        self.checkBoxCreatePath.setText(_translate("Form", "Create Path"))
        self.checkBoxUpgradePath.setText(_translate("Form", "Upgrade Path"))
        self.pushButtonRandomBeginning.setText(_translate("Form", "Random Beginning"))
        self.pushButtonPathBack.setText(_translate("Form", "BACK"))
        self.pushButtonPathSave.setText(_translate("Form", "SAVE"))
        self.label_10.setText(_translate("Form", "Generative AI"))
        self.pushButtonGenerateHuman.setText(_translate("Form", "Generate Human"))
        self.pushButtonGenerateInstructions.setText(_translate("Form", "Generate Instructions"))
from PyQt5.QtChart import QChart, QChartView


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
