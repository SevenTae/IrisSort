# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FSort.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import sys
from MLZY.network.qttest.finall import qtgui
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton


class Ui_FSort(QMainWindow):
    def setupUi(self, FSort):
        FSort.setObjectName("FSort")
        FSort.resize(864, 530)
        self.pushButton = QtWidgets.QPushButton(FSort)
        self.pushButton.setGeometry(QtCore.QRect(60, 390, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(FSort)
        self.pushButton_2.setGeometry(QtCore.QRect(370, 380, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label = QtWidgets.QLabel(FSort)
        self.label.setGeometry(QtCore.QRect(10, 20, 191, 16))
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(FSort)
        self.lineEdit.setGeometry(QtCore.QRect(70, 100, 113, 21))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(FSort)
        self.lineEdit_2.setGeometry(QtCore.QRect(340, 100, 113, 21))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(FSort)
        self.lineEdit_3.setGeometry(QtCore.QRect(70, 270, 111, 21))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_4 = QtWidgets.QLineEdit(FSort)
        self.lineEdit_4.setGeometry(QtCore.QRect(340, 260, 113, 21))
        self.lineEdit_4.setText("")
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.label_2 = QtWidgets.QLabel(FSort)
        self.label_2.setGeometry(QtCore.QRect(0, 50, 121, 31))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(FSort)
        self.label_3.setGeometry(QtCore.QRect(300, 50, 151, 41))
        self.label_3.setSizeIncrement(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(FSort)
        self.label_4.setGeometry(QtCore.QRect(10, 200, 121, 16))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(FSort)
        self.label_5.setGeometry(QtCore.QRect(300, 190, 111, 31))
        self.label_5.setObjectName("label_5")
        self.line = QtWidgets.QFrame(FSort)
        self.line.setGeometry(QtCore.QRect(540, 10, 20, 521))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label_6 = QtWidgets.QLabel(FSort)
        self.label_6.setGeometry(QtCore.QRect(590, 50, 141, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(FSort)
        self.label_7.setGeometry(QtCore.QRect(580, 220, 261, 101))
        font = QtGui.QFont()
        font.setPointSize(23)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")

        self.pushButton.clicked.connect(self.on_pushButton1_clicked)
        self.pushButton_2.clicked.connect(self.on_pushButton2_clicked)

        self.retranslateUi(FSort)
        QtCore.QMetaObject.connectSlotsByName(FSort)

    def retranslateUi(self, FSort):
        _translate = QtCore.QCoreApplication.translate
        FSort.setWindowTitle(_translate("FSort", "鸢尾花分类系统"))
        self.pushButton.setText(_translate("FSort", "预测"))
        self.pushButton_2.setText(_translate("FSort", "清空"))
        self.label.setText(_translate("FSort", "请输入鸢尾花各属性值："))
        self.label_2.setText(_translate("FSort", "SepalLength"))
        self.label_3.setText(_translate("FSort", "SepalWidth"))
        self.label_4.setText(_translate("FSort", "PetalLength"))
        self.label_5.setText(_translate("FSort", "PetalWidth"))
        self.label_6.setText(_translate("FSort", "预测结果"))
        self.label_7.setText(_translate("FSort", ""))

    def on_pushButton1_clicked(self):
            _translate = QtCore.QCoreApplication.translate
            sender = self.sender()
            print(sender.text() + '被点击')
            sele = float(self.lineEdit.text())
            sewd = float(self.lineEdit_2.text())
            pale = float(self.lineEdit_3.text())
            pawd = float(self.lineEdit_4.text())
            output = qtgui(sele,sewd,pale,pawd)

            self.label_7.setText(output)




    def on_pushButton2_clicked(self):
            # _translate = QtCore.QCoreApplication.translate
            sender = self.sender()
            print(sender.text() + '2被点击')
            self.lineEdit.setText("")
            self.lineEdit_2.setText("")
            self.lineEdit_3.setText("")
            self.lineEdit_4.setText("")
            self.label_7.setText("")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_FSort()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())