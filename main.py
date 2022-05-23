import sys
import os
from PyQt5 import QtGui
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
import sqlite3 as lite
import time

import numpy as np
import cv2
import torch
from torch import nn as nn
import torchvision
from torchvision import transforms

from create_account import *

#config device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform image
transform = transforms.Compose([transforms.ToTensor()])


def inference_image(img, model, device, transform):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img)
    img = img.resize_(1, img.shape[0], img.shape[1], img.shape[2])
    output = model(img.to(device))
    _, pred = torch.max(output, 1)
    pred = pred.cpu().numpy()[0]
    return pred

#============LOAD MODEL ==============
model = torchvision.models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)
 
state_dict = torch.load('checkpoints/resnet18_10epochs.pth', map_location='cuda:0')
model.load_state_dict(state_dict)
model.eval()
model = model.to(device)

path_db = os.getcwd() + '/data_login.db'
print(path_db)
create_connection(path_db)


'''
class WelcomeScreen(QDialog):
    def __init__(self):
        super(WelcomeScreen, self).__init__()
        loadUi('welcome.ui', self)
        self.login.clicked.connect(self.gotoLogin)
        self.create_account.clicked.connect(self.gotoCreateAccount)

    #===================================
    #Function to change different screen
    #===================================
    def gotoLogin(self):
        login = LoginScreen()
        widget.addWidget(login)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def gotoCreateAccount(self):
        create = CreateAccountScreen()
        widget.addWidget(create)
        widget.setCurrentIndex(widget.currentIndex() + 1)
'''

def convert_cv_qt(cv_img):
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    p = convert_to_Qt_format.scaled(1600, 941, Qt.KeepAspectRatio)
    return QPixmap.fromImage(p)

class CreateAccountScreenForPatient(QDialog):
    def __init__(self):
        super(CreateAccountScreenForPatient, self).__init__()
        loadUi('createacc_for_patient.ui', self)
        self.password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.confirm_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.signup.clicked.connect(self.signupFunction)
        self.return_function.clicked.connect(self.returnFunction)

    def signupFunction(self):
        username = self.username.text()
        password = self.password.text()
        role = 'patient'
        confirm_password = self.confirm_password.text()

        con = lite.connect(path_db)
        cur = con.cursor()
        query_user = "SELECT username FROM login_info where username='{}'".format(username)
        cur.execute(query_user)
        if cur.fetchone():
            self.error.setText('Username is exist. Please choose differently username.')
        else:
            if len(username) == 0 or len(password) == 0 or len(confirm_password) == 0:
                self.error.setText('Please fill all fields!')
            
            
            elif password != confirm_password:
                self.error.setText('Passwords do not match.')
            else:
                new_info = [username, password, role]
                cur.execute('insert into login_info values(?, ?, ?)', new_info)
                self.error.setText('Account successfully created.')
        con.commit()
        con.close()

    def returnFunction(self):
        function_screen = ChooseFunctionScreenForNurse()
        widget.addWidget(function_screen)
        widget.setCurrentIndex(widget.currentIndex() + 1)

class CreateAccountScreenForNurse(QDialog):
    def __init__(self):
        super(CreateAccountScreenForNurse, self).__init__()
        loadUi('createacc_for_nurse.ui', self)
        self.password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.confirm_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.signup.clicked.connect(self.signupFunction)
        self.return_function.clicked.connect(self.returnFunction)

        self.position_now = ''

    def returnFunction(self):
        next_screen = ChooseFunctionScreenForDev()
        widget.addWidget(next_screen)
        widget.setCurrentIndex(widget.currentIndex() + 1)


    def signupFunction(self):
        username = self.username.text()
        password = self.password.text()
        role = 'nurse'
        confirm_password = self.confirm_password.text()

        con = lite.connect(path_db)
        cur = con.cursor()
        query_user = "SELECT username FROM login_info where username='{}'".format(username)
        cur.execute(query_user)
        if cur.fetchone():
            self.error.setText('Username is exist. Please choose differently username.')
        else:
            if len(username) == 0 or len(password) == 0 or len(confirm_password) == 0:
                self.error.setText('Please fill all fields!')
            
            
            elif password != confirm_password:
                self.error.setText('Passwords do not match.')
            else:
                new_info = [username, password, role]
                cur.execute('insert into login_info values(?, ?, ?)', new_info)
                self.error.setText('Account successfully created.')
        con.commit()
        con.close()

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super(VideoThread, self).__init__()
        self.run_flag = True

        self.position = ''

    def run(self):
        #capture from web cam
        self.run_flag = True
        self.cap = cv2.VideoCapture(0)    # run video stream webcam

        while self.run_flag:
            ret, frame = self.cap.read()

            # predict position
            predict = inference_image(frame, model, device, transform)
            label_frame = 'class ' + str(predict+1)

            self.position = label_frame

            frame = cv2.resize(frame, (1600, 941))
            frame = cv2.putText(frame, label_frame, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=3)
            # print(frame.shape)
            if ret:
                self.change_pixmap_signal.emit(frame)

        self.cap.release()
    
    def stop(self):
        #sets run flag to False and waits for thread to finish
        self.run_flag = False

class SleepPositionScreenForNurse(QDialog):
    def __init__(self):
        super(SleepPositionScreenForNurse, self).__init__()
        loadUi('sp_screen_for_nurse.ui', self)

        self.exit.clicked.connect(self.return_listFunction)
        self.start.clicked.connect(self.start_capture_video)
        self.stop.clicked.connect(self.stop_capture_video)

        self.thread = VideoThread()

    # start show video on QLabel
    def start_capture_video(self):
        self.thread.change_pixmap_signal.connect(self.update_image)

        #start thread
        self.thread.start()
        
    #stop show video
    def stop_capture_video(self):
        self.thread.change_pixmap_signal.disconnect()
        self.thread.stop()

    # updata frame
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.frame.setPixmap(qt_img)

    #convert original image to qt image
    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(1600, 941, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    #function to return screen before
    def return_listFunction(self):
        if self.thread.isRunning():
            self.thread.stop()
        listFunctionNew = ChooseFunctionScreenForNurse()
        widget.addWidget(listFunctionNew)
        widget.setCurrentIndex(widget.currentIndex() + 1)

class SleepPositionScreenForPatient(QDialog):
    def __init__(self):
        super(SleepPositionScreenForPatient, self).__init__()
        loadUi('screen_sp_patient.ui', self)
        
        self.exit.clicked.connect(self.return_listFunction)
        self.start.clicked.connect(self.start_capture_video)
        self.stop.clicked.connect(self.stop_capture_video)

        self.thread = VideoThread()

        self.position_now = ''
        self.time1 = 0
        self.time2 = 0

    # start show video on QLabel
    def start_capture_video(self):
        self.thread.change_pixmap_signal.connect(self.update_image)

        #start thread
        self.thread.start()

        #time for capture image
        self.time1 = int(time.time())

    def get_position_now(self):
        self.position_now = self.thread.position
        return self.position_now
        
    #stop show video
    def stop_capture_video(self):
        self.thread.change_pixmap_signal.disconnect()
        self.thread.stop()

    # updata frame
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.frame.setPixmap(qt_img)

        self.time2 = int(time.time())
        print('Time: ', self.time2)
        #setup time to write position to log
        if self.time2 - self.time1 == 4:
            print(self.get_position_now())
            self.log.append(self.get_position_now())
            self.time1 = self.time2
            print('Time: ', self.time2, self.time1)

    #convert original image to qt image
    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(1600, 941, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    #function to return screen before
    def return_listFunction(self):
        if self.thread.isRunning():
            self.thread.stop()
        listFunctionNew = LoginScreen()
        widget.addWidget(listFunctionNew)
        widget.setCurrentIndex(widget.currentIndex() + 1)

class LoginScreen(QDialog):
    def __init__(self):
        super(LoginScreen, self).__init__()
        loadUi('login.ui', self)
        self.setFixedSize(1920, 1080)
        self.password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.login.clicked.connect(self.loginfunction)
        # self.background.setStyleSheet('background-image: url(background_image.png);')

    def loginfunction(self):
        '''
        Check username and password to login app.
        '''
        user = self.username.text()
        password = self.password.text()

        #connect database and check infomation's account
        con = lite.connect(path_db)
        cur = con.cursor()
        query_user = "SELECT username FROM login_info where username='{}'".format(user)
        cur.execute(query_user)
        if cur.fetchone():
            if len(user) == 0 or len(password) == 0:
                self.error.setText('Please input all fields!')
            else:            
                #get password from database
                query_password = 'SELECT password FROM login_info WHERE username =\''+user+"\'"
                cur.execute(query_password)
                result_password = cur.fetchone()[0]

                #get role from database
                query_role = 'SELECT role FROM login_info WHERE username =\''+user+"\'"
                cur.execute(query_role)
                result_role = cur.fetchone()[0]

                if result_password == password:
                    if result_role == 'patient':
                        sp_screen = SleepPositionScreenForPatient()
                        widget.addWidget(sp_screen)
                        widget.setCurrentIndex(widget.currentIndex() + 1)
                    elif result_role == 'developer':
                        next_screen = ChooseFunctionScreenForDev()
                        widget.addWidget(next_screen)
                        widget.setCurrentIndex(widget.currentIndex() + 1)
                    else:
                        next_screen = ChooseFunctionScreenForNurse()
                        widget.addWidget(next_screen)
                        widget.setCurrentIndex(widget.currentIndex() + 1)
                else:
                    self.error.setText('Invalid username or password!')
        else:
            self.error.setText('Invalid username or password!')

class ChooseFunctionScreenForNurse(QDialog):
    def __init__(self):
        super(ChooseFunctionScreenForNurse, self).__init__()
        loadUi('list_function_for_nurse.ui', self)
        self.sleep_position.clicked.connect(self.gotoSleepPositionScreen)
        self.create_account_for_patient.clicked.connect(self.createAccountPatient)
        self.log_out.clicked.connect(self.returnFunctionScreen)

    #Move to different screen
    def gotoSleepPositionScreen(self):
        sp_screen = SleepPositionScreenForNurse()
        widget.addWidget(sp_screen)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def createAccountPatient(self):
        screen_create_account = CreateAccountScreenForPatient()
        widget.addWidget(screen_create_account)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def returnFunctionScreen(self):
        loginScreen_new = LoginScreen()
        widget.addWidget(loginScreen_new)
        widget.setCurrentIndex(widget.currentIndex() + 1)

class ChooseFunctionScreenForDev(QDialog):
    def __init__(self):
        super(ChooseFunctionScreenForDev, self).__init__()
        loadUi('list_function_for_dev.ui', self)
        self.create_account_for_nurse.clicked.connect(self.createAccountNurse)
        self.log_out.clicked.connect(self.returnFunctionScreen)

    def createAccountNurse(self):
        screen_create_account = CreateAccountScreenForNurse()
        widget.addWidget(screen_create_account)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def returnFunctionScreen(self):
        loginScreen_new = LoginScreen()
        widget.addWidget(loginScreen_new)
        widget.setCurrentIndex(widget.currentIndex() + 1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # welcome = WelcomeScreen()
    login = LoginScreen()
    widget = QtWidgets.QStackedWidget()
    widget.addWidget(login)
    widget.setFixedHeight(1080)
    widget.setFixedWidth(1920)
    # widget.showFullScreen()
    widget.show()
    try:
        sys.exit(app.exec_())
    except:
        print("Exiting")