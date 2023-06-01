# 👇体感互动——Hand Pose Estimation Based on Mediapipe And LSTM

目录：

- [👇体感互动——Hand Pose Estimation Based on Mediapipe And LSTM](#体感互动hand-pose-estimation-based-on-mediapipe-and-lstm)
    - [🐱Introduce](#introduce)
    - [🐖QuickStart](#quickstart)
        - [Dependencies](#dependencies)
        - [Inference1](#inference1)
        - [Inference2](#inference2)
        - [体感互动](#体感互动)
    - [😔Training](#training)
    - [🐒Reference](#reference)
    - [🐕Thanks](#thanks)

## 🐱Introduce

**体感互动——Hand Pose Estimation Based on Mediapipe And LSTM**

基于mediapipe提取人体姿势、人脸、手部关键点和LSTM算法实现手势识别，并进一步实现体感互动的功能，如进行隔空移动、抓取、放大、缩小等手势操作，可为商业显示提供智能交互应用，如3D模型的展示。

数据：一共采集3类手势，每一个手势30个视频，每一个视频30帧，每一帧提取的landmarks序列长度为1662（33个4维人体姿势landmarks，468个3维人脸landmarks，21个3维左手landmarks，21个3维右手landmarks，33✖4＋468✖3+21✖3+21✖3=1662）。

网络：三层lstm([lstm_net.py](https://edu.gitee.com/jhcyun/repos/jhcyun/HandPoseEstimationBasedOnMediapipeAndLstm/blob/master/lstm_net.py))

* demo1:01_predict_hand__pose_.py

    ![demo1.gif](assets/demo1.gif)

* demo2：02_predict.py

    ![demo2.gif](assets/demo2.gif)

* demo3：03_predict之体感互动.py

    ![demo3.gif](assets/demo3.webp)

## 🐖QuickStart

### Dependencies

``pip install -r requirements.txt
``

### Inference1

``python 01_predict_hand_pose.py
``

### Inference2

``python 02_predict.py
``

### 体感互动

1、 打开https://720.vrqjcs.com/t/9332870054821ffc

2、
``python 03_predict之体感互动.py
``

## 😔Training

详见[train_my_hand_pose/hand pose detection.ipynb](https://edu.gitee.com/jhcyun/repos/jhcyun/HandPoseEstimationBasedOnMediapipeAndLstm/blob/master/train_my_hand_pose%2Fhand%20pose%20detection.ipynb)

具体开发流程见[Hand-Pose-Estimation.pdf](https://edu.gitee.com/jhcyun/repos/jhcyun/HandPoseEstimationBasedOnMediapipeAndLstm/blob/master/Hand-Pose-Estimation.pdf)
思维导图

## 🐒Reference

* [Mediapipe](https://google.github.io/mediapipe/)

* [@NicholasRenotte](https://www.youtube.com/watch?v=doDUihpj6ro)

* [VRMap](https://720.vrqjcs.com/t/9332870054821ffc)

## 🐕Thanks

* @Studio:JHC Software Dev Studio

* @Mentor:HuangRiChen

* @Author:YuJunYu
