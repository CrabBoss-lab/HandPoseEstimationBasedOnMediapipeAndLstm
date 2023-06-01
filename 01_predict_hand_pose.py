# -*- codeing = utf-8 -*-
# @Time :2023/3/31 20:18
# @Author :yujunyu
# @Site :
# @File :predict.py
# @software: PyCharm

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import tensorflow as tf

mp_holistic = mp.solutions.holistic  # holistic model
# mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR->RGB
    image.flags.writeable = False  # image is no longer writeable
    results = model.process(image)  # make prediction
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB->BGR
    return image, results


def draw_styled_landmarks(image, results):
    # draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 0, 121), thickness=1, circle_radius=1)
                              )
    # draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    # draw left hand  connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(
        468 * 3)
    lh = np.array(
        [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
        21 * 3)
    rh = np.array(
        [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)

    #
    if results.left_hand_landmarks:
        lh_xy_list = []
        for index, lm in enumerate(results.left_hand_landmarks.landmark):
            lh_xy_list.append([lm.x, lm.y])
    else:
        lh_xy_list = []

    if results.right_hand_landmarks:
        rh_xy_list = []
        for index, lm in enumerate(results.right_hand_landmarks.landmark):
            rh_xy_list.append([lm.x, lm.y])
    else:
        rh_xy_list = []

    return np.concatenate([pose, face, lh, rh]), lh_xy_list, rh_xy_list


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        # print(num, prob)
        '''
        0 0.01494145
        1 0.8505417
        2 0.13451675
        '''
        # print(num)   # 0 1 2
        cv2.rectangle(output_frame, (0, 60 + num * 35), (int(prob * 100), 90 + num * 35), colors[num], -1)
        cv2.putText(output_frame, actions[num] + ' {:.2f}'.format(prob), (0, 85 + num * 35), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)

    return output_frame


def show_hand_center(image, w, h, lh_xy_list, rh_xy_list):
    if len(lh_xy_list) != 0:
        # 取hand的9，最靠近中间的坐标
        lx, ly = int(lh_xy_list[9][0] * w), int(lh_xy_list[9][1] * h)
        print(f'左:{lx, ly}')
        cv2.putText(image, 'lh', (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    if len(rh_xy_list) != 0:
        rx, ry = int(rh_xy_list[9][0] * w), int(rh_xy_list[9][1] * h)
        print(f'右:{rx, ry}')
        cv2.putText(image, 'rh', (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)


def real_time_predict():
    # 1.New detection variables
    sequence = []
    sentence = []
    threshold = 0.8

    # Open capture
    cap = cv2.VideoCapture(0)

    # fps
    fps_count = 0
    start_time = time.time()
    pTime = 0

    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            rer, frame = cap.read()

            # filp
            # frame = cv2.flip(frame, 1)

            # Make detection
            image, results = mediapipe_detection(frame, holistic)
            # print(results)

            # Draw landmarkd
            draw_styled_landmarks(image, results)

            # 2. Prediction logic
            keypoints, lh_xy_list, rh_xy_list = extract_keypoints(results)
            # 1)
            # sequence.insert(0, keypoints)
            # 2) error
            # sequence.append(keypoints)
            # sequence=sequence[:30]
            sequence.append(keypoints)
            sequence = sequence[-30:]

            # show hand center
            h, w, c = image.shape
            show_hand_center(image, w, h, lh_xy_list, rh_xy_list)

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                # print(res)
                print(actions[np.argmax(res)], np.max(res))

                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])

                    else:
                        sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                print(sentence[-1])

                # 4.Vix probabilities
                image = prob_viz(res, actions, image, colors)

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # # 计算FPS
            fps_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1:  # 每隔1秒钟更新一次FPS
                fps = fps_count / elapsed_time
                print("\033[33mFPS:{:.2f}\033[0m".format(fps))
                fps_count = 0
                start_time = time.time()

            # FPS
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(image, f'fps:{int(fps)}', (500, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 117, 255), 2, cv2.LINE_AA)

            # Show to screen
            cv2.imshow('Real-Time-Test', image)

            # Break
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu

    actions = np.array(['hello', 'thanks', 'iloveyou'])
    model = tf.keras.models.load_model("action_example.h5")
    colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

    real_time_predict()
