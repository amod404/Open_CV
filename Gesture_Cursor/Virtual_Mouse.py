import cv2
import mediapipe as mp
import numpy as np
import util
import pyautogui
from pynput.mouse import Button, Controller
mouse = Controller()

screen_width, screen_height = pyautogui.size()

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,   
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]

    return None

def is_left_click(landmark_list, thumb_index_dist):
    return (util.get_angle(landmark_list[5],landmark_list[6],landmark_list[8]) < 50 and
             util.get_angle(landmark_list[9],landmark_list[10],landmark_list[12]) > 90 and
             thumb_index_dist > 50)

def is_right_click(landmark_list, thumb_index_dist):
    return (util.get_angle(landmark_list[9],landmark_list[10],landmark_list[12]) < 50 and
            util.get_angle(landmark_list[5],landmark_list[6],landmark_list[8]) > 90 and
             thumb_index_dist > 50)

def is_double_click(landmark_list, thumb_index_dist):
    return (util.get_angle(landmark_list[5],landmark_list[6],landmark_list[8]) < 50 and
             util.get_angle(landmark_list[9],landmark_list[10],landmark_list[12]) < 50 and
             thumb_index_dist > 50)

def is_screenshot(landmark_list, thumb_index_dist):
    return (util.get_angle(landmark_list[5],landmark_list[6],landmark_list[8]) < 50 and
             util.get_angle(landmark_list[9],landmark_list[10],landmark_list[12]) < 50 and
             thumb_index_dist < 50)

def detect_gestures(frame,landmark_list,processed):
    if len(landmark_list)>=21:

        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = util.get_distance([landmark_list[4],landmark_list[5]])

        if(thumb_index_dist<50 and util.get_angle(landmark_list[5],landmark_list[6],landmark_list[8])>90):
            move_mouse(index_finger_tip)

        elif is_left_click(landmark_list, thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        elif is_right_click(landmark_list, thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        elif is_double_click(landmark_list, thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
        elif is_screenshot(landmark_list, thumb_index_dist):
            im1 = pyautogui.screenshot()
            label = np.random.randint(1,10000)
            im1.save(f'my_screenshot_{label}.png')
            cv2.putText(frame, "Screenshot Taken", (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)

def move_mouse(index_finder_tip):
    if index_finder_tip is not None:
        x = int(index_finder_tip.x * screen_width)
        y = int(index_finder_tip.y * screen_height)
        pyautogui.moveTo(x,y)

def main():
    cap = cv2.VideoCapture(0)
    draw = mp.solutions.drawing_utils
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)
            frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            processed = hands.process(frameRGB)

            landmark_list = []

            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame,hand_landmarks,mpHands.HAND_CONNECTIONS)

                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x,lm.y))
            
            detect_gestures(frame, landmark_list,processed)


            cv2.imshow('Frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


