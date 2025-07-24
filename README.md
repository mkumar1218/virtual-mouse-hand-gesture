import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
screen_w, screen_h = pyautogui.size()

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Smoothing variables
prev_x, prev_y = 0, 0
smoothening = 7

# Click debounce
click_time = 0
click_delay = 0.5  # seconds

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        lm_list = []
        h, w, _ = img.shape

        for id, lm in enumerate(hand_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append((id, cx, cy))

        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get coordinates
        x_index, y_index = lm_list[8][1], lm_list[8][2]
        x_thumb, y_thumb = lm_list[4][1], lm_list[4][2]

        # Map to screen
        screen_x = screen_w * x_index / 640
        screen_y = screen_h * y_index / 480

        # Smoothing
        curr_x = prev_x + (screen_x - prev_x) / smoothening
        curr_y = prev_y + (screen_y - prev_y) / smoothening
        pyautogui.moveTo(curr_x, curr_y)
        prev_x, prev_y = curr_x, curr_y

        # Click detection
        distance = math.hypot(x_thumb - x_index, y_thumb - y_index)
        if distance < 40:
            if time.time() - click_time > click_delay:
                pyautogui.click()
                click_time = time.time()
                cv2.circle(img, (x_index, y_index), 15, (0, 255, 0), cv2.FILLED)

    cv2.imshow("Virtual Mouse Improved", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# virtual-mouse-hand-gesture
