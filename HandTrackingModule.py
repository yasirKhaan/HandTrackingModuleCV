import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False, max_num_hands = 2, model_complexity = 1, min_detection_confidence = 0.5,  min_tracking_confidence = 0.5):
        self.mode= mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_num_hands, self.model_complexity,
                                        self.min_detection_confidence, self.min_tracking_confidence)  # If want to check default parameters ctrl+click on Hands()

        self.mpDraw = mp.solutions.drawing_utils  # Draws line between dots of hand

    def find_hands(self, img, draw= True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks: # results.multi_hand_landmarks  detects hand
            for each_hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, each_hand_landmarks, self.mpHands.HAND_CONNECTIONS) # Draws dots and lines on hands
        return img

    def find_hand_position(self, img, hand_number=0, draw=True):
        land_mark_list = []
        if self.results.multi_hand_landmarks: # results.multi_hand_landmarks  detects hand
            myHand = self.results.multi_hand_landmarks[hand_number]

            for id, co_ords_of_hands in enumerate(myHand.landmark):
                # id, co_ords_of_hands gives value in points which are ratio of image
                # print(id, co_ords_of_hands)
                height, width, channels = img.shape
                cx, cy = int(co_ords_of_hands.x*width), int(co_ords_of_hands.y*height)
                land_mark_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                # if id == 8 or id == 12:
                #     cv2.circle(img, (cx,cy), 25, (255, 0, 255), cv2.FILLED)
                #     print(id, cx, cy)
        return land_mark_list



def main():
    previous_time = 0  # Time to initialize FPS
    current_time = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.find_hands(img)
        list_of_hand_vals = detector.find_hand_position(img)
        if len(list_of_hand_vals) != 0:
            print(list_of_hand_vals[8])
        current_time = time.time()
        fps = 1/(current_time-previous_time)
        previous_time = current_time

        cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_TRIPLEX, 3, (255, 0, 255), 3)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow('Image -->', img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
