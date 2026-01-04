from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import cv2
import numpy as np
import mss
import sys
import mss.tools
from ultralytics import YOLO
import time


class bot_controller:
    
    def __init__(self, matching_confidence = 0.8, offy = 125, lowerwidth = 630):
        self.confidence = matching_confidence
        self.scc = mss.mss()
        self.offset_y = offy
        self.offset_x = lowerwidth/2
        self.offset_width = lowerwidth
        self.driver = webdriver.Chrome()
        self.driver.get("https://flappybird.io/")
        self.canvas = self.driver.find_element(By.ID, "application-canvas")
        self.model = YOLO('/Users/naijei/ML-Bots/Flappy-Bot/FlappyBirdYoloModel/my_model.pt') 
        self.started = False
        
        self.debugCount = 0
    
    @staticmethod
    def loadImage(file_path):
        #Loads images in numpy black and white
        try:
            img_arry = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
            # cv2.imshow("Image", img_arry)
            # cv2.waitKey(0)
            return img_arry
        except:
            print("Invalid asset file path, exiting program...")
            sys.exit()
    
    def getScreenShot(self):
        #Captures the screen and turns into a black and white numpy array
        window_data = self.driver.get_window_rect()
        window_position_x = window_data.get("x")
        window_position_y = window_data.get("y")
        
        rect = self.canvas.rect
        canvas_w = rect["width"]
        canvas_h = rect["height"]
    
        monitor = {
            "top": window_position_y + self.offset_y, 
            "left": window_position_x + self.offset_x, 
            "width": canvas_w - self.offset_width, 
            "height": canvas_h
        } #Sets region
        screenshot_array = np.array(self.scc.grab(monitor)) #Gets screenshot
        return screenshot_array[:, :, :3] #Turns to black and white
        
    
    def runYOLO(self, img, imgsz=320, conf=0.5):
        results = self.model.predict(source=img, save=False, imgsz=imgsz, conf=conf)
        return results
    
    def get_state(self, img, imgsz=320, conf=0.5):
        """
        Use YOLO to extract a simple RL state from the current frame.

        Returns a dict:
            {
                "bird_x": ...,
                "bird_y": ...,
                "pipe_x": ...,
                "pipe_y": gap_center_y,
                "dx": pipe_x - bird_x,
                "dy": gap_center_y - bird_y,
                "bird_vy": estimated vertical velocity
            }
        or None if detections are missing.
        """

        # Run YOLO
        results_list = self.runYOLO(img, imgsz=imgsz, conf=conf)

        if not results_list:
            return None

        result = results_list[0]
        boxes = result.boxes
        names = result.names  # class id

        bird = None
        bottoms = []  # (cx, cy)
        tops = []     # (cx, cy)

        for box in boxes:
            cls_id = int(box.cls[0])
            label = names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            lname = label.lower()

            if lname == "bird":
                bird = (cx, cy)
            elif "bottom_pipe" in lname:
                bottoms.append((cx, cy))
            elif "top_pipe" in lname:
                tops.append((cx, cy))

        if bird is None or len(bottoms) == 0:
            return None

        bird_x, bird_y = bird

        columns = [] 

        for bx, by in bottoms:
            if tops:
                tx, ty = min(tops, key=lambda t: abs(t[0] - bx))
                col_x = (bx + tx) / 2.0
                gap_center_y = (by + ty) / 2.0
            else:
                col_x = bx
                gap_center_y = by 

            columns.append((col_x, gap_center_y))

        if len(columns) == 0:
            return None

        cols_ahead = [c for c in columns if c[0] > bird_x]
        if len(cols_ahead) > 0:
            pipe_x, gap_center_y = min(cols_ahead, key=lambda c: c[0])
        else:
            # If none ahead, take the closest one (e.g., just after passing)
            pipe_x, gap_center_y = min(columns, key=lambda c: c[0])

        if not hasattr(self, "_last_bird_y"):
            self._last_bird_y = bird_y
            bird_vy = 0.0
        else:
            bird_vy = bird_y - self._last_bird_y
            self._last_bird_y = bird_y

        dx = pipe_x - bird_x
        dy = gap_center_y - bird_y

        state = {
            "bird_x": bird_x,
            "bird_y": bird_y,
            "pipe_x": pipe_x,
            "pipe_y": gap_center_y,
            "dx": dx,
            "dy": dy,
            "bird_vy": bird_vy,
        }

        return state
    
    def is_game_over(self, img):
        """
        Takes region over the score, if the score is visable then the game is active
        Returns true if game is active
        false otherwise
        """

        h, w, _ = img.shape

        x1 = int(w * 0.42)
        x2 = int(w * 0.58)
        y1 = int(h * 0.05)
        y2 = int(h * 0.12)

        roi = img[y1:y2, x1:x2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        white_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.size
        white_ratio = white_pixels / total_pixels

        return white_ratio < 0.002 
    
    def startGame(self):
        print("Waiting 1")
        #CLICKS START BUTTONT
        time.sleep(2.15)
        print("DONE 1")
        offset_x = -50
        offset_y = 250
        
        actions = ActionChains(self.driver)
        actions.move_to_element_with_offset(self.canvas, offset_x, offset_y)
        actions.click()
        actions.perform()
        if(self.started):
            #CLICKS START BUTTON AFTER RESTART BUTTON, Ran after the first run
            print("Waiting 2")
            time.sleep(0.25)
            print("DONE 2")
            actions.click()
            actions.perform()
        self.debugCount += 1
        print(self.debugCount,"----------------GAME RESTARTED -----------------------------")
        self.started = True
        time.sleep(0.45)
        #THIS OFFICALLY STARTS
        actions.click()
        actions.perform()

    
bot = bot_controller(matching_confidence = 0.8, offy = 125, lowerwidth = 630)
while True:
    img = bot.getScreenShot() #UNCOMMENT TO DOUBLE CHECK ALINEMENT: I have found that matching_confidence = 0.8, offy = 125, lowerwidth = 630 works best
    #cv2.imshow("preview", img)
    #cv2.waitKey(0) 
    print("GAME OVER?: ", bot.is_game_over(img))
    if bot.is_game_over(img):
        bot.startGame()
    state = bot.get_state(img, imgsz=192, conf=0.5)
    if state is None:
        continue
    #print(state)



#To control bird use --> canvas.click()


# First detect pipes and where the bird is
# -- Detect bird position
# -- Detect pipe positiong
# We can use mss and openCV to do this, with template matching


# Train model to use the information of bird and pipe potition
# plan: TBD