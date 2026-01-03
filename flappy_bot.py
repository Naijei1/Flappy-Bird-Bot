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
        return cv2.cvtColor(screenshot_array, cv2.COLOR_BGRA2GRAY) #Turns to black and white
        
    
    def templateMatch(self, template_image, current_screen_shot = None):
        pass
    
    
#bot = bot_controller(matching_confidence = 0.8, offy = 125, lowerwidth = 630)
#img = bot.getScreenShot() #UNCOMMENT TO DOUBLE CHECK ALINEMENT: I have found that matching_confidence = 0.8, offy = 125, lowerwidth = 630 works best
#cv2.imshow("preview", img)
#cv2.waitKey(0) 

model = YOLO('/Users/naijei/ML-Bots/Flappy-Bot/FlappyBirdYoloModel/my_model.pt') 

results = model.predict(source="/Users/naijei/ML-Bots/Flappy-Bot/data/yolo_frames/frame_00130.png", save=True, conf=0.5)

for result in results:
    boxes = result.boxes 




#To control bird use --> canvas.click()


# First detect pipes and where the bird is
# -- Detect bird position
# -- Detect pipe positiong
# We can use mss and openCV to do this, with template matching


# Train model to use the information of bird and pipe potition
# plan: TBD