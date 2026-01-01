from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import cv2
import numpy as np
import mss
import sys
import mss.tools

class bot_controller:
    
    def __init__(self, matching_confidence = 0.8):
        self.confidence = matching_confidence
        self.scc = mss.mss()
        
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
        window_data = self.driver.get_window_rect()
        window_position_x = window_data.get("x")
        window_position_y = window_data.get("y")
        window_height =  window_data.get("height")
        window_width =  window_data.get("width")
        
        monitor = {"top": window_position_y, "left": window_position_x, "width": window_width, "height": window_height}
        
        return self.scc.grab(monitor)
        
    
    def templateMatch(self, template_image, current_screen_shot = None):
        pass
    
bot = bot_controller()
img = bot.getScreenShot()
mss.tools.to_png(img.rgb, img.size, output="test")



#To control bird use --> canvas.click()


# First detect pipes and where the bird is
# -- Detect bird position
# -- Detect pipe positiong
# We can use mss and openCV to do this, with template matching


# Train model to use the information of bird and pipe potition
# plan: TBD