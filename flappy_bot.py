from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

driver = webdriver.Chrome()
driver.get("https://flappybird.io/")

canvas = driver.find_element(By.ID, "application-canvas") #Starts up the game

#To control bird use --> canvas.click()



# First detect pipes and where the bird is
# -- Detect bird position
# -- Detect pipe positiong
# We can use mss and openCV to do this, with template matching


# Train model to use the information of bird and pipe potition
# plan: TBD