import pyautogui
import time
import keyboard
import random
#Check is overallbrightness of any column <100
#row1: X: 2050 y:545 
#row2: X: 2132 y:545
#row3: X: 2212 y:545
#row3  X: 2290 y:545

def click(row):
    rows = {
        1:2050,
        2:2132,
        3:2212,
        4:2290
    }
    y = 445
    x = rows[row]

    pyautogui.click(x, y)

while keyboard.is_pressed('q') == False:
    if pyautogui.pixel(2050, 445)[0] < 100:
        click(1)
    if pyautogui.pixel(2134, 445)[0] < 100:
        click(2)
    if pyautogui.pixel(2212, 445)[0] < 100:
        click(3)
    if pyautogui.pixel(2290, 445)[0] < 100:
        click(4)
    