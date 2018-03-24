# coding:utf-8
__author__ = "Mistery"

import os
import sys
import math
import logging
from random import uniform
import time
from time import sleep
import cv2
import numpy as np
from sklearn import svm
from sklearn.externals import joblib

class Hog_descriptor():
    def __init__(self, img, cell_size=16, bin_size=8):
        self.img = img
        self.img = np.sqrt(img / np.max(img))
        self.img = img * 255
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 / self.bin_size
        assert type(self.bin_size) == int, "bin_size should be integer,"
        assert type(self.cell_size) == int, "cell_size should be integer,"
        assert type(self.angle_unit) == int, "bin_size should be divisible by 360"

    def extract(self):
        height, width = self.img.shape
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)
        cell_gradient_vector = np.zeros((height / self.cell_size, width / self.cell_size, self.bin_size))
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)
        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)
        return hog_vector, hog_image

    def global_gradient(self):
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        return idx, (idx + 1) % self.bin_size, mod

    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image


exitPos = (70, 90)
confirmPos = (1160, 610)
chap25Pos = (1770, 910)
explorePos = (1430, 820)


device_x, device_y = 1920, 1080

logging.basicConfig(format='%(asctime)s %(message)s',
					datefmt='%m/%d/%Y %I:%M:%S %p:',
					level=logging.DEBUG)

def tap_screen(pos):
	base_x, base_y = 1920, 1080
	x = int(float(pos[0]) / base_x * device_x + uniform(0, 10) - 5)
	y = int(float(pos[1]) / base_y * device_y + uniform(0, 10) - 5)
	os.system("adb shell input tap {} {}".format(x, y))

def mySleep(s):
	rand = uniform(-0.5, 0.5)
	sleep(s+rand)

def getScreenShot():
	os.system("adb shell screencap -p /sdcard/tmp.png")
	os.system("adb pull /sdcard/tmp.png {}/tmp.png".format(os.getcwd()))
	os.system("adb shell rm /sdcard/tmp.png")

def getMonsterPos():
	getScreenShot()
	img = cv2.imread("tmp.png", 0)
	template = cv2.imread("temp.png", 0)
	w, h = template.shape[::-1]
	img2 = img.copy()
	result = cv2.matchTemplate(img2, template, cv2.TM_CCOEFF)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
	top_left = max_loc
	bottom_right = top_left[0] + w, top_left[1] + h
	
	tar = img[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
	
	clf = joblib.load('svm_model.pkl')
	hog = Hog_descriptor(tar, cell_size=16, bin_size=8)
	vector, image = hog.extract()
	vector = np.array([np.array(vector, dtype=np.float32).flatten()])
	res = clf.predict(vector)
	if res == True:
		pos = (top_left[0] + bottom_right[0])/2, (top_left[1] + bottom_right[1])/2
		return pos
	else:
		return None
	# poses = []
	# threshold = 0.9999995 * max_val
	# loc = np.where( result >= threshold)
	# for pt in zip(*loc[::-1]):
	# 	pos = (pt[0] + w, pt[1] + h)
	# 	poses.append(pos) 
	# if len(poses) > 0:
	# 	return poses[0] 
	# else:
	# 	return None

def clickMonster(pos):
	logging.debug("Click the monster.")
	monsterPos = pos
	tap_screen(monsterPos)
	# if monsterPos is not None:
	x_, y_ = monsterPos
	# t2 = time.time()
	# print t2-t1
	possiblePoses = [(x+x_, y+y_) for x in xrange(-50, 51, 50) 
				for y in xrange(-50, 51, 50)]
	for pos in possiblePoses:
		tap_screen(monsterPos)
		# mySleep(0.1 + uniform(-0.05, 0.05))
		# print 'working'
	# else:
	# 	print "No monster found."

def generateRandPos(pos):
	return round(uniform(-100, 100)) + pos[0], round(uniform(-100, 100)) + pos[1]

def work():
	logging.debug("Click Chapter25.")
	tap_screen(chap25Pos)
	mySleep(2)

	logging.debug("Click Explore.")
	tap_screen(explorePos)
	mySleep(3)

	logging.debug("Try to find monster's Position.")
	t1 = time.time()
	count = 0
	t = time.time() - t1
	while count < 4 and t < 150:
		
		pos = getMonsterPos()
		if pos != None:
			count += 1
			clickMonster(pos)

			mySleep(15)
			logging.debug("Tap the screen.")
			randPos1 = generateRandPos((960, 800))
			tap_screen(randPos1)
			mySleep(2)
			# logging.debug("Tap the screen.")
			# randPos2 = generateRandPos((960, 800))
			# tap_screen(randPos2)
			# mySleep(2)
			# logging.debug("Tap the screen.")
			# randPos3 = generateRandPos((960, 800))
			# tap_screen(randPos3)
			# mySleep(2)
		else:
			os.system("adb shell input swipe 1500 500 1000 500")
			mySleep(2)
		t = time.time() - t1
	os.system("adb shell input swipe 750 500 1500 500")
	pos = getMonsterPos()
	clickMonster(pos)

	mySleep(15)
	logging.debug("Tap the screen.")
	randPos1 = generateRandPos((960, 800))
	tap_screen(randPos1)
	mySleep(2)

	tap_screen(exitPos)
	logging.debug("Tap the exit button.")
	mySleep(2)

	tap_screen(confirmPos)
	logging.debug("Tap the confirm button.")
	mySleep(2)



		





if __name__ == '__main__':
	work()