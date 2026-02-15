from datetime import datetime
from time import sleep
from picamzero import Camera
from exif import Image
from datetime import datetime
import cv2
import math
import numpy as np
import os
os.makedirs("Photos", exist_ok=True)


class Captures:
    def __init__(self, engine):
        self.engine = engine
        self.cam = Camera()
        self.images = []
        self.img_count = 1

    def acquire_image(self):
        filename = f"Photos/img{self.img_count}.jpg"
        self.cam.take_photo(filename)
        img_string = filename
        self.images.append(img_string)
        self.img_count += 1
        



class Calculator:
    def __init__(self, engine):
        self.engine = engine

    def preprocess(self, image):
        # CLAHE = adaptive contrast boost (very good for Earth images)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        return clahe.apply(image)

    def get_time(self, image):
        with open(image, 'rb') as image_file:
            img = Image(image_file)
            time_str = img.get('datetime_original')

        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
        return time
    
    def get_time_difference(self, image_1, image_2):
        time_1 = self.get_time(image_1)
        time_2 = self.get_time(image_2)
        time_difference = time_2 - time_1
        return time_difference.seconds
    
    def convert_to_cv(self, image):
        cv_image = self.preprocess(cv2.imread(image, 0))
        return cv_image
    
    def calculate_features(self, image_1_cv, image_2_cv, max_feature_nb):
        orb = cv2.ORB_create(nfeatures=max_feature_nb)
        mask1 = self.create_circular_mask(image_1_cv)
        mask2 = self.create_circular_mask(image_2_cv)
        keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, mask1)
        keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, mask2)
        return keypoints_1, keypoints_2, descriptors_1, descriptors_2, mask1, mask2
    
    def calculate_matches(self, descriptors_1, descriptors_2):
        brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = brute_force.match(descriptors_1, descriptors_2) # a list of landmarks that are matches between both images.
        matches_sorted = sorted(matches, key=lambda x: x.distance) # sort the list to start with the landmarks that match the most.
        return matches_sorted
    
    def find_matching_coordinates(self, keypoints_1, keypoints_2, matches):
        coordinates_1 = []
        coordinates_2 = []
        for match in matches: # each element is: "ok, the data of this coordinate_id from img1 looks like the data of that coordinate_id from img2."
            image_1_idx = match.queryIdx # saving the "the data of this coordinate_id from img1" id.
            image_2_idx = match.trainIdx # saving the "the data of this coordinate_id from img2" id.
            (x1,y1) = keypoints_1[image_1_idx].pt # making a tuple for the point associated with the coordinate_id data of image 1.
            (x2,y2) = keypoints_2[image_2_idx].pt # making a tuple for the point associated with the coordinate_id data of image 2.
            coordinates_1.append((x1,y1))
            coordinates_2.append((x2,y2))
        return coordinates_1, coordinates_2

    def calculate_mean_distance(self, coordinates_1, coordinates_2):
        all_distances = 0
        nb_of_distances = 0
        merged_coordinates = list(zip(coordinates_1, coordinates_2))
        for coordinate_pair in merged_coordinates:
            point_1 = coordinate_pair[0]
            point_2 = coordinate_pair[1]
            point_1_x, point_2_x, = point_1[0], point_2[0]
            point_1_y, point_2_y, = point_1[1], point_2[1]
            delta_x = abs(point_2_x - point_1_x) # 'abs' because we don't care about the direction, only the value of the displacement.
            delta_y = abs(point_2_y - point_1_y) # same idea.
            distance = math.hypot(delta_x, delta_y)
            all_distances += distance
            nb_of_distances += 1
        average_distance = all_distances/nb_of_distances
        return average_distance
    
    def calculate_speed_in_kmps(self, feature_distance, GSD, time_difference):
        distance = feature_distance * GSD / 100000 # [pixels] * [centimers/pixels] * 100000 = [pixels]*[kilometers/pixels] = [kilometers]
        speed = distance/time_difference # [kilometers] / [seconds] = [kilometers/seconds]
        return speed

    def create_circular_mask(self, image):
        h, w = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        radius = min(center[0], center[1]) - 5  # small margin
        cv2.circle(mask, center, radius, 255, -1)
        return mask
    

    




class Engine:
    def __init__(self, start_time):
        self.start_time = start_time # Define a starting timestamp.
        self.now_time = datetime.now()
        self.elapsed_time = (self.now_time - self.start_time).total_seconds()  # Save the elapsed time (out of 10 minutes) in seconds.
        self.estimate_speed_kmps = 0
        self.number_of_results = 0
        self.captures = Captures(self)
        self.calculator = Calculator(self)
        self.ISS_MOVING_TIME_SECONDS = 1
        self.GSD = 12648  # GSD = 12648 for the High Quality Camera on the ISS.


    def update_timer(self):
        self.now_time = datetime.now()
        self.elapsed_time = (self.now_time - self.start_time).total_seconds()

    def save_result(self, estimate_speed_kmps):
        estimate_kmps_formatted = "{:.4f}".format(estimate_speed_kmps)
        output_string = estimate_kmps_formatted
        file_path = "result.txt"
        with open(file_path, 'w') as file:
            file.write(output_string) # write the final result in "result.txt".
            print("Data written to", file_path)
            file.close()

    def apply_mask(self, image, mask):
        return cv2.bitwise_and(image, image, mask=mask)

    def display_matches(self, image_1_cv, image_2_cv, keypoints_1, keypoints_2, matches, mask1, mask2):
        masked_img1 = self.apply_mask(image_1_cv, mask1)
        masked_img2 = self.apply_mask(image_2_cv, mask2)
        matches_image = cv2.drawMatches(masked_img1, keypoints_1, masked_img2, keypoints_2, matches[:100], None)
        resize = cv2.resize(matches_image, (900, 350), interpolation=cv2.INTER_AREA)
        cv2.imshow('Masked Matches', resize)
        cv2.waitKey(0)
        cv2.destroyWindow('Masked Matches')
    
    def update(self):
        calc = self.calculator
        self.captures.acquire_image()
        sleep(self.ISS_MOVING_TIME_SECONDS)
        self.captures.acquire_image()
        img1 = self.captures.images[0]
        img2 = self.captures.images[1]
        time_difference = calc.get_time_difference(img1, img2)
        max_nb_features = 100
        keypoints1, keypoints2, descriptors1, descriptors2, mask1, mask2 = calc.calculate_features(calc.convert_to_cv(img1), calc.convert_to_cv(img2), max_nb_features)
        matches = calc.calculate_matches(descriptors1, descriptors2)
        coordinates_1, coordinates_2 = calc.find_matching_coordinates(keypoints1, keypoints2, matches)
        average_feature_distance = calc.calculate_mean_distance(coordinates_1, coordinates_2) # average distance between 2 associated feature points.
        speed = calc.calculate_speed_in_kmps(average_feature_distance, self.GSD, time_difference)
        #self.display_matches(calc.convert_to_cv(img1), calc.convert_to_cv(img2), keypoints1, keypoints2, matches, mask1, mask2)
        self.estimate_speed_kmps += speed
        self.number_of_results += 1
        self.captures.images = []
        self.clear_folder()

    def clear_folder(self):
        for file in os.listdir("Photos"):
            file_path = os.path.join("Photos", file)
            if os.path.isfile(file_path):
                os.remove(file_path)  

    def run(self):
        while self.elapsed_time < 10*60 - 10: # (a little less than 10 minutes (which is 10 times 60 seconds) just to leave some time for file saving).
            self.update()
            self.update_timer()

        self.save_result(self.estimate_speed_kmps/self.number_of_results)
        




start_time = datetime.now()

engine = Engine(start_time)
engine.run()


