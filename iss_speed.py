from exif import Image
from datetime import datetime
import cv2
import math

img1 = 'ExamplePhotos/atlas_photo_012.jpg'
img2 = 'ExamplePhotos/atlas_photo_013.jpg'

def get_time(image):
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        time_str = img.get('datetime_original')
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    return time
    
def get_time_difference(image_1, image_2):
    time_1 = get_time(image_1)
    time_2 = get_time(image_2)
    time_difference = time_2 - time_1
    return time_difference.seconds

def convert_to_cv(image):
    cv_image = cv2.imread(image, 0)
    return cv_image

def calculate_features(image_1_cv, image_2_cv, max_feature_nb):
    orb_algorithm_instance = cv2.ORB_create(nfeatures = max_feature_nb)
    keypoints_1, descriptors_1 = orb_algorithm_instance.detectAndCompute(image_1_cv, None)
    keypoints_2, descriptors_2 = orb_algorithm_instance.detectAndCompute(image_2_cv, None)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2

def calculate_matches(descriptors_1, descriptors_2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2) # a list of landmarks that are matches between both images.
    matches = sorted(matches, key=lambda x: x.distance) # sort the list to start with the landmarks that match the most.
    return matches

def find_matching_coordinates(keypoints_1, keypoints_2, matches):
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


def calculate_mean_distance(coordinates_1, coordinates_2):
    all_distances = 0 # additive identity element
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

def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    distance = feature_distance * GSD / 100000 # [pixels] * [centimers/pixels] * 100000 = [pixels]*[kilometers/pixels] = [kilometers]
    speed = distance/time_difference # [kilometers] / [seconds] = [kilometers/seconds]
    return speed


def display_matches(image_1_cv, image_2_cv, keypoints_1, keypoints_2, matches):
    matches_image = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches[:100], None)
    resize = cv2.resize(matches_image, (900,350), interpolation = cv2.INTER_AREA)
    cv2.imshow('Matches', resize) # 'Matches' is the window name btw.
    cv2.waitKey(0)
    cv2.destroyWindow('Matches')


time_difference = get_time_difference(img1, img2)
max_nb_features = 10000
features = calculate_features(convert_to_cv(img1), convert_to_cv(img2), max_nb_features)
matches = calculate_matches(features[2], features[3])
coordinates_1, coordinates_2 = find_matching_coordinates(features[0], features[1], matches)
average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2) # average distance between 2 associated feature points.
speed = calculate_speed_in_kmps(average_feature_distance, 12648, time_difference) # GSD = 12648 for the High Quality Camera on the ISS.

print(speed) # with this sample (just two images) we already get something pretty close to the actual value, (7.66 ± 0.05)kmps  normally quoted for the ISS and 7.75kmps here.
#display_matches(convert_to_cv(img1), convert_to_cv(img2), features[0], features[1], matches)

# next maybe calculate some z-score or get more images for more precision.
# note, it looks like with a bigger nb of features, our answer is 7.86kmps which is even further from the actual value.
# maybe those two images yield a speed that we got closer to with more nb of features, but that speed itself is not the actuall (7.66 ± 0.05)kmps expected.
# plus, increasing max nb of features increases run-time... [not good for efficiency].


# check the website for possible ideas of improvements in the program.
# use the astropi replay tool to test with more images.

