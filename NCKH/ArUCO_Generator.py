import numpy as np
import cv2.aruco
import os

ARUCODICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
} # Create Dict to define types of ARUCO MARKER


# Create a folder to storage the Tag/Marker image
folder_name = "Tag_Storage"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

for id in range(30):
    aruco_type = "DICT_6X6_250" # Choose 6x6_250 ARUCO MARKER
    arucoDict = cv2.aruco.getPredefinedDictionary(ARUCODICT[aruco_type])
    print("Aruco Type '{}' with ID '{}' ".format(aruco_type, id))
    tag_size = 250 # size of tag image
    tag = np.zeros((tag_size, tag_size, 1), dtype="uint8")
    cv2.aruco.Dictionary.generateImageMarker(arucoDict, id, tag_size, tag, 1)

    tag_name = os.path.join(folder_name, "{}_{}.png".format(aruco_type, id)) # save image in folder
    cv2.imwrite(tag_name, tag)
    cv2.imshow("ArUco Tag", tag)

cv2.waitKey(0)
cv2.destroyAllWindows()