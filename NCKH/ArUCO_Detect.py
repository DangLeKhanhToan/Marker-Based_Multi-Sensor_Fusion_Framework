import cv2
from cv2 import aruco


# ARUCODIT & VARIABLES
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

}
aruco_type = "DICT_6X6_250"
Marker_Dict = cv2.aruco.getPredefinedDictionary(ARUCODICT[aruco_type])
Marker_Param = aruco.DetectorParameters()

# Set up cam screen
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# Function for detected box of ArUCO
def ArUCO_Display(corners, ids, reject, image):
    if len(corners) > 0:
        ids = ids.flatten()
        for (markerCorners, markerID) in zip(corners, ids):
            corners = markerCorners.reshape((4, 2))

            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # define corners with id of aruco (x,y)
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # draw line around of object
            cv2.line(image, topLeft, topRight, (0, 0, 255), 2)
            cv2.line(image, topRight, bottomRight, (0, 0, 255), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 0, 255), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 0, 255), 2)
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            # put text in object
            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)
            print("[Inference] ArUco Marker ID: {}".format(markerID))
    return image

while cap.isOpened():
    ret, img = cap.read()
    # Resize image
    h, w, _ = img.shape
    width = 1000
    height = int(width * (h / w))
    img = cv2.resize(img, (width, height), cv2.INTER_CUBIC)
    # Detect ARUCO
    corners, ids, reject = cv2.aruco.detectMarkers(img, Marker_Dict, parameters=Marker_Param)
    detected_markers = ArUCO_Display(corners, ids, reject, img)
    print(ids)
    img = cv2.flip(img, 1)
    cv2.imshow("Image", detected_markers)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("x"):
        break
cv2.destroyAllWindows()
cap.release()