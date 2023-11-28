import numpy as np
import math
import cv2

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


def aruco_display(corners, ids, reject, image):
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
            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            # put text in object
            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)
            print("[Inference] ArUco Marker ID: {}".format(markerID))
    return image


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    # parameters = cv2.aruco.DetectorParameters()
    cameraMatrix = matrix_coefficients
    distCoeff = distortion_coefficients
    corners_ps, ids_ps, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, cameraMatrix, distCoeff)
    # If markers are detected
    rvecs = []
    tvecs = []
    markerPoints = []
    if len(corners_ps) > 0:
        for i in range(0, len(ids_ps)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners_ps[i], 0.02,
                                                                           matrix_coefficients,
                                                                           distortion_coefficients)

            rvecs.append(rvec)
            tvecs.append(tvec)
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners_ps, borderColor= (0, 0, 255))
            # Draw axis
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''
    return frame, rvecs, tvecs


aruco_type = "DICT_6X6_250"

arucoDict = cv2.aruco.getPredefinedDictionary(ARUCODICT[aruco_type])

# arucoParams = cv2.aruco.DetectorParameters()


#-----------------------------------------------------------
intrinsic_camera = np.array(((907.83627812, 0, 856.52305979), (0, 775.13996941, 341.82951344), (0, 0, 1)))
distortion = np.array((0.35772536, -0.37022379, -0.00657575, 0.11881465, 0.22432219))
#-----------------------------------------------------------
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    frame, rvecs, tvecs = pose_estimation(img, ARUCODICT[aruco_type], intrinsic_camera, distortion)

    if len(rvecs) > 0:
        rvec = np.array(rvecs[0][0])  # Access the rotation vector of the first marker
        tvec = np.array(tvecs[0][0])  # Access the translation vector of the first marker
        rvec = rvec.reshape((3, 1))  # Reshape rotation vector to (3, 1)
        # print(f'First, rvec like this: {rvec}')
        matrix_rvec, _ = cv2.Rodrigues(rvec)
        # print(f'Rvec after Rodrigues transform: {matrix_rvec}')
        tvec = tvec.reshape((3, 1))  # Reshape translation vector to (3, 1)
        # print("tvec: ", tvec)
        mid_matrix = np.hstack((matrix_rvec, tvec))
        # print(f'Finally the mid matrix is: {mid_matrix}')

        # Here we calculate angle [AnglePhiOne ] and distance [Distance_d] between camera adn the ArUCO marker:
        r31 = matrix_rvec[2][0]
        r32 = matrix_rvec[2][1]
        r33 = matrix_rvec[2][2]
        t1 = tvec[0]
        t3 = tvec[2]
        Llat = 1 # Lateral distance btw camera position and center point of tje rear axle of the vehicle
        Llon = 1 # Longitudinal distance btw camera position and center point of tje rear axle of the vehicle
        AnglePhiOne = math.atan2(-r31, math.sqrt(r32**2 + r33**2))
        Distance_d = math.sqrt((t1 +Llat)**2 + (t3 + Llon)**2)
    else:
        print("No markers detected.")
    cv2.imshow("EstimatePose", frame)



    key = cv2.waitKey(1) & 0xFF
    if key == ord("x"):
        break
cap.release()
cv2.destroyAllWindows()

