from lib import find_marker
import numpy as np
import cv2
import marker_dectection
import sys
import setting

calibrate = False

if len(sys.argv) > 1:
    if sys.argv[1] == 'calibrate':
        calibrate = True

gelsight_version = 'Bnz'
# gelsight_version = 'HSR'

cap = cv2.VideoCapture(2)

# Resize scale for faster image processing
setting.init()
RESCALE = setting.RESCALE
x0_marker = setting.x0_
y0_marker = setting.y0_
dx_marker = setting.dx_
dy_marker = setting.dy_


def match_markers(initial_positions, current_positions, previous_positions, max_distance=100):
    matched_positions = []
    
    for idx, initial in enumerate(initial_positions):
        closest = None
        min_dist = float('inf')
        # Predict new position based on previous position
        predicted = previous_positions[idx] if idx < len(previous_positions) else initial
        
        for current in current_positions:
            dist = np.linalg.norm(np.array(predicted) - np.array(current))
            if dist < min_dist and dist < max_distance:
                min_dist = dist
                closest = current
        
        if closest:
            matched_positions.append(closest)
            previous_positions[idx] = closest
        else:
            matched_positions.append(predicted)

    return matched_positions


def get_marker_positions():
    """
    Get the positions of markers in a grid.
    
    :param initial_marker: List containing the x, y position of the top-left marker [x0, y0].
    :param dx: Distance between markers in the x-direction.
    :param dy: Distance between markers in the y-direction.
    :param rows: Number of rows of markers.
    :param cols: Number of columns of markers.
    :return: List of lists containing positions of all markers.
    """
    markers = []
    x0, y0 = x0_marker, y0_marker

    for row in range(setting.N_):
        for col in range(setting.M_):
            # Calculate the position of each marker
            x = x0 + col * dx_marker
            y = y0 + row * dy_marker
            markers.append([int(x), int(y)])
    
    return markers

loc_of_markers = get_marker_positions()


def draw_circle(img, x, y):

    center_coordinates = (int(x), int(y)) # (x, y) coordinates of the center
    radius = 1
    color = (0, 0, 255)
    thickness = -1
    cv2.circle(img, center_coordinates, radius, color, thickness)

# Create Mathing Class
m = find_marker.Matching(
    N_=setting.N_, 
    M_=setting.M_, 
    fps_=setting.fps_, 
    x0_=setting.x0_, 
    y0_=setting.y0_, 
    dx_=setting.dx_, 
    dy_=setting.dy_)
"""
N_, M_: the row and column of the marker array
x0_, y0_: the coordinate of upper-left marker
dx_, dy_: the horizontal and vertical interval between adjacent markers
"""

previous_positions = loc_of_markers.copy()

while(True):

    ret, frame = cap.read()
    if not(ret):
        break

    if gelsight_version == 'HSR':
        frame = marker_dectection.init_HSR(frame)
    else:
        frame = marker_dectection.init(frame)
    
        for i in range(63):
            draw_circle(frame, loc_of_markers[i][0], loc_of_markers[i][1])
        
    mask = marker_dectection.find_marker(frame)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    mc = marker_dectection.marker_center(mask, frame)

    current_centroids_list = []
    buf_centroids_list = []

    if calibrate == False:

        m.init(mc)
        m.run()

    mask_img = mask.astype(frame[0].dtype)
    mask_img = cv2.merge((mask_img, mask_img, mask_img))

    frame1 =np.max(frame,axis=2)
    frame1 = cv2.medianBlur(frame1, 3)
    frame1[frame1<70] = 255
    frame1[frame1<255] = 0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frame1, 10, cv2.CV_32S)

    imgC = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)

    for j in range(len(stats)):
        stat = stats[j]
        x, y, w, h, area = stat

        if area != np.max(stats[:, 4]):  # Exclude background
            if area > 100:
                cv2.rectangle(imgC, (x, y), (x + w, y + h), (255, 0, 0), 2)
                x, y = centroids[j].astype(int)
                cv2.circle(imgC, (x, y), 2, (0, 0, 255), 2)
                
                current_centroids_list.append([int(x),int(y)])

    current_centroids_list = [[x, y] for x, y in current_centroids_list if x > 50 and x < 740 and y > 40 and y < 580]

    # print(len(loc_of_markers), len(current_centroids_list))
    
    previous_positions = loc_of_markers.copy()
    cleaned_list = match_markers(loc_of_markers, current_centroids_list, previous_positions)

    for j in range(63):
        cv2.arrowedLine(frame, tuple(loc_of_markers[j]), tuple(cleaned_list[j]), (0, 0, 255), 2,  tipLength=0.2)

    if calibrate:
        cv2.imshow('mask',mask_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    imgH = np.hstack((imgC, frame))
    cv2.imshow('Processed Image', imgH)


cap.release()
cv2.destroyAllWindows()