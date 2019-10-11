import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    # create image variable
    img = cv2.imread(image)
    # convert image to grayscale color
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # canny grayscale image 
    canny = cv2.Canny(gray, 50,150)
    return canny

def region_of_interest(image):
    HEIGHT = image.shape[0]-100
    # make a roi polygon
    polygons = np.array([
    [(400, HEIGHT),(1000,HEIGHT),(800, 450)]
    ])
    mask = np.zeros_like(image)
    #fill mask with roi polygon
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # determine linear line of best fit
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        # negative slope corresponds to left lane, positive to right lane
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
            
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    # create the left and right line coordinates
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])
    
        
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return line_image

# Working on some sort of median line
def midpoint(image, average_lines):
    left_point = (average_lines[0][0],average_lines[0][1])
    right_point = (average_lines[1][0],average_lines[1][1])

    # Calculate the midpoint
    midpoint_x = int((right_point[0]-left_point[0])/2 + left_point[0])
    midpoint_y = 720
    midpoint = (midpoint_x, midpoint_y)

    # Draw the midpoint
    cv2.circle(image, midpoint, 10, (0,255,0), -1)
    return image

# carsample.jpg is a temporary image from the data set
lane_image = cv2.imread("carsample.jpg")

# apply canny edge detection and region of interest bounding
canny_image = region_of_interest(canny('carsample.jpg'))

# apply HoughLines to find lines in image
lines = cv2.HoughLinesP(canny_image,2,np.pi/180, 20, np.array([]), minLineLength=40, maxLineGap=5)

# find an average line to use for the left and right lane 
averaged_lines = average_slope_intercept(lane_image, lines)

# line_image contains only the lane lanes drawn
line_image = display_lines(lane_image, averaged_lines)

# combo_image combines the line_image and lane_image
lane_with_lines_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
lane_with_lines_image = cv2.addWeighted(lane_with_lines_image, 0.8, midpoint(lane_image, averaged_lines), 1, 1)

# plt helps show the image in a graph
#plt.imshow(lane_image)
#plt.show()

# display image
cv2.imshow("image", lane_with_lines_image)
cv2.waitKey(0)


