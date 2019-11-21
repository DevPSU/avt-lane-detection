import cv2
import numpy as np

def canny(src):
    # convert image to grayscale color
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # canny grayscale image 
    canny = cv2.Canny(gray, 50, 150)
    return canny

def region_of_interest(src):
    HEIGHT = src.shape[0]-100
    # make a roi polygon
  
    # height = 620
    polygons = np.array([
    [(200, HEIGHT),(900,HEIGHT),(900,HEIGHT-300), (200,HEIGHT-300)]
    ])

    # RECTANGLE 
    # 200, HEIGHT = Bottom Right point
    # 900, HEIGHT = Top Right point
    # 900, HEIGHT-300 = Bottom Left point
    # 200, HEIGHT-300 = Top Left point

    mask = np.zeros_like(src)
    #fill mask with roi polygon
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(src,mask)
    return masked_image

def displayLines(frame, lines):
  line_image = np.zeros_like(frame)
  if lines is not None:
    for line in lines:
      for x1, y1, x2, y2 in line:
          cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
          lines_edge = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
  return line_image

def average_slope_intercept(frame, lines):
    
    
    #This function combines line segments into one or two lane lines
    #If all line slopes are < 0: then we only have detected left lane
    #If all line slopes are > 0: then we only have detected right lane
    
    left_fit = []
    right_fit = []

    if lines is None:
        print('No line segments detected')
        return []

    height, width, _ = frame.shape

    boundary = 1/3
    left_region_boundary = width * (1 - boundary) # left line segment should be on the left 2/3 portion of screen
    right_region_boundary = width * boundary      # right line segment should be on the right 2/3 portion of screen

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        
        for x1, y1, x2, y2 in line:
          if x1 == x2:
            print("skipping verticle line segment")
            continue

          # determine linear line of best fit
          parameters = np.polyfit((x1, x2), (y1, y2), 1)
          slope = parameters[0]
          intercept = parameters[1]
          
          # negative slope corresponds to right lane, positive to left lane
          if slope < 0:
              right_fit.append((slope, intercept))
          else:
              left_fit.append((slope, intercept))
          if not left_fit:
            print("Couldn't detect left lane lines")
          if not right_fit:
              print("Couldn't detect right lane lines")
          left_fit_average = np.average(left_fit, axis = 0)
          right_fit_average = np.average(right_fit, axis = 0)

    # create the left and right line coordinates
    left_line = make_coordinates(frame, left_fit_average)
    right_line = make_coordinates(frame, right_fit_average)
    return np.array([left_line, right_line])

def make_coordinates(frame, line_parameters):
    try:
      slope, intercept = line_parameters
    except TypeError:
      slope, intercept = 0,0

    y1 = frame.shape[0]
    y2 = int(y1*(3/5))

    if slope == 0:
      if intercept == 0:
        x1 = y1
        x2 = y2
    else:
      x1 = int((y1-intercept)/slope)
      x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])

def display_lines(frame, lines):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return line_image

 #-------------------------------------   



# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
#cap = cv2.VideoCapture('videosample.mp4')
cap = cv2.VideoCapture('videosample2.mp4')
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    # add canny and region of interest to image
    processed_frame = region_of_interest(canny(frame))
    #processed_frame = region_of_interest(frame)
    lines = cv2.HoughLinesP(processed_frame, 1, np.pi/180, 50, maxLineGap=50)

    # find an average line to use for the left and right lane 
    averaged_lines = average_slope_intercept(frame, lines)

    # line_image contains only the lane lines drawn
    line_image = display_lines(frame, averaged_lines)


    lane_with_lines_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    # Display the resulting frame
    cv2.imshow('Original Frame',processed_frame)
    cv2.imshow('Processed Frame',lane_with_lines_image)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()