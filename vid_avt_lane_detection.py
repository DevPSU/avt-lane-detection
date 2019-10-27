import cv2
import numpy as np

def canny(src):
    # convert image to grayscale color
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # canny grayscale image 
    canny = cv2.Canny(gray, 50,150)
    return canny

def region_of_interest(src):
    HEIGHT = src.shape[0]-100
    # make a roi polygon
    polygons = np.array([
    [(200, HEIGHT),(1000,HEIGHT),(200, 200)]
    ])

    #poly = np.array([
    #  [(0,(src.shape[0]/2)), ((src.shape[1]/2),200)  , (src.shape[1],(src.shape[0]/2))]
    #])

    mask = np.zeros_like(src)
    #fill mask with roi polygon
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(src,mask)
    return masked_image

 #-------------------------------------   



# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('videosample.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
 
    frame = canny(frame)
    processed_frame = region_of_interest(frame)
    # Display the resulting frame
    cv2.imshow('Frame',processed_frame)
 
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