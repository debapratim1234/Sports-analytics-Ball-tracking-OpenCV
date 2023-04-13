import cv2
import cvzone
from cvzone.ColorModule import ColorFinder

# Initialize the Video
cap = cv2.VideoCapture("Resources/Video (1).mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
result = cv2.VideoWriter('output.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

# Create the ColorFinder object
myColorFinder = ColorFinder(False)
hsvVals={'hmin': 155, 'smin': 75, 'vmin': 164, 'hmax': 176, 'smax': 248, 'vmax': 255}

# Position List
posList = []

while True:
    # To read Video
    success,img=cap.read()
    # To find the colour of the Basket-ball, uncomment this part
    #img=cv2.imread("Ball.PNG")
    #img = img[0:500,0:500]

    # Find the color of the Basket-Ball
    imgColor,mask = myColorFinder.update(img,hsvVals)

    # Find location of the Basket-ball
    imgContour, contour = cvzone.findContours(img,mask,minArea=1)

    # Ball-tracking
    if contour:
        posList.append(contour[0]['center'])
    for i,pos in enumerate(posList):
        cv2.circle(imgContour,pos,9,(0,255,0),cv2.FILLED)
        if i==0:
            cv2.line(imgContour,pos,pos,(0,0,255),3)
        else:
            cv2.line(imgContour,pos,posList[i-1],(0,0,255),3)
    result.write(imgContour)

    # DISPLAY
    # We resize the image as it is too big to fit the screen
    #img = cv2.resize(img, (0, 0), None, 0.6, 0.6)
    imgColor=cv2.resize(imgColor,(0,0),None,0.7,0.7)
    imgContour=cv2.resize(imgContour,(0,0),None,0.5,0.5)
    #cv2.imshow("Image",img)
    cv2.imshow("ImageContour", imgContour)
    cv2.waitKey(200)
result.release()
cap.release()
cv2.destroyAllWindows()