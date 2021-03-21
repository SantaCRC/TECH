import numpy as np 
import cv2
import pyfirmata
import time
from numpy import ones,vstack
from numpy.linalg import lstsq
board = pyfirmata.Arduino('/dev/ttyACM0')
print("Communication Successfully started")
usleep = lambda x: time.sleep(x/1000000.0)

# Capturing video through webcam
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_BUFFERSIZE,5)
ancho = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
mitadx = int(ancho/2)
derecha = int(540)
izquierda = int(540)
izquierda = int(mitadx+izquierda)
derecha = int(mitadx-derecha)
print('{} con mitad en {}, giro derecha en {} y giro en izquierda en {}'.format(ancho,mitadx,derecha,izquierda))
def recta(x,y):
    points = [(x),(y)]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords,ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]
    return c,m
def interseccion(y,c,m):
    x=(y-c)/m
    return x
    
    
# Start a while loop 
while(1): 
      
    # Reading the video from the 
    # webcam in image frames 
    _, imageFrame = webcam.read()
    azul=cv2.line(imageFrame,(derecha,int(alto/2)),(mitadx,int(alto/2)),(255,0,0),5)
    verde=cv2.line(imageFrame,(mitadx,0),(mitadx,alto),(55,55,0),5)
    roja=cv2.line(imageFrame,(mitadx,int(alto/2)),(izquierda,int(alto/2)),(0,0,255),5)
  
    # Convert the imageFrame in  
    # BGR(RGB color space) to  
    # HSV(hue-saturation-value) 
    # color space 
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV) 
  
    # Set range for red color and  
    # define mask 
    red_lower = np.array([12, 40, 60], np.uint8) 
    red_upper = np.array([30, 255, 255], np.uint8) 
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 
  
    # Set range for green color and  
    # define mask 
      
    # Morphological Transform, Dilation 
    # for each color and bitwise_and operator 
    # between imageFrame and mask determines 
    # to detect only that particular color 
    kernal = np.ones((5, 5), "uint8") 
      
    # For red color 
    red_mask = cv2.dilate(red_mask, kernal) 
    res_red = cv2.bitwise_and(imageFrame, imageFrame,  
                              mask = red_mask) 
      
   
    # Creating contour to track red color 
    contours, hierarchy = cv2.findContours(red_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE) 
      
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 300): 
            x, y, w, h = cv2.boundingRect(contour) 
##            imageFrame = cv2.rectangle(imageFrame, (x, y),  
##                                       (x + w, y + h),  
##                                       (0, 0, 255), 2) 
              
            cv2.putText(imageFrame, "amarillo", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                        (0, 0, 255))
            if(x>mitadx ):
                linea=cv2.line(imageFrame,(x,y),(x+w,y+h),(0, 0, 255), 2)
                c,m=recta([x,y],[x+w,y+h])    
                x2=interseccion(int(alto/2),c,m)
                cv2.circle(imageFrame, (int(x2),int(alto/2)), 5, (0, 255, 255), 2)
    
            elif(x<mitadx):
                lineb=cv2.line(imageFrame,(x+w,y),(x,y+h),(0, 0, 255), 2)
                c2,m2=recta([x+w,y],[x,y+h])
                x3=interseccion(int(alto/2),c2,m2)
                cv2.circle(imageFrame, (int(x3),int(alto/2)), 5, (0, 255, 0), 2)
            try:
                if(x3 <= mitadx and x3>= derecha and x2>=mitadx):
                    print('derecha')
                    board.digital[3].write(1)
                    for i in range(400):
                        board.digital[2].write(1)
                        usleep(400)
                        board.digital[2].write(0)
                        usleep(400)
                elif(x2>mitadx and x2<= izquierda and x3<=mitadx):
                    print('izquierda')
                    board.digital[3].write(0)
                    for i in range(400):
                        board.digital[2].write(1)
                        usleep(400)
                        board.digital[2].write(0)
                        usleep(400)
            except:
                pass
                
  
    # Creating contour to track green color 
##    contours, hierarchy = cv2.findContours(green_mask, 
##                                           cv2.RETR_TREE, 
##                                           cv2.CHAIN_APPROX_SIMPLE) 
##      
##    for pic, contour in enumerate(contours): 
##        area = cv2.contourArea(contour) 
##        if(area > 300): 
##            x, y, w, h = cv2.boundingRect(contour) 
##            imageFrame = cv2.rectangle(imageFrame, (x, y),  
##                                       (x + w, y + h), 
##                                       (0, 255, 0), 2) 
##              
##            cv2.putText(imageFrame, "Green Colour", (x, y), 
##                        cv2.FONT_HERSHEY_SIMPLEX,  
##                        1.0, (0, 255, 0)) 
##  
##    # Creating contour to track blue color 
##    contours, hierarchy = cv2.findContours(blue_mask, 
##                                           cv2.RETR_TREE, 
##                                           cv2.CHAIN_APPROX_SIMPLE) 
##    for pic, contour in enumerate(contours): 
##        area = cv2.contourArea(contour) 
##        if(area > 300): 
##            x, y, w, h = cv2.boundingRect(contour) 
##            imageFrame = cv2.rectangle(imageFrame, (x, y), 
##                                       (x + w, y + h), 
##                                       (255, 0, 0), 2) 
##              
##            cv2.putText(imageFrame, "Blue Colour", (x, y), 
##                        cv2.FONT_HERSHEY_SIMPLEX, 
##                        1.0, (255, 0, 0)) 
              
    # Program Termination 
    cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame) 
    if cv2.waitKey(10) & 0xFF == ord('q'): 
        cap.release() 
        cv2.destroyAllWindows() 
        break
