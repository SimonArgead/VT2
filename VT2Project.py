import cv2
import numpy as np
import math
import scipy
from scipy.spatial.distance import cdist
import time

def ShiTomasi(input2gray):
    # Shi-Tomasi corner detection function
    # We are detecting only 100 best corners here
    corners = cv2.goodFeaturesToTrack(img2, 30, 0.01, 10)
    # convert corners values to integer
    corners = np.int0(corners)
    
    #we can compute the euclidean distance between the points.
    #In case of a small distance, we discard the points.
    #in case of a distance to large, we also discard them. Using this, we can get the points we want.
    
    coordinate1 = corners[0]
    coordinate2 = corners[1]
    coordinate3 = corners[3]
    
    hyp = float(scipy.spatial.distance.cdist(coordinate1,coordinate2))
    cathodeA = float(scipy.spatial.distance.cdist(coordinate1,coordinate3))
    cathodeB = float(scipy.spatial.distance.cdist(coordinate2,coordinate3))
    
    return(hyp, cathodeA, cathodeB)
    

if __name__ == "__main__":
    start = time.time()
    input1 = cv2.imread("C:/Users/shans/OneDrive/Skrivebord/AAU/8 Semester/Billeder/VT2.8Reduced.png", cv2.IMREAD_COLOR)
    input2 = cv2.imread("C:/Users/shans/OneDrive/Skrivebord/AAU/8 Semester/Billeder/VT2.2Reduced.png", cv2.IMREAD_COLOR)
    #img1 = cv2.fastNlMeansDenoisingColored(input1,None,7,21,5,10)
    #img2 = cv2.fastNlMeansDenoisingColored(input2,None,7,21,5,10)
    input1gray = cv2.cvtColor(input1, cv2.COLOR_BGR2GRAY)
    input2gray = cv2.cvtColor(input2, cv2.COLOR_BGR2GRAY)
    img1 = cv2.GaussianBlur(input1gray, (3, 3), 0)
    img2 = cv2.GaussianBlur(input2gray, (3, 3), 0)
    min_area = 5000
    max_area = 100000
    image_number = 0
    
    # Apply edge detection method on the image
    edges = cv2.Canny(img1,50,150,apertureSize = 3) #img1,100,200,apertureSize = 3
    
    #we start by finding the circle
    detected_circles = cv2.HoughCircles(edges, 
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
                param2 = 30, minRadius = 12, maxRadius = 40)
    Center_circle = []
    #Draw circles that are detected.
    if detected_circles is not None:
  
    # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
  
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            
            # Draw the circumference of the circle.
            cv2.circle(input1, (a, b), r, (0, 255, 0), 2)
            
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(input1, (a, b), 1, (0, 0, 255), 3)
            center = [a,b]
            Center_circle.append(center)
            #convert pixels to mm
            d = r*0.26458*2
            
    #we move on to find the square
    cnts = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    squarePoints = []
    squareDist = []
    
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area and area < max_area:
            x,y,w,h = cv2.boundingRect(c)
            ROI = img1[y:y+h, x:x+w]
            cv2.rectangle(input1, (x, y), (x + w, y + h), (36,255,12), 2)
            image_number += 1
            areamm = area*0.26458
    p1 = np.array([[x,h]])
    p2 = np.array([[x,w]])
    p3 = np.array([[y,h]])
    p4 = np.array([[y,w]])
    squarePoints.append(p1)
    squarePoints.append(p2)
    squarePoints.append(p3)
    squarePoints.append(p4)
    
    #now we want to calculate the distance between the circle and the 2 nearest edges of the object.
    #We want to do the same for the square.

    #Shi-Tomasi
    corners = cv2.goodFeaturesToTrack(img1, 22, 0.01, 10)
    # convert corners values to integer
    corners = np.int0(corners)
    
    for i in corners:
        x, y = i.ravel()
        cv2.circle(input1, (x, y), 3, (255, 0, 0), -1)
    
    CornerPoints = []
    distance = []
    
    for e in range(len(corners)-1):
        k = e
        val1 = corners[k]
        for j in range(len(corners)):
            h = j
            val2 = corners[h]
            dist = float(scipy.spatial.distance.cdist(val1,val2))
            if dist < 1500 and dist > 950:
                CornerPoints.append([val1, val2])
                distance.append(dist)
    
    #distance from the circle center to 2 of the edges
    #corrections are needed
    # for m in range(len(distance)):
    #     l = m
    #     side = distance[l]
    #     if side < 480 and side > 476:
    #         break
    
    maximum = 1
    #split_points =  np.array_split(CornerPoints,1)
    for o in range(len(distance)):
        value = distance[o]
        if value > maximum:
            maximum = value
            indice = o
        
    points = CornerPoints[indice] #split into 2
    #split_points =  np.array_split(points,2)
    # Her har vi lige noget der skal rettes op på
    P1 = points[0]
    #P1 = np.int0(P1)
    P2 = points[1]
    #P2 = np.int0(P2)
    #circlec = np.int0([Center_circle])
    shape = np.shape(Center_circle)
    circlec = shape
    circlec = np.int0([circlec])
    distC = []
    if len(Center_circle) > 1:
        for i in range(len(Center_circle)):
            cc = Center_circle[i]
            cc = np.int0([cc])
            location_distance1 = float(scipy.spatial.distance.cdist(P1,cc))*0.26458
            location_distance2 = float(scipy.spatial.distance.cdist(P2,cc))*0.26458
            dist = [location_distance1, location_distance2]
            distC.append(dist)
    else:
        location_distance1 = float(scipy.spatial.distance.cdist(P1,circlec))*0.26458
        location_distance2 = float(scipy.spatial.distance.cdist(P2,circlec))*0.26458
    #location_distance1 = math.sqrt((278-117)**2+(384-466)**2)*0.26458#should have been Center_circle[0] and P1[0], P1[1]. But somehow, that's out of bounds...?
    #location_distance2 = math.sqrt((278-596)**2+(384-466)**2)*0.26458
    
    #Now we do the same for the square. The principle is the exact same. We just don't need to find the 2 corner points
    BottomPoints = squarePoints[0] #prev 5. We choose the bottom points. It doesn't matter in principle, but we just do.
    split_points =  np.array_split(BottomPoints,2)
    square_distance1 = float(scipy.spatial.distance.cdist(squarePoints[0],P1))*0.26458 #because an array like: array = [x,y] is not 2D, I had to make it like this. 
    square_distance2 = float(scipy.spatial.distance.cdist(squarePoints[1],P2))*0.26458
    square_distance3 = float(scipy.spatial.distance.cdist(squarePoints[0],P1))*0.26458
    square_distance4 = float(scipy.spatial.distance.cdist(squarePoints[1],P2))*0.26458
    
    #We now need to compute the angle of the bend on the object. To do this, we need a new image
    #since we can't see it on the one we already used (at least not enough).



    [hyp, cathodeA, cathodeB] = ShiTomasi(img2)
    
    angle = np.rad2deg(math.acos((cathodeA**2+cathodeB**2-hyp**2)/(2*cathodeA*cathodeB))) 
    end = time.time()
    
    #start_point = (16,12)
    #end_point = (979,32)
    # Green color in BGR
    #color = (0, 0, 255)
    # Line thickness of 9 px
    #thickness = 9
    #image = cv2.line(input1, start_point, end_point, color, thickness)
    
    # Using cv2.line() method
    # Draw a diagonal green line with thickness of 9 px
    # Displaying the image 
    #cv2.imshow("image", image) 
    
    
    if len(Center_circle) > 1:
        print("Circle distances to corners are:", distC, "mm")
        for i in range(len(Center_circle)):
            j = i+1
            print("The diameter of circle", j, "is:", d, "mm")
            
    else:
        print("The diameter of the circle is:", d, "mm")
        print("Circle distance to corners is:", location_distance1, "and", location_distance2, "mm")
        
        
    print("The area of the square is:", areamm, "mm")
    print("the largest angle is: ", angle, "degrees") #the actual angle is 135, so the result is not great, not terrible
    print("Square distance to corner is:", square_distance1,"and", square_distance2, "for the first corner.", square_distance3, "and", square_distance4, "for the other. All in mm")
    print("Program execution time :", end-start, "seconds")
    
    # Showing the final image.
    cv2.imshow("Image with Borders", input2)
    #cv2.imshow("Detected features", input2)
    #cv2.imshow("angle", edges)
    cv2.waitKey(0)