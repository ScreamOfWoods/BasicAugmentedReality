#!/usr/bin/python
import cv2

def display(im, bbox):
    n = len(bbox)
    coords = []
    for i in range(n):
        cv2.line(im, tuple(bbox[i][0]), tuple(bbox[ (i+1)%n ][0]), (255,0,0), 3)
        coords.append(tuple(bbox[i][0]))
        coords.append(tuple(bbox[(i + 1) % n][0]))
    return im, coords


fps_cap = 30
fps_cnt = 0

cam = cv2.VideoCapture(0)

#cv2.namedWindow("Test")
qrDecoder = cv2.QRCodeDetector()
coords = []
while True:
    ret, frame = cam.read()
    if not ret:
        break
    data, bbox, rectified = qrDecoder.detectAndDecode(frame)
    if len(data) > 0:
        frame, coords = display(frame, bbox)
        cv2.imshow("Rectified", rectified)
        print("Qr Data: " + data + " Coords:  ")
        print(coords)
        #COORDS holds the points that form the lines arround the qr code
        #a single tuple represents a point defined by X and Y coordinate
        #two tuples define a single line
        #the list contains 4 pairs of tuples:
        # [ (startLineX1, startLineY1), (endLineX1, endLineY1) ... (startLineX4, startLineY4), (endLineX4, endLineY4)  ]
    
    cv2.imshow("Found Qr", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cam.release()
cv2.destroyAllWindows()



