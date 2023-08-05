import cv2
from ultralytics import YOLO
import cvzone
import math
from sort import *
import numpy as np



vid = cv2.VideoCapture('test video/cars.mp4')
frame_width=int(vid.get(3))
frame_height=int(vid.get(4))

model = YOLO('yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
  
mask = cv2.imread('mask.png')


#Tracking 
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)
limits=[423,297,673,297]
total_counts=[]

video=cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'MJPG'),10,(frame_width,frame_height))
    
while(True):
  
    ret, frame = vid.read()
    imgRegion=cv2.bitwise_and(frame,mask)
    imgGraphics=cv2.imread("graphics.png",cv2.IMREAD_UNCHANGED)
    frame=cvzone.overlayPNG(frame,imgGraphics,(0,0))
    results=model(imgRegion,stream=True)
    
    detections = np.empty((0,5))
    
    for r in results:
        boxes=r.boxes
        for box in boxes:
            (x1,y1,x2,y2)=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1
            
            conf=math.ceil((box.conf[0]*100))/100
            # print(conf)
            cls=int(box.cls[0])
            
            currentclass = classNames[cls]
            
            if currentclass == 'car' or currentclass == 'motorbike' or currentclass == 'bus' or currentclass == 'truck' and conf >0.3:
                # cvzone.putTextRect(frame,f'{currentclass} {conf}',(max(0,x1),max(35,y1)),scale=0.6,thickness=1,offset=3)
                cvzone.cornerRect(frame,(x1,y1,w,h),l=9,rt=5)

                currentArray=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currentArray))

            
            
    resultTracker=tracker.update(detections)
    cv2.line(frame,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),5)
    for result in resultTracker:
        x1,y1,x2,y2,id=result
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
       
        w , h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame,(x1,y1,w,h),l=5,rt=1,colorR=(255,0,0))
        cvzone.putTextRect(frame,f'{int(id)}',(max(0,x1),max(35,y1)),scale=2,thickness=3,offset=10)
        cx,cy=x1+w//2,y1+h//2
        cv2.circle(frame,(cx,cy),5,(255,0,255),cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if total_counts.count(id)==0:
                total_counts.append(id)
                cv2.line(frame,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)
        # cvzone.putTextRect(frame,f'Count:{len(total_counts)}',(50,50))
        
        
    cv2.putText(frame,str(len(total_counts)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)
            






    video.write(frame)
    cv2.imshow('frame', frame)
    # cv2.imshow('imgregion',imgRegion)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release() 
vid.release()
cv2.destroyAllWindows()   
