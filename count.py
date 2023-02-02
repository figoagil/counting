import cv2
import numpy as np
import time
from time import sleep

def importvideo(inputvideo):
    global cap
    cap = cv2.VideoCapture(inputvideo)
    return cap
    
importvideo("Traffic.mp4")    
    
    
frame_id = 0
nomor = 0
fpstotal = []
path_label = 'coco.names'
classes = []
with open (path_label, 'rt') as f:
    #classes = [line.strip() for line in f.readlines()]
    #classes = f.read().rstrip('\n').split('\n')
    classes=[line.rstrip() for line in f]
countline = 336
wght_hght_target = 416
model_Config = 'yolov4-tiny.cfg'
model_Weights = 'yolov4-tiny.weights'
confThreshold = 0.4
nmsThreshold = 0.4
inccount1 = 0
#inccount2 = 0
#inccount3 = 0
#inccount4 = 0
#inccount5 = 0
inccount_reset = 0
start_time = time.time()

network = cv2.dnn.readNetFromDarknet(model_Config,model_Weights)
network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
network.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

result = cv2.VideoWriter('result.avi',cv2.VideoWriter_fourcc(*'XVID'),20,(640,480))

def centerdot(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy
    
def FPScount(fps):
    global fpstotal
    fpstotal.append(fps)

    
    
def findObject(outputs,img):
    heightTar, weightTar, channelsTar =img.shape
    bbox = []
    classId = []
    confidence = []
    count1 = 0
    global nomor
    for output in outputs:
        for det in output:
            score = det[5:]
            classIds = np.argmax(score)
            confids = score[classIds]
            if classIds == 2 or classIds == 3 or classIds == 7 or classIds == 5:
                if  confids > confThreshold:
                    w = int(det[2]*weightTar)
                    h = int(det[3]*heightTar)
                    x = int((det[0]*weightTar)-w/2)
                    y = int((det[1]*heightTar)-h/2)
                    bbox.append([x,y,w,h])
                    confidence.append(float(confids))
                    center = centerdot(x,y,w,h)
                    if 432 > y:
                        file='C:/Figo File/Semester 8/TA/count/foto/foto'+str(nomor)+'.png'
                        cv2.imwrite(file,img)
                        nomor = nomor +1
                        count1 = count1 +1
                        cv2.line(img, (0,countline),(640,countline) , (100,0,0), 3)                        
            else:
                continue
    draw_box = cv2.dnn.NMSBoxes(bbox, confidence, confThreshold,nmsThreshold)
    for i in draw_box:
        i = i[0]
        box = bbox[i]
        center = centerdot(x,y,w,h)
        x,y,w,h = box[0],box[1],box[2],box[3] 
        cv2.rectangle(img,(x,y),(x+w, y+h),(255,255,0),2)
        cv2.putText(img,f'Car {int(confidence[i]*100)}%', (x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,0,0), 2)
        cv2.circle(img,center, 4, (0,0,255),-1)   
    return count1
# get fps from original video
#vid_fps =int(cap.get(cv2.CAP_PROP_FPS))
#vid_width,vid_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#out = cv2.VideoWriter('results.avi', codec, vid_fps, (vid_width, vid_height))

while True:
    success, img = cap.read()
    img = cv2.resize(img,(640,480))
    frame_id += 1
    cv2.imshow('video',img)
    blob = cv2.dnn.blobFromImage(img,1/255,(wght_hght_target,wght_hght_target),[0,0,0,0],1,crop=False)
    network.setInput(blob)
    LayerNames = network.getLayerNames()
    outputNames = [LayerNames[i[0]-1] for i in network.getUnconnectedOutLayers()]
    outputs = network.forward(outputNames)
    cv2.line(img,(330,countline),(640,countline),(0,0,100),3) 
    counter1 = findObject(outputs,img)
    
    inccount1 = inccount1 + counter1
          
    run_time = time.time()
    inccount_reset = int(time.time()-start_time)
    if inccount_reset == 3600 :
        inccount1 = 0
        inccount_reset = 0
        start_time = run_time 
    cv2.putText(img,f'counting Car : {inccount1}', (25,20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,0,0), 2) 
    elapsed_time = time.time() - start_time
    fp = frame_id / elapsed_time
    fps = round(fp, 2)
    FPScount(fps)
    cv2.putText(img, "FPS: " + str(round(fps, 2)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 0), 2)
    print("car is detected : "+str(inccount1))
    print(fpstotal)
    result.write(img)
    
    cv2.imshow('video',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
result.release()
cv2.destroyAllWindows()