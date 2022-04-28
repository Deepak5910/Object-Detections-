#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install opencv-python')


# In[4]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


yolo  = cv2.dnn.readNet("./yolov3-tiny.weights" ,"./yolov3-tiny.cfg")


# In[6]:


classes = []
with open("./coco.names",'r') as f:
  classes  = f.read().splitlines()


# In[63]:


img  = cv2.imread("./soccer22.jpg")
height,width,_ = img.shape
blob = cv2.dnn.blobFromImage(img, 1/255, (320,320), (0,0,0), swapRB = True, crop  = False)


# In[64]:


i = blob[0].reshape(320,320,3)


# In[65]:


yolo.setInput(blob)


# In[66]:


output_layes_name = yolo.getUnconnectedOutLayersNames()
layeroutput = yolo.forward(output_layes_name)


# In[67]:


boxes = []
confidences = []
class_ids = []

for output in layeroutput:
  for detection in output:
    score = detection[5:]
    class_id = np.argmax(score)
    confidence = score[class_id]
    if confidence > 0.7:
      center_x = int(detection[0]*width)
      center_y = int(detection[0]*height)
      w = int(detection[0]*width)
      h = int(detection[0]*height)

      x = int(center_x-w/2)
      y = int(center_y-h/2)

      boxes.append([x,y,w,h])
      confidences.append(float(confidence))
      class_ids.append(class_id)


# In[68]:


indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


# In[69]:


font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size = (len(boxes), 3))


# In[70]:


if len(indexes) > 0:
  for i in indexes.flatten():
     x,y,w,h = boxes[i]
   
     label = str(classes[class_ids[i]])
     confi = str(round(confidences[i],2))
     color = colors[i]
   
     cv2.rectangle(img , (x,y),(x+w ,y+h),color, 1)
     cv2.putText(img, label +" "+confi,(x, y+20),font,2,(255,255,255),1)


# In[71]:


plt.imshow(img)


# In[72]:


cv2.imwrite("output.jpg",img)


# In[ ]:




