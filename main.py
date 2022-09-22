import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def read_mask_image(name):
  image = st.file_uploader("Upload an "+ name, type=["jpg", "jpeg"])
  if image:
    im = Image.open(image)
    im.filename = image.name
    return im
    
def _draw_mask(mask):    
  mask = cv2.cvtColor(np.array(mask), cv2.COLOR_BGR2GRAY)
  cnts,_= cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  return cnts   
    
def show_image(image, mask):
  cnts = _draw_mask(mask)
  _drawBboxMask(image, cnts)
  __suspected_tuberculosis(image, cnts)
  

def _drawBboxMask(image, cnts):
  fig, (ax1, ax2) = plt.subplots(1, 2)
  ax1.axis('off')
  ax2.axis('off')
  _draw_bbox(image, cnts, ax1)
  _mask(image, cnts, ax2)
  st.pyplot(fig)  
  
  
def _draw_bbox(image, cnts, ax):
  ax.imshow(image)
  for c in cnts:
    area = cv2.contourArea(c)
    if area < 10:
      continue
    [x, y, w, h] = cv2.boundingRect(c)
    ax.add_patch(Rectangle((x, y), w, h, color = "red", fill = False)) 
    
    
def _mask(image, cnts, ax):
  img = _detectionAlteration(image, cnts, False)
  ax.imshow(img)       
  
  
def _detectionAlteration(image, cnts, fill=True):
  '''
        In Marie's orginal project for an intelligent model for image
      segmentation and detection of pathological features, this algorithm's is owned by the company Beevi and
      is in the patent process at INPE - Brazil. This is a simplified segmentation and detection algorithm. 
        One of the feature detection algorithms is Fast RCNN, which was trained with 3 million images. Due to the LGPD in Brazil, the data belongs to the company and is not released for distribution, respecting the General Law for the Protection of Personal Data.
  '''
  image = np.array(image)
  markers = np.zeros((image.shape[0], image.shape[1]))
  heatmap_img = cv2.applyColorMap(image, cv2.COLORMAP_JET)
  t = 2
  if fill:
    t = -1
  cv2.drawContours(markers, cnts, -1, (255, 0, 0), t)
  mask = markers>0
  image[mask,:] = heatmap_img[mask,:]
  return image  
  
  
def _suspected_tuberculosis(image, cnts):
  fig2 = plt.figure()
  plt.axis('off')
  hm = st.slider("Changed Regions", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
  img = _detectionAlteration(image, cnts)
  plt.imshow(img, alpha=hm)
  plt.imshow(image, alpha=1-hm)
  plt.title("Result")
  st.pyplot(fig2) 
  
def main():
  image = read_mask_image('image')
  '''
      In Marie's orginal project for an intelligent model for image
      segmentation, this algorithm's is owned by the company Beevi and
      is in the patent process at INPE - Brazil.
      
  '''    
  
  mask = read_mask_image('mask')
  if image and mask:
    show_image(image, mask)     
