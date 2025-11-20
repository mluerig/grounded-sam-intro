#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:54:00 2024

@author: mlurig@ad.ufl.edu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:52:50 2024

@author: mlurig@ad.ufl.edu
"""
#%% imports

import copy
import numpy as np
from io import StringIO
from contextlib import redirect_stdout

import cv2
from phenopype.core.segmentation import detect_contour

#%% functions
def filter_mask(image, mask, min_area, ret_area=False):    

        ## convert to binary mask
        mask = mask.astype(np.uint8) * 255

        ## detect all contours above threshold
        with redirect_stdout(StringIO()):
            annotations = detect_contour(mask, min_area=min_area, stats_mode="circle")
        countour_coords = annotations["contour"]["a"]["data"]["contour"]
        contour_info = annotations["contour"]["a"]["data"]["support"]

        ## find largest contour
        idx_largest = max(range(len(contour_info)), key=lambda i: contour_info[i]['area']) 
        coords = countour_coords[idx_largest]
        contour_info = contour_info[idx_largest]
        contour_info = {key: value for key, value in contour_info.items() if not key[:9] == "hierarchy"}

        ## draw mask
        rx, ry, rw, rh = cv2.boundingRect(coords)
        object_mask = np.zeros((rh, rw), dtype="uint8")
        object_mask = cv2.drawContours(
            image=object_mask,
            contours=[coords],
            contourIdx=0,
            thickness=-1,
            color=255,
            offset=(-rx, -ry),
        )          
        
        ## extract image and change background to white 
        object_image = copy.deepcopy(image[ry : ry + rh, rx : rx + rw])
        object_image[object_mask==0] = 0
        
        # Convert to RGBA
        rgba_image = cv2.cvtColor(object_image, cv2.COLOR_RGB2RGBA)
        rgba_image[:, :, 3] = object_mask
        
        ## prep info
        info = {
            "area": contour_info["area"],
            "bbox":  [rx, ry, rw, rh],
            "center": contour_info["center"],
            "diameter": contour_info["diameter"],
            }
        
        return rgba_image, info
    
def filter_mask_list(image, masks, image_name=None, min_area=10000, ret_max_area=False):

        # Check if there are any masks
        assert len(masks) > 0, "No detections!"

        # Iterate over all masks and perform bitwise AND operation
        mask = masks[0]
        for m in masks[1:]:
            mask = np.bitwise_or(mask, m)
        mask = mask.astype(np.uint8) * 255

        ## detect all contours above threshold
        with redirect_stdout(StringIO()):
            annotations = detect_contour(mask, min_area=min_area)
        countour_coords = annotations["contour"]["a"]["data"]["contour"]
        contour_info = annotations["contour"]["a"]["data"]["support"]

        l_area, l_info, l_image, l_mask = [], [], [], []
        for coords, info in zip(countour_coords, contour_info):
            
            ## draw mask
            rx, ry, rw, rh = cv2.boundingRect(coords)
            object_mask = np.zeros((rh, rw), dtype="uint8")
            object_mask = cv2.drawContours(
                image=object_mask,
                contours=[coords],
                contourIdx=0,
                thickness=-1,
                color=255,
                offset=(-rx, -ry),
            )          
            
            ## extract image
            object_image = image[ry : ry + rh, rx : rx + rw]
            
            ## change background to white 
            object_image[object_mask==0] = 255
            
            # Convert to RGBA
            rgba_image = cv2.cvtColor(object_image, cv2.COLOR_RGB2RGBA)
            rgba_image[:, :, 3] = object_mask
            
            l_area.append(info["area"]) 
            
            l_info.append(info)
            l_image.append(rgba_image) 
            l_mask.append(object_mask) 

        if ret_max_area:
            largest = l_area.index(max(l_area))            
            return l_image[largest], l_mask[largest], l_info[largest]
        else:
            return l_image, l_mask, l_info
            

#%% temp for pres
def draw_contours_on_image(image, mask, min_area=10000, alpha=0.5, max_dim=1000, resize=False, draw_outline=False):
    # Convert to binary mask
    mask = mask.astype(np.uint8) * 255

    # Detect all contours above the threshold
    with redirect_stdout(StringIO()):
        annotations = detect_contour(mask, min_area=min_area, stats_mode="circle")
    contour_coords = annotations["contour"]["a"]["data"]["contour"]
    contour_info = annotations["contour"]["a"]["data"]["support"]

    # Create a copy of the original image to draw contours on
    image_with_contours = copy.deepcopy(image)

    # Create an overlay for the contours
    overlay = image_with_contours.copy()

    # Define a list of colors to use for drawing contours
    colors = [(255,0,255), (0, 255, 0), (0,255,255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

    # Draw all contours on the overlay
    for idx, coords in enumerate(contour_coords):
        color = colors[idx % len(colors)]
        cv2.drawContours(overlay, [coords], -1, color, cv2.FILLED)

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, image_with_contours, 1 - alpha, 0, image_with_contours)
    
    ## draw outline
    if draw_outline: 
        for idx, coords in enumerate(contour_coords):
            color = colors[idx % len(colors)]
            cv2.drawContours(overlay, [coords], -1, color, thickness=1)
        
    # Resize the image if necessary
    if resize:
        height, width, channels = image_with_contours.shape
        if max(height, width) > max_dim:
            if height > width:
                new_height = max_dim
                new_width = int((max_dim / height) * width)
            else:
                new_width = max_dim
                new_height = int((max_dim / width) * height)
            resized_image = cv2.resize(image_with_contours, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized_image
    return image_with_contours
