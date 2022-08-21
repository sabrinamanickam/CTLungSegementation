# common packages 
import numpy as np 
import os
import copy
from math import *
import matplotlib.pyplot as plt
from functools import reduce
# reading in dicom files
import pydicom
# skimage image processing packages
from skimage import measure, morphology
from skimage.morphology import ball, binary_closing
from skimage.measure import label, regionprops
# scipy linear algebra functions 
from scipy.linalg import norm
import scipy.ndimage
# ipywidgets for some interactive plots
from ipywidgets.widgets import * 
import ipywidgets as widgets
# plotly 3D interactive graphs 
import plotly
from plotly.graph_objs import *
import chart_studio.plotly as py
from scipy import ndimage, misc
import cv2
from skimage import data
from skimage.exposure import histogram
from skimage import feature

from skimage import data
from skimage.exposure import histogram
from skimage.feature import canny
from skimage.filters import sobel
from skimage.segmentation import watershed
from scipy import ndimage as ndi
"""LOADING DICOM DATA"""
def load_scan(path):
    slices = [pydicom.dcmread(path+'/' + s) for s in               
              os.listdir(path)]
    slices = [s for s in slices if 'SliceLocation' in s]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
       slice_thickness = np.abs(slices[0].ImagePositionPatient[2]-slices[1].ImagePositionPatient[2])
    except:
       slice_thickness = np.abs(slices[0].SliceLocation-slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices
def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

# set path and load files 
path = r"C:\Users\user\OneDrive\Desktop\data"
patient_dicom = load_scan(path)
patient_pixels = get_pixels_hu(patient_dicom)
#sanity check
plt.imshow(patient_pixels[90], cmap=plt.cm.bone)

"""IMAGE PROCESSING"""


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None
    
    
    
    
def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image >= -700, dtype=np.int8)+1
    labels = measure.label(binary_image)
 
    # Pick the pixel in the very corner to determine which label is air.
    # Improvement: Pick multiple background labels from around the patient
    # More resistant to “trays” on which the patient lays cutting the air around the person in half
    background_label = labels[0,0,0]
 
    # Fill the air around the person
    binary_image[background_label == labels] = 2
 
    # Method of filling the lung structures (that is superior to 
    # something like morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice-1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
 
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
 
    # Remove other air pockets inside body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image
# get masks 
segmented_lungs = segment_lung_mask(patient_pixels,    
                  fill_lung_structures=False)
segmented_lungs_fill = segment_lung_mask(patient_pixels,     
                       fill_lung_structures=True)
internal_structures = segmented_lungs_fill - segmented_lungs
# isolate lung from chest
copied_pixels = copy.deepcopy(patient_pixels)
for i, mask in enumerate(segmented_lungs_fill): 
    get_high_vals = mask == 0
    copied_pixels[i][get_high_vals] = 0
seg_lung_pixels = copied_pixels
# sanity check
plt.imshow(seg_lung_pixels[7], cmap=plt.cm.bone)

    
slice_id = 90
plt.figure(1)
plt.title('Original Dicom')
plt.imshow(patient_pixels[slice_id], cmap=plt.cm.bone)
plt.figure(2)
plt.title('Lung Mask')
plt.imshow(segmented_lungs_fill[slice_id], cmap=plt.cm.bone)
plt.figure(3)
plt.title('Parenchyma Mask on Segmented Lung')
plt.imshow(seg_lung_pixels[slice_id], cmap=plt.cm.bone)
plt.imshow(internal_structures[slice_id], cmap='jet', alpha=0.7)
plt.figure(4)

# slide through dicom images using a slide bar 
plt.figure(1)
def dicom_animation(x):
    plt.imshow(patient_pixels[x])
    return x
interact(dicom_animation, x=(90, len(patient_pixels)-1))

"""FILTERING"""
"""MEDIAN"""
fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side
ascent = patient_pixels[90]
result = ndimage.median_filter(ascent, size=20)
ax1.imshow(ascent)
ax2.imshow(result)
plt.show()

"""GAUSSIAN"""
fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side
ascent = patient_pixels[90]
result = ndimage.gaussian_filter(ascent,sigma=5)
ax1.imshow(ascent)
ax2.imshow(result)
plt.show()

"""minimum"""
fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side
ascent = patient_pixels[90]
result = ndimage.minimum_filter(ascent,size=20)
ax1.imshow(ascent)
ax2.imshow(result)
plt.show()

"""MAXIMUM"""
fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side
ascent = patient_pixels[90]
result = ndimage.maximum_filter(ascent,size=20)
ax1.imshow(ascent)
ax2.imshow(result)
plt.show()

"""SOBEL"""
fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side
ascent = patient_pixels[90]
result = ndimage.sobel(ascent)
ax1.imshow(ascent)
ax2.imshow(result)
plt.show()

"""PREWITT"""
fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side
ascent = patient_pixels[90]
result = ndimage.prewitt(ascent)
ax1.imshow(ascent)
ax2.imshow(result)
plt.show()



"""HISTOGRAM"""

histogram, bin_edges = np.histogram(patient_pixels[53], bins=512, range=(0,121))
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.plot(bin_edges[0:-1], histogram)  # <- or here
plt.show()



histogram, bin_edges = np.histogram(patient_pixels[18], bins=512, range=(0,121))
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.plot(bin_edges[0:-1], histogram)  # <- or here
plt.show()

"""EDGE BASED SEGMENTATION"""
lung= patient_pixels[90]
hist, hist_centers = histogram(lung)
edges = canny(lung/512.)
plt.imshow(edges,cmap=plt.cm.bone)
fill_lung = ndi.binary_fill_holes(edges)
label_objects, nb_labels = ndi.label(fill_lung)
sizes = np.bincount(label_objects.ravel())
mask_sizes = sizes > 20
mask_sizes[0] = 0
lung_cleaned = mask_sizes[label_objects]
plt.imshow(lung_cleaned,cmap=plt.cm.bone)

"""Region-based segmentation"""
markers = np.zeros_like(lung)
markers[lung < 50] = 1
markers[lung> 500] = 2

elevation_map = sobel(lung)
plt.imshow(elevation_map,cmap=plt.cm.bone)
markers = np.zeros_like(lung)
markers[lung < 30] = 1
markers[lung > 150] = 2
plt.imshow(markers,cmap=plt.cm.bone)

"""WATERSHED TRANSFORM"""

segmentation = watershed(elevation_map, markers)
plt.imshow(segmentation,cmap=plt.cm.bone)

segmentation = ndi.binary_fill_holes(segmentation - 1)
labeled_lung, _ = ndi.label(segmentation)
plt.imshow(labeled_lung,cmap=plt.cm.bone)




