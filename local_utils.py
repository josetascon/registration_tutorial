# -*- coding: utf-8 -*-
# @Author: jose
# @Date:   2018-12-06 11:31:42
# @Last Modified by:   jose
# @Last Modified time: 2019-01-03 15:38:02

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import clear_output

def image_info( image ):
    # Return a string with image information details. Pixel type, dimensions, scale.
    # Input: sitk.Image
    # Output: string
    info = '\n===== Image Information ====='
    info += '\nPixel type: \t\t' + str(image.GetPixelIDTypeAsString())
    info += '\nPixel channels: \t' + str(image.GetNumberOfComponentsPerPixel())
    info += '\nDimensions: \t\t' + str(image.GetDimension())
    info += '\nSize: \t\t\t' + str(image.GetSize())
    info += '\nLength (mm): \t\t' + str(image.GetSpacing())
    info += '\nTotal Elements: \t' + str(image.GetNumberOfPixels())
    info += '\n'
    return info

def imshow_2d(image_itk, title = '', show = True, axis = False):
    # Function to show a 2D image with matplotlib
    # Inputs: sitk.Image, string with title
    # Output: None
    channels = image_itk.GetNumberOfComponentsPerPixel() # get the number of channels
    image_npa = sitk.GetArrayFromImage(image_itk) # get a copy as numpy array with the image data
    if channels == 1: # set the color map according to the channels
        plt.imshow(image_npa, cmap = plt.cm.gray) # 1 channel for monochrome image
    else:
        plt.imshow(image_npa) 	# default color map of pyplot
    plt.title(title)
    if not axis: plt.axis('off') # disable to see axis
    if show: plt.show()			 # stop show, useful when subplotting
    return

def imshow_axial(range_z, image_itk):
    # Function to show the slices in the axial view interactively
    spacing = image_itk.GetSpacing() # scale in mm
    size = image_itk.GetSize()       # pixel width and height
    # Scale the image with the pixel size and spacing in mm
    extent = (0, np.ceil(spacing[0]*size[0]), np.ceil(spacing[1]*size[1]), 0) #image limits
    
    image_npa = sitk.GetArrayViewFromImage(image_itk) # get numpy array

    # if horizontal and vertical and reverse:
    #     image = image_npa[-range_z,::-1,::-1]
    # elif horizontal and vertical and not reverse:
    #     image = image_npa[range_z,::-1,::-1]
    # elif horizontal and not vertical and reverse:
    #     image = image_npa[-range_z,:,::-1]
    # elif horizontal and not vertical and not reverse:
    #     image = image_npa[range_z,:,::-1]
    # elif not horizontal and vertical and reverse:
    #     image = image_npa[-range_z,::-1,:]
    # elif not horizontal and vertical and not reverse:
    #     image = image_npa[range_z,::-1,:]
    # elif not horizontal and not vertical and reverse:
    #     image = image_npa[-range_z,:,:]
    # else:
    #     image = image_npa[range_z,:,:]
    
    # Create a figure with the axial
    plt.imshow(image_npa[range_z,:,:], extent=extent, cmap=plt.cm.gray) 
    plt.title('Axial image')
    plt.axis('off')
    plt.show() # Draw the axial image
    return

def imshow_sagital(range_y, image_itk):#, image_itk):
    # Function to show the slices in the sagital view interactively
    spacing = image_itk.GetSpacing() # scale in mm
    size = image_itk.GetSize()       # pixel width and height
    # Scale the image with the pixel size and spacing in mm
    extent = (0, np.ceil(spacing[1]*size[1]), np.ceil(spacing[2]*size[2]), 0) #image limits
    
    image_npa = sitk.GetArrayViewFromImage(image_itk) # get numpy array
    
    # Create a figure with the sagital
    plt.imshow(image_npa[::-1,:,range_y], extent=extent, cmap=plt.cm.gray)
    plt.title('Sagital image')
    plt.axis('off')
    plt.show() # Draw the sagital image
    return

def imshow_coronal(range_x, image_itk):#, image_itk):
    # Function to show the slices in the coronal view interactively
    spacing = image_itk.GetSpacing() # scale in mm
    size = image_itk.GetSize()       # pixel width and height
    # Scale the image with the pixel size and spacing in mm
    extent = (0, np.ceil(spacing[0]*size[0]), np.ceil(spacing[2]*size[2]), 0) # image limits
    
    image_npa = sitk.GetArrayViewFromImage(image_itk) # get numpy array
    
    # Create a figure with the coronal
    plt.imshow(image_npa[::-1,range_x,:], extent=extent, cmap=plt.cm.gray)
    plt.title('Sagital image')
    plt.axis('off')
    plt.show()
    return

def get_name_transform( str_transform ):
    # Return class name of transformation using its string object
    # Input: string (transform.ToString())
    # Output: string
    name = str_transform.split()
    return name[1] # transform class name always in second line

def transform_info( transform ):
    # Return a string with information about the transformation.
    # Input: sitk.Transform
    # Output: string
    info = '\n===== Transform Info ====='
    # info += '\nTransform type: \t' + str(transform.GetName())
    info += '\nTransform type: \t' + get_name_transform(str(transform))
    info += '\nDimensions: \t\t' + str(transform.GetDimension())
    info += '\nParameters: \t\t' + str(transform.GetParameters())
	# info += '\nMatrix: \t\t' + str(transform.GetMatrix())
    info += '\n'
    return info

def transform_point(point, transform, verbose = True):
    # Apply a transformation to a point
    # Inputs: tuple, sitk.Transform
    # Output: tuple
    transformed_point = transform.TransformPoint(point)
    if verbose:
        info = '\n===== Transform Point ====='
        info += '\nPoint: \t\t' + str(point)
        info += '\nTransformed: \t' + str(transformed_point)
        info += '\n'
        print(info)
    return transformed_point

def transform_image(image, transform, interpolator = sitk.sitkLinear, default_value = 0.0):
    # Apply a transform to an image
    # Inputs: sitk.Image, sitk.Transform, sitk.InterpolatorEnum, double
    # Output: sitk.Image
	reference_image = image
	interpolator = interpolator
	return sitk.Resample(image, reference_image, transform, interpolator, default_value)

def displacement_vectors(size, transform, samples = 30):
    # Create the coordinates
    numSamplesX = samples
    numSamplesY = samples                 
    coordsX = np.linspace(0, size[0]-1, numSamplesX)
    coordsY = np.linspace(0, size[1]-1, numSamplesY)
    XX, YY = np.meshgrid(coordsX, coordsY)

    # Transform points and compute the vectors.
    vectorsX = np.zeros(XX.shape)
    vectorsY = np.zeros(YY.shape)
    for index, value in np.ndenumerate(XX):
        px,py = transform.TransformPoint((value, YY[index]))
        vectorsX[index] = px - value
        vectorsY[index] = py - YY[index]
        
    return XX, YY, vectorsX, vectorsY

def plot_registration(fixed_image, moving_image, transform, samples = 30):
    
    registered_image = transform_image(moving_image, transform)
    
    # Grid generation
    grid = sitk.GridSource(outputPixelType=sitk.sitkFloat32, size=moving_image.GetSize(), 
                           sigma=(0.2, 0.2), gridSpacing=(3, 3), spacing=(0.2,0.2))
    grid.SetSpacing((1.0,1.0))

    registered_grid = transform_image(grid, transform)
    
    XX, YY, vectorsX, vectorsY = displacement_vectors(moving_image.GetSize(), 
                                                      transform, samples = samples)
    
    # Plot all the images
    f, axs = plt.subplots(2,3,figsize=(15,9))
    plt.subplot(231)
    imshow_2d(moving_image, 'Moving Image', show=False)
    plt.subplot(232)
    imshow_2d(fixed_image, 'Fixed Image', show=False)
    plt.subplot(233)
    imshow_2d(registered_image, 'Registered Image', show=False)
    
    ax = plt.subplot(234)
    plt.quiver(XX, YY, vectorsX, vectorsY)
#     plt.xlim(0,fixed_image.GetWidth())
#     plt.ylim(0,fixed_image.GetHeight())
    plt.title('Displacement Vectors')
    plt.axis('off')
    ax.set_aspect(1.0)
    
    plt.subplot(235)
    imshow_2d(fixed_image-registered_image, 'Difference Image', show=False)
    plt.subplot(236)
    imshow_2d(registered_grid, 'Grid Warp', show=False)
    plt.show()

# Callback invoked when the StartEvent happens, sets up our new data.
def start_register_plot():
    global metric_values, multires_iterations
    
    metric_values = []
    multires_iterations = []

# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_register_plot():
    global metric_values, multires_iterations
    
    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()

# Callback invoked when the IterationEvent happens, update our data and display new figure.    
def plot_register_values(registration_method):
    global metric_values, multires_iterations
    
    metric_values.append(registration_method.GetMetricValue())                                       
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    plt.show()
    
# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the 
# metric_values list. 
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))