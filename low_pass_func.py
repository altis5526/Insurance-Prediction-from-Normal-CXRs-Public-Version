import cv2
import numpy as np
from scipy import fftpack
from PIL import Image, ImageDraw

def low_pass(img, diameter=50):
    img = img[:,:,0] # gray-scale image
    #fft of image
    fft1 = fftpack.fftshift(fftpack.fft2(img))

    #Create a low pass filter image
    x,y = img.shape[0],img.shape[1]
    #size of circle
    e_x,e_y=diameter,diameter
    #create a box 
    bbox=((x/2)-(e_x/2),(y/2)-(e_y/2),(x/2)+(e_x/2),(y/2)+(e_y/2))

    low_pass=Image.new("L",(img.shape[0],img.shape[1]),color=0)

    draw1=ImageDraw.Draw(low_pass)
    draw1.ellipse(bbox, fill=1)

    low_pass_np=np.array(low_pass)

    #multiply both the images
    filtered=np.multiply(fft1,low_pass_np)

    #inverse fft
    ifft2 = np.real(fftpack.ifft2(fftpack.ifftshift(filtered)))
    ifft2 = np.maximum(0, np.minimum(ifft2, 255))
    ifft2 = np.tile(ifft2, (3,1,1))
    
    return ifft2

def high_pass(img, diameter=50):
    img = img[:,:,0] # gray-scale image
    #fft of image
    fft1 = fftpack.fftshift(fftpack.fft2(img))

    #Create a low pass filter image
    x,y = img.shape[0],img.shape[1]
    #size of circle
    e_x,e_y=diameter,diameter
    #create a box 
    bbox=((x/2)-(e_x/2),(y/2)-(e_y/2),(x/2)+(e_x/2),(y/2)+(e_y/2))

    high_pass=Image.new("L",(img.shape[0],img.shape[1]),color=1)

    draw1=ImageDraw.Draw(high_pass)
    draw1.ellipse(bbox, fill=0)

    high_pass_np=np.array(high_pass)

    #multiply both the images
    filtered=np.multiply(fft1,high_pass_np)

    #inverse fft
    ifft2 = np.real(fftpack.ifft2(fftpack.ifftshift(filtered)))
    ifft2 = np.maximum(0, np.minimum(ifft2, 255))
    ifft2 = np.tile(ifft2, (3,1,1))
    
    return ifft2