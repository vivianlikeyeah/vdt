#This is the beginning of the VDT written by Vivian Bakiris 

import sys
import os
import skimage as ski
from skimage import io, color, util 
import imghdr
from matplotlib import pyplot as plt
import time 
import prox_tv as ptv
from skimage import img_as_uint
import warnings
from PIL import Image
import argparse
import cv2
import SimpleITK as sitk
import sys
import os
from math import pi
from numpy import diff

import glob

import numpy as np
import matplotlib.pyplot as plt

import re 
metrics = list() 
# A list of all the metric values obtained from the image registration 
images = list()
# A list of the names of all the images developed in the denoising process 


types = {'png': 'png', 'tiff': 'tif', 'jpeg': 'jpg'}

#File Input 
def input(filename):
	print ("Welcome to VDT: Vivian's Denoising Thing")
	print("You would like to denoise " + filename)	
	
	here = os.path.dirname(os.path.abspath(__file__))

	if args.folder: 
		here = here + "/" + args.folder


	filepath = here + "/" + filename 

	filetype = imghdr.what(filepath)
	X = io.imread(filepath)

	#Check compatibility 
	#This should be expanded to different file types! 
	#Also we should do tests that can confirm this works correctly

	if filetype in types:
		print("Your file type is compatible")
	else:
		print("ERRor! EXitINg! bAD HUmaN")

	X = ski.img_as_float(X)
	#X = color.rgb2gray(X)

	return X 

def command_iteration(method) :
    #if (method.GetOptimizerIteration()==0):
    #    print("Scales: ", method.GetOptimizerScales())
    #print("{0:3} = {1:7.5f} : {2}".format(method.GetOptimizerIteration(),
                                           #method.GetMetricValue(),
                                           #method.GetOptimizerPosition()))
	pass


def register(moving, fixed): 

	start = time.time()




	here = os.path.dirname(os.path.abspath(__file__))
	filepath = here + "/" + str(fixed)
	fixed = sitk.ReadImage(filepath, sitk.sitkFloat32)

	print(filepath)


	filepath2 = here + "/" + str(moving)
	moving = sitk.ReadImage(filepath2, sitk.sitkFloat32)
	print(filepath2)


	#fixed = fixed
	R = sitk.ImageRegistrationMethod()
	R.SetMetricAsMattesMutualInformation(numberOfHistogramBins = 50)
	
	sample_per_axis=12
	if fixed.GetDimension() == 2:
		tx = sitk.Euler2DTransform()
		# Set the number of samples (radius) in each dimension, with a
		# default step size of 1.0
		R.SetOptimizerAsExhaustive([sample_per_axis//2,0,0])
		# Utilize the scale to set the step size for each dimension
		R.SetOptimizerScales([2.0*pi/sample_per_axis, 1.0,1.0])
	elif fixed.GetDimension() == 3:
		tx = sitk.Euler3DTransform()
		R.SetOptimizerAsExhaustive([sample_per_axis//2,sample_per_axis//2,sample_per_axis//4,0,0,0])
		R.SetOptimizerScales([2.0*pi/sample_per_axis,2.0*pi/sample_per_axis,2.0*pi/sample_per_axis,1.0,1.0,1.0])

	# Initialize the transform with a translation and the center o
	# rotation from the moments of intensity

	tx = sitk.CenteredTransformInitializer(moving, fixed, tx)
	R.SetInitialTransform(tx)
	R.SetInterpolator(sitk.sitkLinear)
	#R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )
	outTx = R.Execute(moving,fixed)
	#print("-------")
	#print(outTx)
	#print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
	#print(" Iteration: {0}".format(R.GetOptimizerIteration()))
	#print(" Metric value: {0}".format(R.GetMetricValue()))
	metrics.append(abs(R.GetMetricValue()))
	print(abs(R.GetMetricValue()))

	end = time.time()

	print('Time to Register ' + str(end-start))


#Denoising 
def denoise (inputFile):

	print("Starting denoising")

	start = time.time()

	F = ptv.tv1_2d(inputFile, args.lamb,1,3)

	end = time.time()

	print('Time to denoise ' + str(end-start))
	return F 

def denoiseManual(inputFile,lamb):

	start = time.time()

	F = ptv.tv1_2d(inputFile, lamb,1 ,3)

	end = time.time()

	print('Time elapsed ' + str(end-start))

	return F 

def denoiseAutomated(inputFile):
	print("Start Automated Denoising")

	fileToDenoise = inputFile
	previousFile = inputFile
	original = "hello"

	for x in np.arange(0,0.12,0.005):

		print x

		step = denoiseManual(fileToDenoise,x)
		name = "Oddstep" + str(x) + ".png"
		#showFile(step)
		io.imsave(str(name), step)

		#here = os.path.dirname(os.path.abspath(__file__))
		#filepath = here + "/" + str(name)
		#X = io.imread(filepath)
		#X = ski.img_as_float(X)
		
		#fileToDenoise = X
		#fileToDenoise = step

		images.append(name)
		if x == 0:
			original = name 

		register(name, original)


	#calculateBest(metrics)
		#print x

def denoiseAutomatedPairWise(inputFile):
	print("Start Automated Denoising")

	fileToDenoise = inputFile
	increment = 0.005
	prevx = 0.0 

	for x in np.arange(0,0.12,increment):

		print("The x value is")
		
		print x

		step = denoiseManual(fileToDenoise,x)
		current = "Oddstep" + str(x) + ".png"



		previous = "Oddstep" + str(prevx) + ".png"



		#showFile(step)
		print("Saving the file name", str(current))
		io.imsave(str(current), step)
		images.append(current)

		register(current, previous)
		prevx = x 
		#print x

def calculateBest():


	dif=np.diff(metrics)
	dif2=np.diff(dif)
	#plt.plot(dif)
	#plt.plot(dif2)
	#plt.show()
	
	fn=True
	for k,i in enumerate(dif2):
		if i<0:
			if not fn:
				print(k+1)
				if ((k+1) > 5):
					break

			fn=False

	print (k+1)

	return (k+1)



def showFile(file):

	print("Show file function!")

	#io.imshow(file)
	#plt.show()

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		out = img_as_uint(file)
	
	if args.file: 
	
		outname = outfileName(args.file)
	else: 
		outname = "hello.png"

	#You need to save the image you want to see 

	io.imsave(str(outname), out)
	

	name = "showFile" #+ str(number)
	img = cv2.imread(str(outname),0)
	imS = cv2.resize(img, (700, 760))  
	cv2.namedWindow(name, cv2.WINDOW_NORMAL)
	cv2.moveWindow(name, 05,20);
	cv2.imshow(name,imS)

	cv2.waitKey(0) #closes after a key press 
	cv2.destroyAllWindows()
	'''
	imgOriginal = cv2.imread(args.file,0)
	imOriginal = cv2.resize(imgOriginal,(700,760))
	cv2.namedWindow('original image', cv2.WINDOW_NORMAL)
	cv2.moveWindow('original image', 710,20)
	cv2.imshow('original image', imOriginal)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''

	#image = Image.open("tempfile.png")
	#image.show()


	#fig=plt.figure(figsize=(14, 4))
	#columns = 3
	#rows = 1
	#for i in range(1, columns*rows +1):
	#	fig.add_subplot(rows, columns, i)
	#	io.imshow(file)
	#plt.show()


def openFile(filename):

	print("Open file function!")
	

	name = "Denoised File" #+ str(number)
	img = cv2.imread(filename,0)
	imS = cv2.resize(img, (700, 760))  
	cv2.namedWindow(name, cv2.WINDOW_NORMAL)
	cv2.moveWindow(name, 05,20);
	cv2.imshow(name,imS)
	#cv2.waitKey(0) #closes after a key press 
	#cv2.destroyAllWindows()

	imgOriginal = cv2.imread(args.file,0)
	imOriginal = cv2.resize(imgOriginal,(700,760))
	cv2.namedWindow('original image', cv2.WINDOW_NORMAL)
	cv2.moveWindow('original image', 710,20)
	cv2.imshow('original image', imOriginal)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



def denoiseManualIteration(inputFile,lamb):

	#Change this so only one question is asked - makes it easier. 

	#Maybe we could display on a scale? 
	userIsNotHappy = True 

	print("Before denoise manual")
	
	image = inputFile
	#showFile(inputFile)

	while userIsNotHappy:

		image = denoiseManual(inputFile, lamb)
		#showFile(image)
		#showFile(inputFile)
		happiness = raw_input("Are you happy though? ")

		if happiness == "Yes":
			userIsNotHappy = False
		else:
			userIsNotHappy = True
			moreOrLess = raw_input("Would you like to denoise more or less? ")
			if moreOrLess == "more":
				lamb = lamb + 0.01
			else:
				lamb = lamb - 0.01
				if lamb < 0:
					lamb = 0
					print("Minimum lambda value reached")
			

			print(lamb)


	return image

#File Output
def output(outputFile):

	#io.imshow(outputFile)
	#plt.show()



	#showFile(outputFile)


	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		out = img_as_uint(outputFile)
	
	outname = outfileName()

	io.imsave(str(outname), out)

	#image = Image.open(str(outname))
	#image.show()

	# Load an color image in grayscale
	#img = cv2.imread(str(outname),0)
	#imS = cv2.resize(img, (700, 760))  
	#cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	#cv2.moveWindow("image", 50,50);
	#cv2.imshow('image',imS)
	#cv2.waitKey(0) #closes after a key press 
	#cv2.destroyAllWindows()


	#img = Image.fromarray(outputFile)
	#img.save("save",getFileType(sys.argv[1]))



def outfileName(file):


	filetype = getFileType(file)

	fileExtension = filetypeConversions(filetype)
	outfileName = "outtathisworld." + fileExtension

	return outfileName 


def filetypeConversions(filetype):

	if args.outfiletype: 
		return types[args.outfiletype]
	else: 
		return types[filetype]


def getFileType (filename):

	here = os.path.dirname(os.path.abspath(__file__))

	if args.folder:
		#here = os.path.dirname(os.path.abspath(__file__))
		print (args.folder)
		print(type(args.folder))
		print(here)
		print(type(here))
		print(filename)
		print(type(filename))
		filepath = here + "/" + args.folder + "/" + filename
	else: 
		filepath = here + "/" + filename
	

	filetype = imghdr.what(filepath)

	return filetype 


#Change this name
def somefunction(outfiletype):

	if outfiletype in types: 
		return outfiletype
	else:
		raise argparse.ArgumentTypeError("This output file type is not yet supported. Please check your spelling e.g. TIFF not TIF ")

#Run
if __name__ == "__main__":

	start = time.time()

	parser = argparse.ArgumentParser()
	parser.add_argument("--file", help = "The file you want to denoise")
	parser.add_argument("--folder", help = "The folder containing the files you want to denoise")
	parser.add_argument("--lamb", help ="The lambda value for denoising", default = 0.1, type = float)
	parser.add_argument('--outfiletype', help = "The output file type", type = somefunction)

	args = parser.parse_args()



	#Due to argparse we should be able to remove this try.
	if args.file: 
		try: 
			newfile = input(args.file)
			#exit(1)
			#newfile = input(sys.argv[1])

		except IndexError:
			print("You need to give us a file mate, otherwise how will I know what to denoise?")
			print("Soon we will add the ability to use one of our test files as default!")
			exit(1)


	#denoiseAutomatedPairWise(newfile)

	#fileIndex = calculateBest()

	#print(images[fileIndex])

	#openFile(images[fileIndex])



	#newblah = denoiseManual(newfile,0.1)

	#showFile(newblah)

	#end = time.time()

	#noisyness = list(np.arange(0,0.15,0.001))



	#print(metrics)
	#plt.plot(metrics,'ro')
	#plt.show()



	if args.folder: 
		import os
		arr = os.listdir('.')
		print(arr)

		filepath =  args.folder + '/'
		print("The filepath is", filepath)

		#You need to denoise the first file, 
		#and then use that as a standard. 
		#We need to check the first file though
		#Maybe denoise a file in the middle - it should
		#Theoretically have the most detail 
		#Get the most accurate first reading 


		for filename in os.listdir(filepath):
			print filename
			if filename.endswith(".png"):  
				newfile = input(filename)
				newDenoised = denoiseManual(newfile, 0.1)
				showFile(newDenoised)
	else:
		#newDenoised = denoiseAutomatedPairWise(newfile)
		#showFile(newDenoised)
		#print("Hello")
		denoiseAutomated(newfile)

		fileIndex = calculateBest()


		f = open('datafull.csv','a')

		data = images[fileIndex] + '\t' + args.file + '\n'
		f.write(data)
		f.close()

		print(images[fileIndex])

		#print (metrics)

		#plt.plot(metrics,'ro')
		#plt.show()

		#openFile(images[fileIndex])


	#Check one file 


	#Put all the files in one directory - Go through them. 
	#Upload all the files onto the github 
	#Create a form for Gavin and get him to tell you the best. 
	



	
	#print('TOTAL TIME ONE IMAGE ' + str(end-start))



#this link 
#https://github.com/InsightSoftwareConsortium/DCMTK
