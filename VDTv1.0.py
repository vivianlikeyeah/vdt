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


types = {'png': 'png', 'tiff': 'tif', 'jpeg': 'jpg'}

#File Input 
def input(filename):
	print ("Welcome to VDT: Vivian's Denoising Thing")
	print("You would like to denoise " + filename)	
	
	here = os.path.dirname(os.path.abspath(__file__))
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




#Denoising 
def denoise (inputFile):

	print("Starting denoising")

	start = time.time()

	F = ptv.tv1_2d(inputFile, args.lamb)

	end = time.time()

	print('Time elapsed ' + str(end-start))
	return F 

def denoiseManual(inputFile,lamb):

	start = time.time()

	F = ptv.tv1_2d(inputFile, lamb)

	end = time.time()

	print('Time elapsed ' + str(end-start))
	return F 


def showFile(file):

	print("Show File Function Called")
	#Might be good to include the original as well!
	print("Showing original file")
	plt.imshow(file)
	io.show()

	print("Showing weird figure")

	fig=plt.figure(figsize=(14, 4))
	columns = 3
	rows = 1
	for i in range(1, columns*rows +1):
		fig.add_subplot(rows, columns, i)
		io.imshow(file)
	plt.show()

def denoiseManualIteration(inputFile,lamb):

	#Change this so only one question is asked - makes it easier. 

	#Maybe we could display on a scale? 
	userIsNotHappy = True 

	print("Before denoise manual")
	image = inputFile
	showFile(inputFile)

	while userIsNotHappy:

		showFile(image)

		image = denoiseManual(inputFile, lamb)
		showFile(image)
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

	io.imshow(outputFile)
	plt.show()

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		out = img_as_uint(outputFile)
	
	outname = outfileName()



	io.imsave(str(outname), out)
	#img = Image.fromarray(outputFile)
	#img.save("save",getFileType(sys.argv[1]))



def outfileName():

	#here = os.path.dirname(os.path.abspath(__file__))
	#filepath = here + "/" + sys.argv[1]
	#filetype = imghdr.what(filepath)
	#print(filetype)

	filetype = getFileType(args.file)

	fileExtension = filetypeConversions(filetype)
	outfileName = "outtathisworld." + fileExtension

	#We have to figure out how to do file type to extension
	#Convrsions - for example - filetype tiff = tif 

	#print outfileName
	return outfileName 


def filetypeConversions(filetype):

	#Will have to increase this and essentially do a case by case 
	#basis 

	#We can probably remove these elifs now
	#if filetype == "tiff":
	#	extension = "tif"
	#elif filetype == "png":
	#	extension = "png"
	#else:
	#	pass
	#	print ("Incompatible filetype")

	#print("Hello")
	#print(types[filetype])
	#If they specify a file type, return that - otherwise do the given file type
	if args.outfiletype: 
		return types[args.outfiletype]
	else: 
		return types[filetype]


def getFileType (filename):

	here = os.path.dirname(os.path.abspath(__file__))
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

	parser = argparse.ArgumentParser()
	parser.add_argument("--file", help = "The file you want to denoise")
	parser.add_argument("--lamb", help ="The lambda value for denoising", default = 0, type = float)
	parser.add_argument('--outfiletype', help = "The output file type", type = somefunction)

	args = parser.parse_args()



	#Due to argparse we should be able to remove this try.
	try: 
		newfile = input(args.file)
		#exit(1)
		#newfile = input(sys.argv[1])

	except IndexError:
		print("You need to give us a file mate, otherwise how will I know what to denoise?")
		print("Soon we will add the ability to use one of our test files as default!")
		exit(1)

	#io.imshow(newfile)
	#plt.show()
	image = Image.open(args.file)
	image.show()

	#So it is image.show option 

	print("HellloooOoo")


	print(newfile)
	
	#showFile(newfile)

	#outputFile = denoiseManualIteration(newfile,0)

	io.imshow(newfile)

	io.show()

	#outputFile = denoise(newfile)
	#print(outputFile)
	#output(outputFile)



#this link 
#https://github.com/InsightSoftwareConsortium/DCMTK
