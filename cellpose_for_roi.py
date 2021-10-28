from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np

from glob import glob
import tifffile

import tqdm
from datetime import datetime
import syglass as sy
from cellpose import models
import code
from skimage import color 
from skimage import segmentation
from skimage import exposure
import tkinter as tk
#1 GB == 1600 secs

def predict(vol, roi_number):
	print("in prediction")
	model = models.Cellpose(gpu=True, model_type='cyto')

	masks, flows, styles, diams = model.eval(vol, diameter=None, channels=[2,2], do_3D=True, batch_size=1)
	#code.interact(local=locals())
	masks16 = masks.astype(np.uint16)
	tifffile.imsave('mask16_roi' + str(roi_number) + ".tiff", masks16)
	mask16_extra = masks16[..., np.newaxis]

	return mask16_extra
	
def get_roi_number():
	root=tk.Tk()
	mystring = tk.StringVar()
	def getvalue():
		global returnString 
		returnString = mystring.get()
		root.destroy()
	tk.Label(root, text="ROI #").grid(row=0)  #label
	tk.Entry(root, textvariable = mystring).grid(row=0, column=1) #entry textbox
	tk.WSignUp = tk.Button(root, text="Extract", command=getvalue).grid(row=3, column=0) #button
	root.mainloop()

def main(args):
	startt = datetime.now()
	print("Cellpose for ROI Plugin, by Michael Morehead")
	print("Attempts to apply Cellpose to a specific region of interest")
	print("and inject integer masks (volumetric labels) into the ROI.")
	print("---------------------------------------")
	print("Usage: Highlight a project and use the Script Launcher in syGlass.")
	print("---------------------------------------")
	
	projectList = args["selected_projects"]

	doExtract = True
	if len(projectList) < 1:
		print("Highlight a project before running to select a project!")
		doExtract = False
	
	if len(projectList) > 1:
		print("This script only supports 1 project at a time, please select only one project before running.")
		doExtract = False


	if doExtract:
		get_roi_number()
		global returnString
		project = projectList[0]
		syGlassProjectPath = project.get_path_to_syg_file().string()
		print("Extracting ROI " + str(returnString) + " from: " + syGlassProjectPath)
		roi_block = project.get_roi_data(int(returnString))
		data = np.squeeze(roi_block.data[:,:,:,0])
		tifffile.imsave('data_orig.tiff', data)
		print(data.shape)
		result = predict(data, returnString)
		print(result.shape)
		tifffile.imsave('mask.tiff', result)
		project.import_mask(result, int(returnString))
		
	
	print("total time running entire pipeline\n")
	print(datetime.now() - startt)

