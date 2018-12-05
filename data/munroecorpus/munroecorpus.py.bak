"""
Munroe Corpus Interface
Original (http://blog.xkcd.com/2010/05/03/color-survey-results/) was hand curated according to
McMahan & Stone. A Bayesian Model of Grounded Color Semantics. TACL. 2015.

All filenames returned are full path

Functions available:
get_training_handles(): returns [{colorname:filepath_hue,...},...] where there is a dict for each dim (H,S,V)
get_training_filename(colorname,dim): input dim \in {0,1,2}; returns training filepath

get_dev_handles(): returns {colorname:filepath,...}
get_dev_filename(colorname): returns filepath

get_test_handles(): returns {colorname:filepath,...}
get_test_filename(colorname): returns filepath

open_datafile(filename): returns formated (scaled to dimension) data in a list. for testing, list of 3-length arrays (hsv). else, list of whichever dimension was requested. 

"""

import os
def get_training_handles():
	"""
	Returns Training handles as a three-element array of dictionaries of form {colorname:filepath}.
	Three elements refer to H S and V respectively
	"""
	curdir = os.path.dirname(os.path.abspath(__file__))
	fp = open(curdir+"/corpusindex.txt"); c_index = {x.replace("\n","").split(",")[0]:x.replace("\n","").split(",")[1] for x in fp.readlines()}; fp.close()
        exts = ("h_train","s_train","v_train")
	return [{name:curdir+"/train/%s.%s"%(fn,ext) for name,fn in c_index.items()} for ext in exts]
	
def get_training_filename(name,dim=0):
	"""
	Returns dev filename for given colorname
	"""
	return get_training_handles()[dim][name]


def get_dev_handles():
	"""
	Returns Dev handles as a dictionary of form {colorname:filepath}.
	"""
	curdir = os.path.dirname(os.path.abspath(__file__))
	fp = open(curdir+"/corpusindex.txt"); c_index = {x.replace("\n","").split(",")[0]:x.replace("\n","").split(",")[1] for x in fp.readlines()}; fp.close()
	return {name:curdir+"/5dev/%s.test"%(fn) for name,fn in c_index.items()} 	

def get_dev_filename(name):
	"""
	Returns dev filename for given colorname
	"""
	return get_dev_handles()[name]


def get_test_handles():
	"""
	Returns test handles as a dictionary of form {colorname:filepath}.
	"""
	curdir = os.path.dirname(os.path.abspath(__file__))
	fp = open(curdir+"/corpusindex.txt"); c_index = {x.replace("\n","").split(",")[0]:x.replace("\n","").split(",")[1] for x in fp.readlines()}; fp.close()
	return {name:curdir+"/25test/%s.test"%(fn) for name,fn in c_index.items()} 	

def get_test_filename(name):
	"""
	Returns test filename for given colorname
	"""
	return get_test_handles()[name]


def open_datafile(filename):
	"""
	Open train or test file, process the file into [float,float,...] for trains
	or [[float,float,float],...] for test
	"""
	ext=filename.split('/')[-1].split('.')[-1] 
	fp = open(filename); lines = fp.readlines(); fp.close()
	if ext=='test':
		data = [[float(y)*[360,100,100][i] for i,y in enumerate(x.replace('\n','').split(','))] for x in lines]
	else:
		data = [float(x.replace('\n',''))*{'h':360,'s':100,'v':100}[ext[0]] for x in lines]
	return data
