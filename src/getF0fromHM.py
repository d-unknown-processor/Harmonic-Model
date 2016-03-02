#! /usr/bin/python
"""
written by: Meysam Asgari 15/12/2013 ; CSLU / OHSU
 """
# Usage (one cpu): python src/getF0fromHM.py -a wav_list.txt
import numpy as np , pdb , sys , time
from optparse import OptionParser
from featExt import *


### supporting features: 'voiced_labels' , 'f0' , 'voicingProb' 
fnames = Parser(OptionParser())
#feats = ['voiced_labels','f0' , 'voicingProb']
feats = ['voiced_labels']

for fileName in fnames:
    output = featureExtractor(fileName , *feats )
    outroot = fileName.split('.')[0]+'.txt'
    np.savetxt(outroot,output,fmt="%0.1f")
