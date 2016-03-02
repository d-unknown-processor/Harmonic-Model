"""
written by: Meysam Asgari 15/12/2013 ; CSLU / OHSU
"""
import numpy as np , pdb , wave
from lib import readSpeech , readWavFile
from HM import *

execfile('src/cfg.py')

def featureExtractor(fileName , *feats):
    print fileName
    if len(fileName.strip().split()) > 1:
        outRoot = 'samePath'
        fname , st , en = fileName.strip().split()
        dir , wav = os.path.split(fname)
        ifs = wave.open(fname).getframerate()
        start = int(ifs*float(st))
        end = int(ifs*float(en))
        if outRoot == 'samePath':
            root = dir+'/'+wav[:-4]+'_'+st+'_'+en
        else:
            root = ROOT + wav[:-4]+'_'+st+'_'+en
        try:
            nt, fs, sig = readWavFile(fname , start , end-start)
        except  Exception as Err:
            error = open(root + '.err_wav' , "w")
            error.write("%s\n" %(str(Err)))
            error.close()
            sys.exit('wave file has a error')
    else:
        sig , fs = readWave(fileName)
    sig = sig+eps

    LP_frames  = readSpeech(sig)
    frames = getframes1(sig)
    nframe,FL = frames.shape
    #outFeats = np.empty(())
    outFeats = np.zeros((nframe,1))
    harCof = None; vuv = None
    ## f0 and voicing probability and voiced labels   
    if 'f0' in feats and 'voicingProb' in feats:
        f0 , voicingProb , vuv = getf0Features ( LP_frames , fs, voiceProb = True , PitchDetction = True)
    elif 'f0' in feats:
        f0 , vuv = getf0Features ( LP_frames ,fs , PitchDetction = True)
    elif 'voicingProb' in feats:
        voicingProb , vuv = getf0Features ( LP_frames , fs  , voiceProb = True)
    elif 'voiced_labels' in feats and not 'harmCoeff' in feats:
        vuv = getf0Features ( LP_frames , fs)


    ## MAP estimation-- it now supports HM not TVHM!
    ## f0 and voicing probability and voiced labels
    if 'f0_map' in feats and 'voicingProb_map' in feats:
        f0_map , voicingProb_map , vuv_map = getf0_MAP ( LP_frames , fs, voiceProb = True , PitchDetction = True)
    elif 'f0_map' in feats:
        f0_map , vuv_map = getf0_MAP ( LP_frames ,fs , PitchDetction = True)
    elif 'voicingProb_map' in feats:
        voicingProb_map , vuv_map = getf0_MAP ( LP_frames , fs  , voiceProb = True)
    elif 'voiced_labels_map' in feats and not 'harmCoeff' in feats:
        vuv_map = getf0_MAP ( LP_frames , fs)

    ##############################################################    
    ################ Stacking the features ########################
    if 'voiced_labels' in feats:
        outFeats = np.c_[outFeats,vuv]
    if 'f0' in feats:
        outFeats = np.c_[outFeats,f0]
    if 'voicingProb' in feats:
        outFeats = np.c_[outFeats,voicingProb]

    return outFeats[:,1:]
                 
def Parser ( parser ):
    parser.add_option("-a", "--wList", dest="WavList", help="read .list text file")
    parser.add_option("-j", "--nJob", dest="numOfJob", help="read the job's number")
    parser.add_option("-n", "--nProc", dest="numOfProcessors", help="read the total num of processors")
    (options,args) = parser.parse_args()

    wavList = options.WavList
    nJob =  options.numOfJob
    nProc =  options.numOfProcessors
    if nJob is None: nJob = 0
    else: nJob = int ( nJob )
    if nProc is None: nProc = 1
    else: nProc = int( nProc )

    fip = open(wavList , "r")
    lines=fip.readlines()
    nFiles = len(lines)
    t1=np.arange(0,nFiles) % nProc
    list1 = np.nonzero(t1 == nJob)[0]
    fip.close()
    fnames=[]
    for line in range(len(list1)):
        fnames.append(lines[list1[line]].strip())
    return fnames
