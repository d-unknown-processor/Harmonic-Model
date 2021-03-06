"""
written by: Meysam Asgari 15/12/2013 ; CSLU / OHSU
## common speech processing functions
"""
import numpy as np , wave , struct , pdb
from scipy.fftpack import rfft, irfft

# import matplotlib
# matplotlib.use("PDF")
# from matplotlib import mlab as mlab, pylab as plt
execfile('./src/cfg.py')

from scipy.signal import lfilter, firwin , medfilt

def readWave(wavName):
    from scipy.io.wavfile import read as wavread
    global fs
    fs, x = wavread(wavName)
    return x , fs

def readWavFile(fn,start,duration):
    global fs
    fp = wave.open(fn)
    nt = fp.getnframes()
    fs = fp.getframerate()
    x = readSamples(fp,start,duration)[0][0]
    fp.close()
    return nt, fs, x

def readSamples(fp, start,duration):
    fp.setpos(start)
    wav = fp.readframes(duration)
    sampwidth = fp.getsampwidth()
    fmt = wave._array_fmts[sampwidth]
    wav = struct.unpack('<%d%s' %(len(wav) / sampwidth, fmt), wav)
    nc = fp.getnchannels()
    if nc > 1:
        wavs = []
        for c in xrange(nc):
            wavs.append([wav[si] for si in xrange(c, len(wav), nc)])
    else:
        wavs = [[wav]]
    return wavs

def readSpeech(x):
    x = x.astype(np.float64) / (2 ** 15) # go to 64-bit depth and FP
    x /= max(abs(x)) # normalize to 1.0
    # now lowpass filter
    b = firwin(31, 900. / fs)
    x = lfilter(b, 1., x)
    x[::-1] = lfilter(b, 1., x[::-1])
    frames = getframes(x)
    return frames 

def upsample(x, L):
    assert isinstance(L, int)
    return resample(x, L, 1)

def downsample(x, M):
    assert isinstance(M, int)
    return resample(x, 1, M)

def resample(x, L, M):
    """resample with the ratio of integers L/M"""
    assert isinstance(L, int)
    assert isinstance(M, int)
    assert L > 0
    assert M > 0
    if L == M:
        return x
    if L > 1:
        x = expand(x, L)
        x *= L # gain                                                             
    b = firwin(31, 1. / max(M,L))
    x = lfilter(b, 1., x)
    x[::-1] = lfilter(b, 1., x[::-1]) # and again in reverse to cancel delay      
    if M > 1:
        x = decimate(x, M)
    return x

def expand(x, L):
    assert isinstance(L, int)
    y = zeros(len(x) * L, dtype=x.dtype)
    y[::L] = x
    return y

def decimate(x, M):
    assert isinstance(M, int)
    return x[::M].copy()

def getframes(wav, size=DEFAULT_FRAME_DUR, rate=DEFAULT_FRAME_RATE):
    """Given an array of samples and sampling frequency, return an
    array of frames. By default, a hanning window is applied on the
    frames before they are output. Frame size and rate can be
    specified in seconds using optional arguments."""
    nsize = int(round(size * fs))
    nrate = int(round(rate * fs))
    N = (len(wav) - nsize) // nrate + 1
    frames = np.empty((N, nsize))
    wframes = np.empty((N, nsize))
    if WINDOW == 'hanning':
        win = np.hanning(nsize)
    elif WINDOW == 'rect':
        win = np.ones((nsize),)
    for f in range(N):
        a = f * nrate
        frames[f] = wav[a : (a + nsize)] * win
    print ("Created %d frames (Hanning), each %d long, at %d frame/sec \n "%(N, nsize, 1.0/rate))
    return  frames 

def getframes1(wav, size=DEFAULT_FRAME_DUR, rate=DEFAULT_FRAME_RATE, WINDOW = 'hanning'):
    nsize = int(round(size * fs))
    nrate = int(round(rate * fs))
    N = (len(wav) - nsize) // nrate + 1
    frames = np.empty((N, nsize))
    wframes = np.empty((N, nsize))
    if WINDOW == 'hanning':
        win = np.hanning(nsize)
    elif WINDOW == 'rect':
        win = np.ones((nsize),)
    for f in range(N):
        a = f * nrate
        frames[f] = wav[a : (a + nsize)] * win
    print ("Created %d frames (Hanning), each %d long, at %d frame/sec \n "%(N, nsize, 1.0/rate))
    return  frames


def logspectralentropy(frames):
    """Compute Log-Spectral Entropy"""
    F, d = frames.shape
    specent = np.empty((F, 1))
    for f in range(F):
        specent[f] = estimateEntropy1d(np.log(np.abs(rfft(frames[f])) + eps))
    return specent

def computeRFFT(frames):
    """Compute real FFT of frames."""
    F, d = frames.shape
    if d == 0:
        logger.warn("Skipping utterance, can't compute FFT of zero dim vector")
        return None
    n = round(2 ** np.ceil(np.log2(d) + 0.5))
    rffts = np.empty((F, n/2))
    for f in range(F):
        rffts[f] = rfft(frames[f], n=n)[0:n/2]
    return rffts


def entropyEstimation ( mat ):
   ## we can apply some modifications as well!
   nFrames , FL = mat.shape
   if nFrames == 0:
       logger.warn("Skipping utterance, can't compute FFT of zero dim vector")
       return None
   fftRes = int(2 ** np.ceil(np.log2(FL) + 0.5))
   rffts =    np.abs(np.fft.rfft(mat,fftRes))
   prob = rffts / np.tile( np.sum(rffts , axis = 1)+eps , ( rffts.shape[1],1)).T
   entropy = -np.sum( prob*np.log2(prob+eps) , axis = 1 ) / np.log2(rffts.shape[1])
   entropy = medfilt(entropy , 5)
   return entropy

def computeRenyiEntropy(x, alpha, nbins, range):
    """Compute the Renyi entropy with parameter alpha for a given
    array of observations. Values beyond the given range, a tuple
    (min, max), are ignored and the rest is binned into nbins"""
    hist, edges = histogram(x, bins=nbins, range=range, normed=True)
    rentropy = 0
    ntotal = len(x)
    for i in range(len(nbins)):
        ni = hist[i] * nototal
        if hist[i] > 0:
            d = ni * pow(hist[i], alpha)
            rentropy = rentropy - log(d + eps)
    if alpha != 1:
        rentropy = rentropy / (1.0 - alpha)
    return rentropy

def estimateEntropy1d(x):
    """Estimate entropy of a 1-d signal (Learned-Miller and Fisher,
    JMLR, 2003; Vasicek, Royal Stats Soc, 1976) using the m-spacing
    entropy estimator which puts a variable-width uniform kernels
    locally on each sample whose width extends to the mth nearest
    neighbor. Recommended value of m to achieve low variance is
    sqrt(N)."""
    n = len(x)
    if n == 0:
        return 0
    ix = x.copy()
    ix.sort()
    m = int(np.floor(np.sqrt(n)))
    en = 0.0
    for i in range(m,n):
        intvl = ix[i] - ix[i-m]
        if intvl == 0:
            continue
        en = en + np.log(intvl)
    return en * -1.0

