"""
written by: Meysam Asgari 15/12/2013 ; CSLU / OHSU

## common speech processing functions
"""

import numpy as np , pdb ,os , struct
import scipy.spatial.distance as dist
from scipy import linalg as LA
from lib import *

# import matplotlib
# matplotlib.use("PDF")
# from matplotlib import mlab as mlab, pylab as plt
execfile('src/cfg.py')


def Bases (nbasis , FL):
        I = nbasis # number of basis functions                          
        seg = I-1
        len_b = 2*seg + np.fix(FL/seg)*seg
        len_w = len_b/seg
        ham = np.hamming(2*len_w)
        basis = np.zeros((len_b,I))
        basis[0:len_w , 0] = ham[len_w  :]
        for i in range(I-2):
                basis[i*len_w : i*len_w + 2*len_w ,i+1] = ham
        basis[len_b-len_w : ,I-1] = ham[0:len_w]
        bases =  basis[0:int(FL),:]
        return bases

def genPseudoInvMatsVarToFile( winLen ,Ainv=None , Fmin = F0_MIN , Fmax = F0_MAX , F0=None):
    global TinvMat
    TinvMat = TinvMatDir +'TinvMat_'+ TYPE_of_MODEL+'_'+str(fs)+'Hz'+'_'+str(winLen)+'FL'+'_'+'F0res'+str(F0_RES)+'Hz_'+'F0min'+str(F0_MIN)+'Hz_'+'F0max'+str(F0_MAX)+'Hz_'+'nH'+str(NUMBER_OF_HARMONICS)+'_'+WINDOW+'.bin'
    if not os.path.isfile(TinvMat):
        
        pInvMatsVar = []
        Basis = Bases ( NUM_BASES_FUN , winLen)
        z = np.ones((winLen, 1))
        nH = NUMBER_OF_HARMONICS #number of Harmonics     
        I = len(Basis[0,:])
        D1 = np.zeros((winLen,nH*I))
        D2 = np.zeros((winLen,nH*I))
        fzero= np.arange(Fmin, Fmax+1, F0_RES, dtype=float)
        T_len = winLen*winLen
        T_vec = np.empty((T_len*len(fzero) , ))
        t1= np.arange(1, winLen+1, 1, dtype=float)
        t1=t1/fs
        t2=np.arange(0, nH+1, 1, dtype=float)
        X1, Y1 = np.meshgrid(t2,t1)
        if WINDOW == 'hanning':
            win = np.hanning(winLen)
            W = np.diag(win)
        elif WINDOW == 'rect':
            win = np.ones((winLen),)
            W = np.diag(win)
        if F0!=None:
            fzer0 = F0
        for q in range(len(fzero)):
            omega=2*np.pi*fzero[q]
            Cos=np.cos(omega*X1*Y1)    # WIN_LEN x N_HRMNICS 
            Sin=np.sin(omega*X1*Y1)    # WIN_LEN x N_HRMNICS   
            # Harmonics with time varying amplitudes, composed of I basis
            for t in range(winLen):
                for h in range(nH):
                    D1[t][h*I:(h+1)*I] = Cos[t,h] * Basis[t]  #D1: M x (nH*I) 
                    D2[t][h*I:(h+1)*I] = Sin[t,h] * Basis[t]  #D1: M x (nH*I) 
            A = np.concatenate((z, np.concatenate((D1, D2), 1)), 1)
            A = np.dot(W,A)
            rc = np.max(np.shape(A))*np.max(np.linalg.svd(A)[1])*eps
            PinvA = np.linalg.pinv(A, rcond=rc)
            P = np.dot(A , np.linalg.pinv(A, rcond=rc))
            T_vec [q*T_len : (q+1)*T_len] = P.flatten()
        writeBin(TinvMat , T_vec)

def genPseudoInvMatsVar( winLen ,Ainv=None , Fmin = F0_MIN , Fmax = F0_MAX , F0=None):
    
    pInvMatsVar = []
    AInvMats = []
    Basis = Bases ( NUM_BASES_FUN , winLen)
    z = np.ones((winLen, 1))
    nH = NUMBER_OF_HARMONICS #number of Harmonics    
    I = len(Basis[0,:])
    D1 = np.zeros((winLen,nH*I))
    D2 = np.zeros((winLen,nH*I))
    fzero= np.arange(Fmin, Fmax+1, F0_RES, dtype=float)
    t1= np.arange(1, winLen+1, 1, dtype=float)
    t1=t1/fs
    t2=np.arange(0, nH+1, 1, dtype=float)
    X1, Y1 = np.meshgrid(t2,t1)
    if WINDOW == 'hanning':
        win = np.hanning(winLen)
        W = np.diag(win)
    elif WINDOW == 'rect':
        win = np.ones((winLen),)
        W = np.diag(win)
    if F0!=None:
        fzer0 = F0
    for q in range(len(fzero)):
        omega=2*np.pi*fzero[q]
        Cos=np.cos(omega*X1*Y1)    # WIN_LEN x N_HRMNICS            
        Sin=np.sin(omega*X1*Y1)    # WIN_LEN x N_HRMNICS
            # Harmonics with time varying amplitudes, composed of I basis
        for t in range(winLen):
            for h in range(nH):
                D1[t][h*I:(h+1)*I] = Cos[t,h] * Basis[t]  #D1: M x (nH*I)
                D2[t][h*I:(h+1)*I] = Sin[t,h] * Basis[t]  #D1: M x (nH*I)     
        A = np.concatenate((z, np.concatenate((D1, D2), 1)), 1)
        A = np.dot(W,A)
        rc = np.max(np.shape(A))*np.max(np.linalg.svd(A)[1])*eps
        PinvA = np.linalg.pinv(A, rcond=rc)
        P = np.dot(A , np.linalg.pinv(A, rcond=rc))
        pInvMatsVar.append(P)
        AInvMats.append(PinvA)
    if Ainv != None:
        return pInvMatsVar , AInvMats
    else: return pInvMatsVar

def genPseudoInvMats( winLen , Ainv=None , window=None , Fmin = F0_MIN , Fmax = F0_MAX):
        '''Compute and cache pInvMats for all candidate pitch values'''
        pInvMats = []
        AInvMats = []
        fzero= np.arange(Fmin, Fmax+1, F0_RES, dtype=float)
        t1= np.arange(1, winLen+1, 1, dtype=float)
        t1=t1/fs
        t2=np.arange(0, NUMBER_OF_HARMONICS+1, 1, dtype=float)
        X1, Y1 = np.meshgrid(t2,t1)
        if WINDOW == 'hanning':
            win = np.hanning(winLen)
            W = np.diag(win)
        elif WINDOW == 'rect':
            win = np.ones((winLen),)
            W = np.diag(win)
        for q in range(len(fzero)):
            omega=2*np.pi*fzero[q]
            Cos=np.cos(omega*X1*Y1)    # WIN_LEN x N_HRMNICS
            Sin=np.sin(omega*X1*Y1)    # WIN_LEN x N_HRMNICS
            A=np.c_[Cos,Sin[:,1:]]
            A = np.dot(W,A)
            rc = np.max(np.shape(A))*np.max(np.linalg.svd(A)[1])*eps
            PinvA = np.linalg.pinv(A, rcond=rc)
            P =  np.dot(A , np.linalg.pinv(A, rcond=rc)) 
            pInvMats.append(P)
            AInvMats.append(PinvA)
        
        if Ainv != None:
            return pInvMats , AInvMats
        else: return pInvMats

def genPseudoInvMatsToFile( winLen , Fmin = F0_MIN , Fmax = F0_MAX):
        '''Compute and cache pInvMats for all candidate pitch values'''
        global TinvMat
        TinvMat = TinvMatDir +'TinvMat_'+ TYPE_of_MODEL+'_'+str(fs)+'Hz'+'_'+str(winLen)+'FL'+'_'+'F0res'+str(F0_RES)+'Hz_'+'F0min'+str(F0_MIN)+'Hz_'+'F0max'+str(F0_MAX)+'Hz_'+'nH'+str(NUMBER_OF_HARMONICS)+'_'+WINDOW+'.bin'
        if not os.path.isfile(TinvMat):
            pInvMats = []
            fzero= np.arange(Fmin, Fmax+1, F0_RES, dtype=float)
            T_len = (2*NUMBER_OF_HARMONICS +1)*winLen
            T_vec = np.empty((T_len*len(fzero) , ))
            t1= np.arange(1, winLen+1, 1, dtype=float)
            t1=t1/fs
            t2=np.arange(0, NUMBER_OF_HARMONICS+1, 1, dtype=float)
            X1, Y1 = np.meshgrid(t2,t1)
            if WINDOW == 'hanning':
                win = np.hanning(winLen)
                W = np.diag(win)
            elif WINDOW == 'rect':
                win = np.ones((winLen),)
                W = np.diag(win)
            for q in range(len(fzero)):
                omega=2*np.pi*fzero[q]
                Cos=np.cos(omega*X1*Y1)    # WIN_LEN x N_HRMNICS 
                Sin=np.sin(omega*X1*Y1)    # WIN_LEN x N_HRMNICS      
                A=np.c_[Cos,Sin[:,1:]]
                A = np.dot(W,A)
                tmp = np.linalg.inv(LA.sqrtm( np.dot(A.T,A) ) )
                T = np.dot( tmp , A.T).real # P = np.dot(T.T ,T)      
                T_vec [q*T_len : (q+1)*T_len] = T.flatten()  
            writeBin(TinvMat , T_vec)

def genPinvA( winLen  , Fmin = F0_MIN , Fmax = F0_MAX):
        AInvMats = []
        fzero= np.arange(Fmin, Fmax+1, F0_RES, dtype=float)
        t1= np.arange(1, winLen+1, 1, dtype=float)
        t1=t1/fs
        t2=np.arange(0, NUMBER_OF_HARMONICS+1, 1, dtype=float)
        X1, Y1 = np.meshgrid(t2,t1)
        win = np.hanning(winLen)
        W = np.diag(win)
        for q in range(len(fzero)):
               omega=2*np.pi*fzero[q]
               Cos=np.cos(omega*X1*Y1)    # WIN_LEN x N_HRMNICS        
               Sin=np.sin(omega*X1*Y1)    # WIN_LEN x N_HRMNICS  
               A=np.c_[Cos,Sin[:,1:]]
               A = np.dot(W,A)
               rc = np.max(np.shape(A))*np.max(np.linalg.svd(A)[1])*eps
               PinvA = np.linalg.pinv(A, rcond=rc)
               AInvMats.append(PinvA)
        return AInvMats

def genPseudoInvMats_map( winLen , Lambda , Mean ,  Fmin = F0_MIN , Fmax = F0_MAX ):
    '''Compute and cache pInvMats for all candidate pitch values'''
    pInvMats = []
    penalty = []
    fzero= np.arange( Fmin, Fmax+1, F0_RES, dtype=float)
    t1= np.arange(1, winLen+1, 1, dtype=float)
    t1=t1/fs
    t2=np.arange(0, NUMBER_OF_HARMONICS+1, 1, dtype=float)
    X1, Y1 = np.meshgrid(t2,t1)
    if WINDOW == 'hanning':
        win = np.hanning(winLen)
        W = np.diag(win)
    elif WINDOW == 'rect':
        win = np.ones((winLen),)
        W = np.diag(win)
    for q in range(len(fzero)):
        omega=2*np.pi*fzero[q]
        Cos=np.cos(omega*X1*Y1)    # WIN_LEN x N_HRMNICS         
        Sin=np.sin(omega*X1*Y1)    # WIN_LEN x N_HRMNICS        
        A=np.c_[Cos,Sin[:,1:]]
        A = np.dot(W,A)
        rc = np.max(np.shape(A))*np.max(np.linalg.svd(A)[1])*eps
        d = np.dot (A , np.linalg.inv(np.dot(A.T,A)+Lambda) )
        P = np.dot( d , A.T)
        pen = np.dot( np.dot(d , Lambda) , Mean)
        pInvMats.append(P)
        penalty.append(pen)
        #pdb.set_trace()
    return pInvMats , penalty

def silDetection(Frames):
    if SIL_ENG:
        pw = 10 * np.log10(np.mean(Frames * Frames,1))
        sil = pw < SIL_THRESHOLD  #silence detection
    else:
        sil = pw > 100
    if ENTROPY :
        entropy = entropyEstimation(Frames)
        sil_entropy= entropy > ENTROPY_THR
        sil = np.logical_or(sil,sil_entropy)
    no_sil =  np.logical_not(sil) # not silence (either voiced or unvoiced or even noise)
    return np.where(sil==True)[0], np.where(no_sil==True)[0]

def getf0Features(  frames  , Fs ,  Fmin = F0_MIN , Fmax = F0_MAX , voiceProb= None , PitchDetction= None):
    global fs
    fs = Fs
    nframe , FL = frames.shape
    sil_ind , sp_ind = silDetection(frames)
    noise_var , noise_var_min = NoiseVarianceEstimation(frames[sp_ind])
    LL = -FL/2*np.log(noise_var)
    #hard_pitch = F0_MIN + (np.argmax(LL,0))*F0_RES
    #######################################################
    if PitchDetction :
      pitch = np.zeros(( nframe ))
      pitch[sp_ind] =  F0_MIN + F0_RES * viterbi(LL, genTransMat(SIGMA_TRANS \
                                                     , Fmin = F0_MIN , Fmax = F0_MAX ) , ret='pitch')
    ######################################################
    vuvll = np.zeros((2,len(sp_ind)))
    E_Y = np.sum(frames[sp_ind] ** 2, 1)
    vuvll[0] = np.log(E_Y - noise_var_min ) # Reconsrtucted signal Energy (voiced state)
    vuvll[1] = np.log( noise_var_min ) # Noise Energy ( unvoiced state)     
    vuvll[1] = vuvll[1,:] + VU_THRESHOLD
    vuv = viterbi(vuvll, np.log(VUV_TRANSITION) , ret='vuv')
    vuv1 = findUVerr(vuv) # smoothing
    vIdx = np.nonzero(vuv == 1)[0]
    uvIdx = np.nonzero(vuv == 0)[0]
    vuv = np.zeros(( nframe ))
    vuv[sp_ind] = vuv1
    if voiceProb :
        probvoice = vuvll[0,:]-vuvll[1,:] - VU_THRESHOLD
        probvoice = featNormalize(probvoice,0.05,1)
        prob_voice = np.zeros(( nframe ))
        prob_voice[sp_ind] = probvoice
        if np.all(np.isnan(prob_voice)):
            print 'Nan in voicing probability '
    if PitchDetction != None and voiceProb != None:
        pitch[uvIdx]=0
        return pitch , prob_voice , vuv
    elif PitchDetction != None:
        pitch[uvIdx]=0
        return pitch , vuv
    elif voiceProb != None:
        return prob_voice , vuv
    else: return   vuv

def thr_adaptation(snr):
        clean_thr = -.3
        somewhat_noisy_thr = -.8
        Noisy_thr = -1.2
        veryNoisy_thr = -1.4 
        if snr <= .3 : THR = veryNoisy_thr
        elif snr > .3 and snr <= .5 : THR =  Noisy_thr
        elif snr > .5 and snr <= 1 : THR =  somewhat_noisy_thr
        else:  THR = clean_thr
        return THR

def genHarCofStat(HarCoff):
    #pdb.set_trace()
    Mean = np.mean(HarCoff,axis =0)                                
    b_Sigma = np.dot((HarCoff-Mean).T,(HarCoff-Mean))/(np.shape(HarCoff)[0]-1) 
    b_inv_Sig = np.linalg.inv(b_Sigma)                              
    #Lambda_HM = b_inv_Sig*np.eye(len(b_inv_Sig))# why is not full cov? 
    return Mean , b_inv_Sig

def NoiseVarianceEstimation_map (  Frames , pInvMats , penalty, F_min=F0_MIN , F_max=F0_MAX  ): 

    FL , nframe= np.shape(Frames)
    pw = np.sum(Frames * Frames, 0)
    cands = np.arange(F_min, F_max+1, F0_RES, dtype=float)
    var = np.zeros((len(cands),nframe))
    pitch = np.zeros((nframe, ))
    for q in range(len(cands)):
        Py =  np.dot(pInvMats[q],Frames)
        gamma = penalty[q]
        for i in range(nframe):
            rec_energy = np.dot( (Py[:,i]+gamma).T , (Py[:,i]+gamma) )
            var[q][i] = ( pw[i] - rec_energy)
    sigma = np.min( var , 0 )
    # Convoloution of the LL
    cll_conv = np.zeros((len(cands),nframe))
    Wnd = np.hamming(HAM_WND)
    cll = var
    half_wnd = len(Wnd)/2-1
    for i  in range(nframe):
        Conv = np.convolve(cll[:,i],Wnd,'valid')
        tmp = np.ones(len(cands),)*np.max(Conv)
        tmp[half_wnd : half_wnd + len(Conv)]  = Conv
        cll_conv[:,i] = np.sqrt(np.dot(cll[:,i],cll[:,i].T)/(np.dot(tmp,tmp.T)+eps))*tmp

    return cll_conv , sigma
    
def NoiseVarianceEstimation (  Frames , F_min=F0_MIN , F_max=F0_MAX  ):
    nframe , FL = np.shape(Frames)
    nframe  , FL  = np.shape(Frames)
    if TYPE_of_MODEL == 'HM': genPseudoInvMatsToFile( FL , Fmin = F0_MIN , Fmax = F0_MAX)
    elif TYPE_of_MODEL =='TVHM' : genPseudoInvMatsVarToFile( FL , Fmin = F0_MIN , Fmax = F0_MAX)
    else: raise StandardError, " The model types has to be either HM or TVHM"

    pw = np.sum(Frames * Frames, 1)
    Wnd = np.hamming(HAM_WND)
    half_wnd = len(Wnd)/2-1
    cands = np.arange(F_min, F_max+1, F0_RES, dtype=float)
    var = np.zeros((len(cands),nframe))
    Tinv = readPinvMat ( TinvMat )
    if TYPE_of_MODEL == 'HM':
        for q in range(len(cands)):
            Tmat = Tinv[q,:,:]  #(2*nH+1)*FL  the T matrix is loaded
            for i in xrange(nframe):
                rec = np.dot(Tmat ,Frames[i,:])
                var[q][i]  = pw[i] -  np.dot(rec.T,rec)
    elif TYPE_of_MODEL == 'TVHM':
        for q in range(len(cands)):
            P = Tinv[q,:,:]
            #Py =  np.dot(Tinv[q,:,:],Frames.T) # the P matrix is loaded
            for i in range(nframe):
                #var[q][i] = ( pw[i] - np.dot(Frames[i,:].T ,Py[:,i]) )
                frm = Frames[i,:]
                var[q][i] = ( np.dot(frm.T,frm) - np.dot(frm.T , np.dot(P,frm)) )
    #### conv ####                                           
    cll_conv = np.zeros((len(cands),nframe))
    Wnd = np.hamming(HAM_WND)
    cll = var
    half_wnd = len(Wnd)/2-1
    for i  in range(nframe):
        Conv = np.convolve(cll[:,i],Wnd,'valid')
        tmp = np.ones(len(cands),)*np.max(Conv)
        tmp[half_wnd : half_wnd + len(Conv)]  = Conv
        cll_conv[:,i] = np.sqrt(np.dot(cll[:,i],cll[:,i].T)/(eps+np.dot(tmp,tmp.T)))*tmp
    sigma = np.min( var , 0 )
    return cll_conv , sigma

    
def getf0_MAP(  frames , Fs , Fmin = F0_MIN , Fmax = F0_MAX , voiceProb= False , PitchDetction= False):
    global fs
    fs = Fs
    nframe , FL = frames.shape
    sil_ind , sp_ind = silDetection(frames)
    noise_var , noise_var_min = NoiseVarianceEstimation(frames[sp_ind])
    noise_variance = np.mean(noise_var_min)
    LL = -FL/2*np.log(noise_var)
    pitch_idx = viterbi(LL, genTransMat(SIGMA_TRANS , Fmin = F0_MIN , Fmax = F0_MAX ) , ret='pitch')
    HarCof = HARMONIC_COFF ( frames[sp_ind] , pitch_idx )
    harCofMean , harCovInvMat = genHarCofStat(HarCof.T)
    Lambda = noise_variance * harCovInvMat
    pInvMats , Gamma = genPseudoInvMats_map( FL , Lambda , harCofMean  )
    noise_var , noise_var_min = NoiseVarianceEstimation_map (  frames[sp_ind].T, pInvMats , Gamma , F_min = Fmin  , F_max = Fmax )
    #pdb.set_trace()
    LL = -FL/2*np.log(noise_var)
    if PitchDetction :
      pitch = np.zeros(( nframe ))
      pitch[sp_ind] =  F0_MIN + F0_RES * viterbi(LL, genTransMat(SIGMA_TRANS \
                                                     , Fmin = F0_MIN , Fmax = F0_MAX ) , ret='pitch')
    ######################################################
    vuvll = np.zeros((2,len(sp_ind)))
    E_Y = np.sum(frames[sp_ind] ** 2, 1)
    vuvll[0] = np.log(E_Y - noise_var_min ) # Reconsrtucted signal Energy (voiced state)                  
    vuvll[1] = np.log( noise_var_min ) # Noise Energy ( unvoiced state)
    vuvll[1] = vuvll[1,:] + VU_THRESHOLD
    vuv = viterbi(vuvll, np.log(VUV_TRANSITION) , ret='vuv')
    vuv1 = findUVerr(vuv) # smoothing
    vIdx = np.nonzero(vuv == 1)[0]
    uvIdx = np.nonzero(vuv == 0)[0]
    vuv = np.zeros(( nframe ))
    vuv[sp_ind] = vuv1
    if voiceProb :
        probvoice = vuvll[0,:]-vuvll[1,:] - VU_THRESHOLD
        probvoice = featNormalize(probvoice,0.05,1)
        prob_voice = np.zeros(( nframe ))
        prob_voice[sp_ind] = probvoice
        if np.all(np.isnan(prob_voice)):
            print 'Nan in voicing probability '
    if PitchDetction != None and voiceProb != None:
        pitch[uvIdx]=0
        return pitch , prob_voice , vuv
    elif PitchDetction != None:
        pitch[uvIdx]=0
        return pitch , vuv
    elif voiceProb != None:
        return prob_voice , vuv
    else: return   vuv

def f0Estimation  (  var  ):
    FL , nframe = np.shape(var)
    f0 = np.zeros((nframe,))
    #pdb.set_trace()
    for frm in range(nframe):
        vec = -var[:,frm]
        f0[frm] =LobeAreaEstimate(vec) #+ F0_MIN
    return f0

def HARMONIC_COFF ( Frames   , pitch_idx  ):
       nframe , FL  = np.shape(Frames)
       PinvA = genPinvA( FL)
       harCoeff = np.zeros((PinvA[0].shape[0] , nframe))
       for i in range(nframe):
           harCoeff[:,i] = np.dot (PinvA[int(pitch_idx[i])] , Frames[i,:]+eps )
       return   harCoeff

def getf0_unvoiced ( frames ):
    nFrame , FL = frames.shape
    var , var_min = NoiseVarianceEstimation (  frames , F_min = F0_MIN , F_max = F0_MAX )
    I_max = np.argmin(var,0)
    return I_max

def viterbi(obs, trans , ret='vuv'):
        m, n = np.shape(obs)
        states    = np.zeros((m, n+1))
        backtrace = np.zeros((m, n+1))
        for i in range(1, n+1):
                for j in range(m):
                        delta = states[:, i-1] + trans[:, j]
                        backtrace[j, i] = np.argmax(delta)
                        states[j, i] = delta[backtrace[j, i]] + obs[j,i-1]
        bestPath = np.zeros(n)
        VUv_lable = np.zeros(n)
        bestPath[n-1] = np.argmax(states[:, n] )
        for i in range(n-2, -1, -1):
                bestPath[i] = backtrace[bestPath[i+1], i+2]
        out = bestPath
        if ret == 'vuv':
            for i in range (n):
                if bestPath[i] < 1 :
                    VUv_lable[i] = 1
                else:
                    VUv_lable[i] = 0
            out = VUv_lable
        return out

def genTransMat(sig_tran , Fmin = F0_MIN , Fmax = F0_MAX):
        fzero = np.arange(Fmin, Fmax+1, F0_RES, dtype=float)
        nf0   = len(fzero)
        pTrans = np.zeros((nf0, nf0))
        for i in range(nf0):
                for j in range(nf0):
                        pTrans[i][j] =  -.5*np.log(2.0*np.pi*sig_tran*sig_tran) -  1/(2.0*sig_tran*sig_tran)* (fzero[i] - fzero[j])**2
        return pTrans

def writeBin( fn , vec , type = 'f'):
    os.system('mkdir -p '+fn.split(fn.split('/')[-1])[0])
    f = open(fn, 'wb')
    for i in xrange(len(vec)):
        f.write(struct.pack( type , vec[i] ))
    f.close()

def readBin ( fn  , fmt = 'f'):
    fp = open(fn, 'rb')
    wav = fp.read()
    sampwidth = 4
    fmt = 'f'
    wav = struct.unpack('<%d%s' %(len(wav) / sampwidth, fmt), wav)
    return wav

def readPinvMat ( fname , Fmin = F0_MIN , Fmax = F0_MAX, F0_res = F0_RES ):
    f0range = len(np.arange(Fmin, Fmax+1, F0_res))
    FL = int(round(DEFAULT_FRAME_DUR * fs))
    float_vec =  readBin ( fname , fmt = 'f')
    if TYPE_of_MODEL == 'HM': pInvMats = np.resize(float_vec , (f0range , 2*NUMBER_OF_HARMONICS +1 , FL))
    elif TYPE_of_MODEL =='TVHM' : pInvMats = np.resize(float_vec , (f0range , FL , FL))
    else: raise StandardError, " The model types has to be either HM or TVHM"

    return pInvMats
          
def findUVerr(vec,target=np.expand_dims(np.array([1,1,0,1,1]),1)):
    ref_len = len(target)
    vec_len = len(vec)
    d = []
    for f in xrange(vec_len - ref_len):
        if np.dot(vec[f:ref_len+f].T , target ) == 4 and  vec[f:ref_len+f][2] == 0:
            d.append(f+ref_len/2)
    if len(d)!=0:
        vec[np.array(d,int)] = 1
    return vec
        
def fileNameExt( name ):
    name = name.strip()
    tmp = name.split('/')
    return name[len(name)-len(tmp[-1]):].strip()

def fileDirExt (name):
    name = name.strip()
    tmp = name.split('/')
    return name[0:len(name) - len(tmp[-1])]

