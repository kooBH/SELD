import librosa
import numpy as np
import soundfile as sf 

import os,glob
import pandas as pd

import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import warnings
#warnings.filterwarnings('error') 
"""
return _methods._mean(a, axis=axis, dtype=dtype,                                      
/home/kiosk/anaconda3/envs/dnn/lib/python3.9/site-packages/numpy/core/_methods.py:189: RuntimeWarning: invalid value encountered in true_divide
"""

root_audio = "/home/data/kbh/DCASE2022/raw/"
root_rir = "/home/data/kbh/DCASE2022/TAU-SRIR_DB_split/"
root_doa = "/home/data/kbh/DCASE2022/TAU-SRIR_DB_DOA/"
root_output = "/home/nas3/DB/DCASE2022/extra-synth/"

idx_shift = 1 # Due to label issue
config = [
    ("Female speech, woman speaking", 0.7),
    ("Male speech, man speaking", 0.7),
    ("Clapping", 5),
    ("Telephone", 3),
    ("Laughter", 15),
    ("Domestic sounds", 0.1),
    ("Walk, footsteps", 7),
    ("Door, open or close", 4),
    ("Music", 0.9),
    ("Musical instrument", 5),
    ("Water tap, faucet", 0.1),
    ("Bell", 7),
    ("Knock", 4)
]

dist = [
    ("Clapping", 17),
    ("Telephone", 10),
    ("Laughter", 26),
    ("Domestic sounds", 13),
    ("Walk, footsteps", 16),
    ("Door, open or close", 15),
    ("Music", 19),
    ("Musical instrument", 6),
    ("Water tap, faucet", 5),
    ("Bell", 10),
    ("Knock", 15) 
]

parser = argparse.ArgumentParser()
parser.add_argument('--dir_out','-o',type=str,required=True)
parser.add_argument('--n_out','-n',type=int,required=True)
args = parser.parse_args()

path_output = os.path.join(root_output,args.dir_out)

acc_freq = 0
for i in dist :
    acc_freq +=i[1]
dist_class = np.zeros(acc_freq,np.int32)

acc_freq = 0
idx_class = 2
for i in dist :
    dist_class[acc_freq:acc_freq+i[1]] = idx_class
    acc_freq +=i[1]
    idx_class +=1

list_audio =[]
for i in range(13):
    list_audio.append([x for x in glob.glob(os.path.join(root_audio,str(i),"*.wav"))])
list_rir = [x for x in glob.glob(os.path.join(root_rir,"*.npy"))]

sr = 24000
len_sec = 60.0
n_total_sample = int(sr*len_sec)

def spread_ratio(
    ratio, 
    category,
    n_total_sample,
    pad_expand_ratio=1.0,
    sr=24000,
    len_label_frame = 2400,
    split_top_db=25
    ):
    occ = 0
    act = 0
    avg_energy_std = None
    is_enough = False

    raw = np.zeros(n_total_sample)
    label = np.zeros((int(n_total_sample/len_label_frame))) # [n_label_frame]
    label[:]=-1

    while not is_enough : 
        len_data = 0
        while len_data == 0 :
            # Load
            path_data = list_audio[category][np.random.randint(len(list_audio[category]))]
            data,_ = librosa.load(path_data,sr=sr)

            # Sampling
            data_sed = librosa.effects.split(data,split_top_db)    
            intv_chunk = data_sed[np.random.randint(len(data_sed))]
            data = data[intv_chunk[0]:intv_chunk[1]]

            len_data = len(data)
        
        # normalization
        data = data/np.max(np.abs(data))
    
        # Padding
        avail = n_total_sample - occ
        poss = int((1-ratio)/ratio*len(data))
        pad = int(0.5*poss + np.random.randint(poss)*pad_expand_ratio)
        
        idx_end = occ+pad+len(data)
        # Can't = > Cut
        if idx_end > n_total_sample : 
            pad = 0
            data = data[:n_total_sample-occ]
            is_enough=True

        ## I don't know what to do
        if len(data) == 0:
            continue
            
        # scaling 
        if avg_energy_std is None :
            avg_energy = np.mean(np.abs(data))
            avg_energy_std  = avg_energy
        else : 
            avg_energy = np.mean(np.abs(data))
            weight = avg_energy_std/(avg_energy)
            data = data*weight
            
        # Append
        raw[occ + pad:occ + pad+len(data)] = data
        # label
        label[int((occ+pad)/2400): int((occ + pad+len(data))/2400)] = category

        # update 
        occ += pad+len(data)
        act += len(data)
        act_ratio = act/n_total_sample
        
        # enough
        if act_ratio >= ratio :
            is_enough = True

    normalization_coef = np.max(np.abs(raw))
    normalized_avg_energy = avg_energy_std/(normalization_coef)
    raw = raw/normalization_coef

    return raw,label, normalized_avg_energy

def spread_amount(
    amount, 
    category,
    n_total_sample,
    sr=24000,
    len_label_frame = 2400,
    split_top_db=25
    ):

    len_chunk = int(n_total_sample/amount)
    raw = np.zeros(n_total_sample)

    label = np.zeros((int(n_total_sample/len_label_frame))) # [n_label_framem]
    label[:]=-1

    avg_energy_std = None
    idx_last = 0

    for i_amount in range(amount) : 
        len_data = 0
        while len_data == 0 :
            # Load
            path_data = list_audio[category][np.random.randint(len(list_audio[category]))]
            data,_ = librosa.load(path_data,sr=sr)

            # Sampling
            data_sed = librosa.effects.split(data,top_db=split_top_db)    
            intv_chunk = data_sed[np.random.randint(len(data_sed))]
            data = data[intv_chunk[0]:intv_chunk[1]]
            
            len_data = len(data)

        # normalization
        data = data/np.max(np.abs(data))

        # interval
        idx_intv = i_amount*len_chunk
        
        # pass
        if idx_last >= (i_amount+1)*len_chunk :
            continue
        
        idx_start = np.random.randint(low=idx_intv,high=(i_amount+1)*len_chunk)

        # over
        if idx_last > idx_start :
            idx_intv = idx_last
        
        # not enough space => cut
        idx_end = idx_start + len(data)
        if idx_end > n_total_sample :
            data=data[:n_total_sample-idx_start]

        ## I don't know what to do
        if len(data) == 0:
            continue

        # scaling 
        if avg_energy_std is None :
            avg_energy = np.mean(np.abs(data))
            avg_energy_std  = avg_energy
        else : 
            avg_energy = np.mean(np.abs(data))
            weight = avg_energy_std/(avg_energy)
            data = data*weight
        
        # Add
        raw[idx_start : idx_start + len(data)] = data
        idx_last = idx_start + len(data)
        # Label
        label[int(idx_start/2400): int((idx_start + len(data))/2400)] = category
        
        
    normalization_coef = np.max(np.abs(raw))
    normalized_avg_energy = avg_energy_std/(normalization_coef)
    raw = raw/normalization_coef

    return raw,label, normalized_avg_energy


def RIR_extract():


    # rir : [n_traj,4,len_filter]
    
    bool_loaded = False

    while not bool_loaded :
        try :
            idx_rir = np.random.randint(0,high=len(list_rir))
            path_rir = list_rir[idx_rir]
            name_rir = path_rir.split('/')[-1]
            rir = np.load(path_rir,allow_pickle=True)
            bool_loaded=True
        except :
            bool_loaded=False

    # reduce reverb
    limit_reverb = np.random.randint(low=int(rir.shape[2]*0.2),high=int(rir.shape[2]*0.5))
    rir = rir[:,:,:limit_reverb]

    # reduce traj
    limit_traj = np.random.randint(low=10,high=rir.shape[0])
    rir = rir[:limit_traj,:,:]

    return rir,limit_traj, name_rir

def mix(idx_out):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    # for Multi-Processing
    import gpuRIR
    gpuRIR.activateMixedPrecision(False)
    gpuRIR.activateLUT(True)

    val_n_src = np.random.rand() 
    """
        n_src : 3   | 4   | 5
                0.3 | 0.4 | 0.3
    """
    if val_n_src > 0.7 :
        n_src = 3
    elif val_n_src > 0.3 :
        n_src = 4
    else :
        n_src = 5

    n_channel = 4

    audio = np.zeros((n_total_sample,n_channel))

    # label = pd.df()

    """
    Class Distribution
        n_speaker |  0   |   1 |   2 |   3 |
        prob      |  0.1 | 0.4 | 0.4 | 0.1 |
    """
    classes = np.zeros(n_src,np.int32)    
    c_class = 0

    val_n_speaker = np.random.rand()

    if val_n_speaker > 0.9 : 
        n_speaker = 3
    elif val_n_speaker > 0.5 :
        n_speaker = 2
    elif val_n_speaker > 0.1 :
        n_speaker = 1
    else :
        n_speaker = 0
    
    for i_class in range(n_speaker) :
        classes[i_class] = int(np.round(np.random.rand()))
    c_class = n_speaker

    for i_class in range(c_class,n_src):
        classes[i_class] = dist_class[np.random.randint(0,high=acc_freq)]

    #print("{} :: classes : {}".format(idx_out,classes))
    
    energy_criterion = None

    label = pd.DataFrame(columns=["idx","class","order","azimuth","elevation"])

    # for each class
    for i_cls in classes : 

        # audio
        if type(config[i_cls][1]) is float :
            t_audio,t_label, t_avg_energy = spread_ratio(config[i_cls][1],i_cls,n_total_sample)
        else :
            t_audio,t_label, t_avg_energy = spread_amount(config[i_cls][1],i_cls,n_total_sample)

        # RIR
        rir, l_traj, name_rir = RIR_extract()
        filtered = gpuRIR.simulateTrajectory(t_audio, rir.astype(np.float32))

        # cut
        filtered=filtered[:n_total_sample,:]

        # scailing
        if energy_criterion is None : 
            energy_criterion = t_avg_energy
        else :
            SIR = (np.random.rand()-0.5)*10 # -5 ~ 5
            weight = (energy_criterion/t_avg_energy)/np.sqrt(np.power(10,SIR/10))
            filtered *= weight
        
        audio[:,:] += filtered

        # Labeling
        DOA = np.load(os.path.join(root_doa,name_rir),allow_pickle=True)
        DOA = DOA[:l_traj,:]

        len_traj_chunk = int((n_total_sample)/l_traj)

        for i_idx in range(len(t_label)) : 
            if t_label[i_idx] != -1 :  
                cur_idx = int((i_idx*2400)/len_traj_chunk)
                x = DOA[cur_idx,0]
                y = DOA[cur_idx,1]
                z = DOA[cur_idx,2]
                azim = np.arctan2(y, x) * 180. / np.pi
                elev = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180. / np.pi

                label.loc[len(label.index)] = [i_idx+idx_shift, i_cls, 0, azim, elev]

    # normalization
    audio = audio/np.max(np.abs(audio)) 

    # Save Wav
    sf.write(os.path.join(path_output,"foa","fold1_"+str(idx_out)+".wav"),audio,sr)

    # Csv
    label = label.sort_values(by=['idx'])
    label.to_csv(os.path.join(path_output,"metadata","fold1_"+str(idx_out)+".csv"),header=False,index=False )

if __name__ == "__main__" : 

    os.makedirs(path_output+"/foa",exist_ok=True)
    os.makedirs(path_output+"/metadata",exist_ok=True)

    n = args.n_out

#    for i in tqdm(range(n)):
#        mix(i)

    num_cpu = cpu_count()

    arr = list(range(n))
    with Pool(num_cpu) as p:
        r = list(tqdm(p.imap(mix, arr), total=len(arr),ascii=True,desc='DCASE2022 extra synthesis'))
