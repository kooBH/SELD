"""
    preprocessing wav files into .pt files 
"""

import os,glob
import librosa
import numpy as np
import torch
from utils.hparams import HParam
import argparse
import pandas as pd

def get_foa_intensity_vectors(linear_spectra,mel_filter,n_mels,eps=1e-8):

    W = linear_spectra[:, :, 0]
    I = np.real(np.conj(W)[:, :, np.newaxis] * linear_spectra[:, :, 1:])
    E = eps + (np.abs(W)**2 + ((np.abs(linear_spectra[:, :, 1:])**2).sum(-1))/3.0 )
    
    I_norm = I/E[:, :, np.newaxis]
    I_norm_mel = np.transpose(np.dot(np.transpose(I_norm, (0,2,1)), mel_filter), (1,0,2))
    """
    foa_iv = I_norm_mel.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], n_mels * 3))
    if np.isnan(foa_iv).any():
        print('Feature extraction is generating nan outputs')
        exit()
    """
    return I_norm_mel


def convert_output_format_polar_to_cartesian(in_dict):
    out_dict = {}
    for frame_cnt in in_dict.keys():
        if frame_cnt not in out_dict:
            out_dict[frame_cnt] = []
            for tmp_val in in_dict[frame_cnt]:

                ele_rad = tmp_val[3]*np.pi/180.
                azi_rad = tmp_val[2]*np.pi/180

                tmp_label = np.cos(ele_rad)
                x = np.cos(azi_rad) * tmp_label
                y = np.sin(azi_rad) * tmp_label
                z = np.sin(ele_rad)
                out_dict[frame_cnt].append([tmp_val[0], tmp_val[1], x, y, z])
    return out_dict

def convert_output_format_cartesian_to_polar(in_dict):
    out_dict = {}
    for frame_cnt in in_dict.keys():
        if frame_cnt not in out_dict:
            out_dict[frame_cnt] = []
            for tmp_val in in_dict[frame_cnt]:
                x, y, z = tmp_val[2], tmp_val[3], tmp_val[4]

                # in degrees
                azimuth = np.arctan2(y, x) * 180 / np.pi
                elevation = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
                r = np.sqrt(x**2 + y**2 + z**2)
                out_dict[frame_cnt].append([tmp_val[0], tmp_val[1], azimuth, elevation])
    return out_dict

def align_meta_pt(df,
               n_sample,
               shift=256,
               n_fft=1024,
               hop_meta_s=0.1,
               fs=24000,
               max_n_target = 6
              ):
    hop_meta = fs*hop_meta_s
    ratio = hop_meta/shift
    n_frame = int(np.ceil(n_sample/shift)+1)
    
    out = torch.zeros(n_frame,max_n_target,3) # 3[class,azimuth,elevation]
    out[:,:,0] = -1 # init
    
    ## 
    
    for idx in df.index :
        idx_start = int(df.iloc[idx,0]*ratio)
        idx_end = int((df.iloc[idx,0]+1)*ratio)
        
        cnt=0
        while out[idx_start,cnt,0] != -1 :
            cnt+=1
        
        out[idx_start:idx_end,cnt,0] = df.iloc[idx,1] # class
        out[idx_start:idx_end,cnt,1] = df.iloc[idx,3] # azimuth
        out[idx_start:idx_end,cnt,2] = df.iloc[idx,4] # elevation
    
    return out

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, required=True)
parser.add_argument('--dir_meta', '-m', type=str, required=True)
parser.add_argument('--dir_audio', '-a', type=str, required=True)
parser.add_argument('--dir_out', '-o', type=str, required=True)
args = parser.parse_args()

hp = HParam(args.config)

name_config = args.config.split('/')[-1]
id_config = name_config.split('.')[0]

list_target = glob.glob(os.path.join(args.dir_audio,"*.wav"))
#print("{}".format(len(list_target)))

mel_wts = librosa.filters.mel(sr=hp.base.fs, n_fft=hp.base.n_fft, n_mels=hp.mel.n_mels)

n_sample = hp.base.n_sample
bin_cutoff = int(1)
bin_bottom = int(2)

list_feature=["STFT","IV","SALSA"]
fs = hp.base.fs
n_fft = hp.base.n_fft
shift = hp.base.shift

def process(idx):
    target_path = list_target[idx]
    target_name = target_path.split('/')[-1]
    target_id = target_name.split('.')[0]

    raw,_ = librosa.load(target_path,sr=hp.base.fs,mono=False)
    #print(raw.shape)

    n_channel, total_sample = raw.shape

    df = pd.read_csv(os.path.join(args.dir_meta,target_id + ".csv"),names=["idx","class","order","azimuth","elevation"])
    pt_label = align_meta_pt(df,total_sample)

    n_frame = 0
    idx_seg = 0
    # process per each segment
    while idx_seg < total_sample :
        #print(idx_seg)
        len_check = idx_seg + n_sample

        # length check : cut
        if len_check <= total_sample : 
            idx_end = len_check
            seg = raw[:,idx_seg:idx_end]

            frame_seg = int(idx_seg/hp.base.shift)
            frame_end = int(np.ceil((idx_end)/hp.base.shift))
            tmp_label = pt_label[frame_seg:frame_end,:]
        # length check : pad
        else : 
            shortage = len_check - total_sample
            idx_end = total_sample
            seg = np.zeros((n_channel,n_sample))
            seg[:,:n_sample- shortage] = raw[:,idx_seg:idx_end]

            frame_seg = int(idx_seg/hp.base.shift)
            frame_end = int(np.ceil((idx_end)/hp.base.shift))
            tmp_label = pt_label[frame_seg:frame_end,:]
            tmp_label = torch.nn.functional.pad(tmp_label,(0,0,0,0,0,int((shortage/hp.base.shift))))
            tmp_label[frame_end:,:,0]=-1



        # label seg
        #print("idx {} ~ {} | frame {} ~ {} ".format(idx_seg,idx_end,frame_seg,frame_end))

        #print("label : {}".format(tmp_label.shape))


        ## process

        # spec
        spec = []
        for c in range(n_channel) : 
            spec.append(librosa.stft(seg[c,:],n_fft=hp.base.n_fft))
        spec = np.array(spec)
        #print("spec : {}".format(spec.shape))

        # mel
        mel = []
        for c in range(1) : 
            mel.append(np.matmul(mel_wts,np.abs(spec[c])))
        mel = np.array(mel)
        #print("mel : {}".format(mel.shape))

        # Intensity Vector

        # Trasnpose for RNN process
        spec = spec
        mel = np.transpose(mel,(0,2,1))

        #print("spec.T : {}".format(spec.shape))

        IV =  get_foa_intensity_vectors(spec.T,mel_wts.T,hp.mel.n_mels)

        #print("mel : {}".format(mel.shape))
        #print("IV : {}".format(IV.shape))

        base = np.concatenate((mel,IV))
        #print("input : {}".format(base.shape))

        mfcc = []
        chroma_stft = []
        spectral_centroid  = []
        spectral_bandwidth = []
        spectral_contrast  = []
        spectral_flatness  = []

        mag = np.abs(spec)
        for c in range(n_channel) : 
            mfcc.append(librosa.feature.mfcc(seg[c,:], sr=fs, n_mfcc=40, n_fft = n_fft,hop_length=shift))
            chroma_stft.append(librosa.feature.chroma_stft(seg[c], sr=fs, n_fft = n_fft,hop_length=shift))

            # 아래는 mag만 가능
            spectral_centroid.append(librosa.feature.spectral_centroid(S=mag[c], sr=fs, n_fft = n_fft,hop_length=shift))

            spectral_bandwidth.append(librosa.feature.spectral_bandwidth(S=mag[c], sr=fs, n_fft = n_fft,hop_length=shift))

            spectral_contrast.append(librosa.feature.spectral_contrast(S=mag[c], sr=fs, n_fft = n_fft,hop_length=shift))

            spectral_flatness.append(librosa.feature.spectral_flatness(S=mag[c], n_fft = n_fft,hop_length=shift))

        mfcc = np.transpose(np.array(mfcc),(0,2,1))
        chroma_stft = np.transpose(np.array(chroma_stft),(0,2,1))
        spectral_centroid = np.transpose(np.array(spectral_centroid),(0,2,1))
        spectral_bandwidth = np.transpose(np.array(spectral_bandwidth),(0,2,1))
        spectral_contrast = np.transpose(np.array(spectral_contrast),(0,2,1))
        spectral_flatness = np.transpose(np.array(spectral_flatness),(0,2,1))

        #print("{}".format(mfcc.shape))
        #print("{}".format(chroma_stft.shape))
        #print("{}".format(spectral_centroid.shape))
        #print("{}".format(spectral_bandwidth.shape))
        #print("{}".format(spectral_contrast.shape))
        #print("{}".format(spectral_flatness.shape))

        data = {}
        # [4, T, F]
        data["data"] = np.concatenate((base,mfcc,chroma_stft,spectral_centroid,spectral_bandwidth,spectral_contrast,spectral_flatness),axis=-1)
        data["label"]=tmp_label


        #print("data : {}".format(data.shape))

        # save
        if target_id[4] in ["4"] : 
            torch.save(data,os.path.join(args.dir_out,id_config,"test",target_id+"_"+str(idx_seg)+'.pt'))
        else :
            torch.save(data,os.path.join(args.dir_out,id_config,"train",target_id+"_"+str(idx_seg)+'.pt'))
        idx_seg +=n_sample

if __name__=="__main__":
    import argparse
    import librosa
    from tqdm import tqdm
    from multiprocessing import Pool, cpu_count

    os.makedirs(os.path.join(args.dir_out,id_config,"train"),exist_ok=True)
    os.makedirs(os.path.join(args.dir_out,id_config,"test"),exist_ok=True)

    ## MP
    cpu_num = int(cpu_count()/2)

    arr = list(range(len(list_target)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(process, arr), total=len(arr),ascii=True,desc='processing'))
