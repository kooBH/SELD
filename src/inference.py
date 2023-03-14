import torch
import argparse
import numpy as np
import torchaudio
import librosa
import pandas as pd
import os
import sys
import glob

import warnings

warnings.filterwarnings('ignore')

# ERROR ..? 
#from preprocess import get_foa_intensity_vectors

from utils.hparams import HParam

from tqdm import tqdm

from label import label2mACCDOA,mACCDOA2label
from EINV2 import EINV2

import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method

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

def preprocess(raw,fs=24000,n_fft=1024,shift=256,n_mels=40): 
    n_channel, n_sample = raw.shape

    mel_wts = librosa.filters.mel(sr=fs, n_fft=n_fft, n_mels=n_mels)

    spec = []
    for c in range(n_channel) : 
        spec.append(librosa.stft(raw[c,:],n_fft=n_fft))
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

    IV =  get_foa_intensity_vectors(spec.T,mel_wts.T,n_mels)

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
        mfcc.append(librosa.feature.mfcc(raw[c,:], sr=fs, n_mfcc=40, n_fft = n_fft,hop_length=shift))
        chroma_stft.append(librosa.feature.chroma_stft(raw[c], sr=fs, n_fft = n_fft,hop_length=shift))

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

    return torch.from_numpy(np.concatenate((base,mfcc,chroma_stft,spectral_centroid,spectral_bandwidth,spectral_contrast,spectral_flatness),axis=-1))

parser = argparse.ArgumentParser()
parser.add_argument('-c','--config',type=str,required=True)
parser.add_argument('-m','--chkpt',type=str,required=True)
parser.add_argument('-i','--dir_input',type=str,required=True)
parser.add_argument('-o','--dir_output',type=str,required=True)
parser.add_argument('-n','--num_process',type=int,default=8)
parser.add_argument('-v','--version',type=str)
parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
args = parser.parse_args()

list_data = glob.glob(os.path.join(args.dir_input,"*.wav"))
print("len data : {}".format(len(list_data)))

## Parameters 
hp = HParam(args.config)
print('NOTE::Loading configuration :: ' + args.config)

device = args.device
torch.cuda.set_device(device)

num_epochs = 1
batch_size = 1
n_track = hp.model.n_track
version = args.version
dir_out = args.dir_output

os.makedirs(os.path.join(dir_out,version,"eval"),exist_ok=True)
os.makedirs(os.path.join(dir_out,version,"detail"),exist_ok=True)
os.makedirs(os.path.join(dir_out,version,"valid"),exist_ok=True)

## Model
model = EINV2(
    n_track = n_track,
    out_format=hp.model.format
).to(device)

model.load_state_dict(torch.load(args.chkpt, map_location=device))
model.share_memory()
model.eval()

detail = False

n_sample = 72000
hop_sample = int(n_sample/2)
fs=24000

shift=256
n_fft= 1024
threshold = [0.22323745, 0.24172139, 0.21299571, 0.22620639, 0.22098835, 0.76041244, 0.21146325, 0.20506471, 0.73513279, 0.2138928,  0.23605729, 0.24789842, 0.22673632]

def inference(path_data):
    with torch.no_grad():
        raw,_ = librosa.load(path_data,sr=fs,mono=False)

        n_channel,total_sample = raw.shape

        n_frame = 0
        idx_seg = 0

        len_total = int(np.ceil(total_sample/shift))

        output = torch.zeros(1,len_total,1,13,3).to(device)



        while idx_seg < total_sample :
            #print(idx_seg)
            len_check = idx_seg + n_sample

            # length check : cut
            if len_check <= total_sample : 
                idx_end = len_check
                seg = raw[:,idx_seg:idx_end]

                frame_seg = int(idx_seg/shift)
                frame_end = int(np.ceil((idx_end)/shift))
            # length check : pad
            else : 
                shortage = len_check - total_sample
                idx_end = total_sample
                seg = np.zeros((n_channel,n_sample))
                seg[:,:n_sample - shortage] = raw[:,idx_seg:idx_end]

                frame_seg = int(idx_seg/shift)
                frame_end = int(np.ceil((idx_end)/shift))
            

            data=preprocess(seg).float().to(device)
            data=torch.unsqueeze(data,dim=0)

        
            tmp_output = model(data)

            idx_start = int(idx_seg/shift)
            idx_new = int((idx_seg + hop_sample)/shift)
            len_over = idx_new - idx_start
            len_end = tmp_output.shape[1]

            if idx_new > len_total : 
                break


            #print("{} {} {}".format(idx_start,len_over,idx_new))
            output[:,idx_start:idx_start + len_over,:,:,:] += tmp_output[:,:len_over,:,:,:]
            output[:,idx_start:idx_start + len_over,:,:,:] /= 2

            if idx_new + len_end > len_total : 
                len_end = len_total - idx_new

            output[:,idx_new:idx_new+(len_end-len_over),:,:,:] = tmp_output[:,len_over:len_end,:,:,:]

            idx_seg +=hop_sample

        #print(output.shape)


        # save csv
        name_data = path_data.split('/')[-1]
        id_data = name_data.split('.')[0]
        path_output = args.dir_output + "/" + id_data + ".csv"

        #print("{} {}".format(idx_seg,output.shape))
        #print(path_output)
        #print(len(df_label.index))

        df_label= mACCDOA2label(output[0].cpu().detach(),detail=False,threshold=threshold,valid=False)
        df_label.to_csv(os.path.join(dir_out,version,"eval",id_data+".csv"),header=False, index=False)

        df_label= mACCDOA2label(output[0].cpu().detach(),detail=False,threshold=threshold,valid=True)
        df_label.to_csv(os.path.join(dir_out,version,"valid",id_data+".csv"),header=False, index=False)

        df_label= mACCDOA2label(output[0].cpu().detach(),detail=True,threshold=threshold,valid=False)
        df_label.to_csv(os.path.join(dir_out,version,"detail",id_data+".csv"),header=False, index=False)

if __name__ == '__main__':
    set_start_method('spawn')
    processes = []
    batch_for_each_process = np.array_split(list_data,args.num_process)

    for path in tqdm(list_data) : 
        inference(path)

    """
    for worker in range(args.num_process):
        p = mp.Process(target=inference, args=(batch_for_each_process[worker][:],) )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    """


 
