import numpy as np
import librosa
import pandas as pd
import soundfile as sf
import scipy

list_rir = [
"rirs_01_bomb_shelter_0_0",
"rirs_02_gym_0_0",
"rirs_03_pb132_0_0",
"rirs_04_pc226_0_0",
"rirs_05_sa203_0_0",
"rirs_06_sc203_0_0",
"rirs_08_se203_0_0",
"rirs_09_tb103_0_0",
"rirs_10_tc352_0_0"
]

root_rir   =  "/home/nas3/DB/DCASE2022/TAU-SRIR_DB_split/"

path_sample_wav_1 = "/home/data/kbh/DCASE2022/raw/0/FSD50K_dev_104707.wav"



raw_1,_ = librosa.load(path_sample_wav_1,sr=24000)

import gpuRIR
gpuRIR.activateMixedPrecision(False)
gpuRIR.activateLUT(True)

for i_rir in list_rir :
    rir_1 = np.load(root_rir + i_rir + ".npy")
    # half cut
    rir_1 =  rir_1[:,:,:int(rir_1.shape[2]/3)]

    print(rir_1.shape)
    mix = gpuRIR.simulateTrajectory(raw_1, rir_1.astype(np.float32))

    print("{} {}".format(rir_1.shape,mix.shape))

    sf.write(i_rir + "_onethrid.wav",mix,24000)