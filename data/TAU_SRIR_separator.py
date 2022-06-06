import os,glob
# MATLAB > 7.3 used h5 format
import h5py
import numpy as np

# utils
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


root_path = "/home/nas3/DB/DCASE2022/TAU-SRIR_DB"
root_out = "/home/nas3/DB/DCASE2022/TAU-SRIR_DB_split"

list_target = [x for x in glob.glob(os.path.join(root_path,"rirs*"))]

os.makedirs(root_out,exist_ok=True)

def split(idx):
    path_target = list_target[idx]
    name_target = path_target.split('/')[-1]
    id_target = name_target.split('.')[0]

    f = h5py.File(path_target,'r')

    s = f['rirs']['foa'].shape
    for idx in  range(s[0]*s[1]) : 
        i = int(idx/s[1])
        j = int(idx%s[1])

        rir = np.array(f[ f['rirs']['foa'][i,j] ])

        np.save(
            os.path.join(
                root_out,"{}_{}_{}.npy".format(id_target,i,j)
            )
            ,rir)

if __name__=='__main__': 
    cpu_num = cpu_count()

    arr = list(range(len(list_target)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(split, arr), total=len(arr),ascii=True,desc='split'))
