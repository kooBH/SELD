import numpy as np
import glob,os
from scipy import io

path = "/home/nas3/DB/DCASE2022/TAU-SRIR_DB/rirdata.mat"
root_out = "/home/nas3/DB/DCASE2022/TAU-SRIR_DB_DOA"

ririnfo = io.loadmat(path)

n_room = len(ririnfo["rirdata"][0,0]["room"]['name'])

print("n_room : {}".format(n_room))
print(ririnfo["rirdata"][0,0]["room"]['rirs'].shape)

os.makedirs(root_out,exist_ok=True)

for idx in range(n_room) : 
    shape_rir = ririnfo["rirdata"][0,0]["room"]['rirs'][idx][0].shape

    id_target = "rirs_{}".format(ririnfo["rirdata"][0,0]["room"]['name'][idx][0][0])

    print("{} {}".format(idx,id_target))

    for idx2 in range(shape_rir[0]*shape_rir[1]) :
        # NOTE : idenx 
        i = int(idx2/shape_rir[1])
        j = int(idx2%shape_rir[1])

        DOA = ririnfo["rirdata"][0,0]["room"]['rirs'][idx][0][i,j]['doa_xyz']

        np.save(
            os.path.join(root_out,"{}_{}_{}.npy".format(id_target,j,i)),
            DOA
        )

