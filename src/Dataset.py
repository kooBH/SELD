import os, glob
import torch
import numpy as np



class Dataset(torch.utils.data.Dataset):
    def __init__(self, 
    root,
    preprocessed=True,
    specaug=False, 
    specaug_ratio=0.3
    ):
        self.root = root
        self.preprocessed = preprocessed

        self.specaug = specaug
        self.specaug_ratio = specaug_ratio

        if preprocessed : 
            self.list_data = glob.glob(os.path.join(root,"*.pt"))
        else :
            raise Exception("ERROR:Dataset::unimplemented for preprocessed == {}".format(preprocessed))

    def __getitem__(self, index):
        path_data = self.list_data[index]

        data = torch.load(path_data)

        n_feature = data["data"].shape[2]

        if self.specaug : 
            # index of spec augmentation starting point
            idx_f_s = np.random.randint(
                low=0,
                high=n_feature)
            # length of spec augmentation
            idx_f_l = np.random.randint(
                low=0,
                high=int(n_feature * self.specaug_ratio))
            idx_f_e = np.min((idx_f_s + idx_f_l, n_feature))
            data["data"][:,:,idx_f_s : idx_f_e] = 0

        # TODO : TEMP
        data["data"]=data["data"][:,:280,:]
        data["label"]=data["label"][:280,:,:]

        return data

    def __len__(self):
        return len(self.list_data)


