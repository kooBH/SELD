import torch
import numpy as np
import pandas as pd

"""
Convert Label to multi-ACCDOA format

Azimuth and elevation angles are given in degrees, rounded to the closest integer value, with azimuth and elevation being zero at the front, azimuth ϕ∈[−180∘,180∘], and elevation θ∈[−90∘,90∘]. Note that the azimuth angle is increasing counter-clockwise (ϕ=90∘ at the left).

args : 
    label : [n_frame, max_n_target,3(class,azimuth,elevation)]
return :
    mACCODA : [n_frame,n_track,n_class,3(x,y,z)]
"""
def label2mACCDOA(label,n_class=13,n_track=2):

    n_frame = label.shape[0]
    max_n_target = label.shape[1]

    mACCDOA = torch.zeros((n_frame,n_track,n_class,3))

    ## convert polar to cartesian
    for i in range(n_frame) : 
        j = 0
        # for all target
        while j < max_n_target and label[i,j,0] != -1:
            if label[i,j,0] == -1 :
                break
            aiz_rad =  label[i,j,1]*torch.pi/180.
            ele_rad =  label[i,j,2]*torch.pi/180.

            x = torch.cos(aiz_rad)*torch.cos(ele_rad)
            y = torch.sin(aiz_rad)*torch.cos(ele_rad)
            z = torch.sin(ele_rad)

            category = int(label[i,j,0])

            # iter all tracks in target category
            k = 0
            while  k <= n_track :
                # more than n_track, replace previous label randomly
                if k == n_track : 
                    mACCDOA[i,np.random.randint(n_track),category,:]=torch.tensor(([x,y,z]))
                    break

                # current track is empty
                if torch.norm(mACCDOA[i,k,category,:]) == 0 : 
                    mACCDOA[i,k,category,:]=torch.tensor(([x,y,z]))
                    break
                k+=1
            j += 1

    return mACCDOA

def mACCDOA2label(mACCDOA,shift_in=256,shift_out=2400,threshold=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],detail=False,valid=False) : 
    n_frame = mACCDOA.shape[0]
    n_track = mACCDOA.shape[1]
    n_class = mACCDOA.shape[2]

    #print("label::mACCDOA2label:n_frame {}".format(n_frame))


    if detail : 
        ret_label = pd.DataFrame(columns=["idx","class","azimuth","elevation","0","1","2","3","4","5","6","7","8","9","10","11","12"])
    elif valid : 
        ret_label = pd.DataFrame(columns=["idx","class","order","x","y","z"])
    else :
        ret_label = pd.DataFrame(columns=["idx","class","azimuth","elevation"])

    ratio =  shift_in/shift_out
    #print(ratio)
    
    prev_idx = -1
    # conversion
    for i_frame in range(n_frame) : 
        cur_idx_start = int(i_frame*ratio)
        cur_idx_end = int(i_frame*ratio)
        
        if prev_idx == cur_idx_start:
            continue
        else :
            prev_idx = cur_idx_start
        
        active = 0
        
        for i_track in range(n_track) : 
            if detail : 
                for i_idx in range(cur_idx_start,cur_idx_end+1): 

                    ret_label.loc[len(ret_label.index)] = [i_idx, 0, 0, 0,
                        float(torch.norm(mACCDOA[i_frame,i_track,0,:])),
                        float(torch.norm(mACCDOA[i_frame,i_track,1,:])),
                        float(torch.norm(mACCDOA[i_frame,i_track,2,:])),
                        float(torch.norm(mACCDOA[i_frame,i_track,3,:])),
                        float(torch.norm(mACCDOA[i_frame,i_track,4,:])),
                        float(torch.norm(mACCDOA[i_frame,i_track,5,:])),
                        float(torch.norm(mACCDOA[i_frame,i_track,6,:])),
                        float(torch.norm(mACCDOA[i_frame,i_track,7,:])),
                        float(torch.norm(mACCDOA[i_frame,i_track,8,:])),
                        float(torch.norm(mACCDOA[i_frame,i_track,9,:])),
                        float(torch.norm(mACCDOA[i_frame,i_track,10,:])),
                        float(torch.norm(mACCDOA[i_frame,i_track,11,:])),
                        float(torch.norm(mACCDOA[i_frame,i_track,12,:]))]
                continue

            for i_class in range(n_class):
                # active label
                if torch.norm(mACCDOA[i_frame,i_track,i_class,:]) > threshold[i_class] :
                
                    x, y, z = mACCDOA[i_frame,i_track,i_class,:]

                    # in degrees
                    azimuth = int(np.round((np.arctan2(y, x) * 180 / np.pi).numpy()))
                    elevation = int(np.round((np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi).numpy()))
                    
                    for i_idx in range(cur_idx_start,cur_idx_end+1): 
                        if valid : 
                            ret_label.loc[len(ret_label.index)] = [int(i_idx), i_class,active ,x.item(), y.item(),z.item()]
                        else :
                            ret_label.loc[len(ret_label.index)] = [i_idx, i_class, azimuth, elevation]
                    active += 1

    return ret_label




