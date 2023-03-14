import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

eps = torch.finfo(torch.float32).eps

class MSELoss:
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        self.name = 'loss_MSE'
        if self.reduction != 'PIT':
            self.loss = nn.MSELoss(reduction='mean')
        else:
            self.loss = nn.MSELoss(reduction='none')
    
    def calculate_loss(self, pred, target):
        #pdb.set_trace()
        if self.reduction != 'PIT':
            return self.loss(pred, target)
        else:
            return self.loss(pred, target).mean(dim=tuple(range(2, pred.ndim)))


class BCEWithLogitsLoss:
    def __init__(self, reduction='mean', pos_weight=None):
        self.reduction = reduction
        self.name = 'loss_BCEWithLogits'
        if self.reduction != 'PIT':
            self.loss = nn.BCEWithLogitsLoss(reduction=self.reduction, pos_weight=pos_weight)
        else:
            self.loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    
    def calculate_loss(self, pred, target):
        #pdb.set_trace()
        if self.reduction != 'PIT':
            return self.loss(pred, target)
        else:
            return self.loss(pred, target).mean(dim=tuple(range(2, pred.ndim)))

class Losses:
    def __init__(self):
        
        self.beta =  0.5
        self.charlie = 0.5

        self.losses = [BCEWithLogitsLoss(reduction='mean'), MSELoss(reduction='mean')]
        self.losses_pit = [BCEWithLogitsLoss(reduction='PIT'), MSELoss(reduction='PIT')]
        
        self.names = ['loss_all'] + [loss.name for loss in self.losses]


    def calculate(self, PIT_type = "tPIT",pred, target, epoch_it=0):
        #pdb.set_trace()
        if 'PIT' not in PIT_type:
            updated_target = target
            loss_sed = self.losses[0].calculate_loss(pred['sed'], updated_target['sed'])
            loss_doa = self.losses[1].calculate_loss(pred['doa'], updated_target['doa'])
        elif PIT_type == 'tPIT':
            loss_sed, loss_doa = self.tPIT(pred, target)
            #loss_sed, loss_doa, updated_target = self.tPIT(pred, target)

        loss_all = self.beta * loss_sed + (1 - self.beta) * loss_doa
        #loss_all = self.beta * loss_sed + self.charlie * loss_doa
        losses_dict = {
            'all': loss_all,
            'sed': loss_sed,
            'doa': loss_doa,
        }

        return losses_dict 

    def tPIT(self, pred, target):
        """Frame Permutation Invariant Training for 6 possible combinations"""
        
        #pdb.set_trace()
        target_flipped132 = {
            'sed': torch.stack((target['sed'][:,:,0,:],target['sed'][:,:,2,:],target['sed'][:,:,1,:]),dim=2),
            'doa': torch.stack((target['doa'][:,:,0,:],target['doa'][:,:,2,:],target['doa'][:,:,1,:]),dim=2)
        }
        target_flipped213 = {
            'sed': torch.stack((target['sed'][:,:,1,:],target['sed'][:,:,0,:],target['sed'][:,:,2,:]),dim=2),
            'doa': torch.stack((target['doa'][:,:,1,:],target['doa'][:,:,0,:],target['doa'][:,:,2,:]),dim=2)
        }
        target_flipped231 = {
            'sed': torch.stack((target['sed'][:,:,1,:],target['sed'][:,:,2,:],target['sed'][:,:,0,:]),dim=2),
            'doa': torch.stack((target['doa'][:,:,1,:],target['doa'][:,:,2,:],target['doa'][:,:,0,:]),dim=2)
        }
        target_flipped312 = {
            'sed': torch.stack((target['sed'][:,:,2,:],target['sed'][:,:,0,:],target['sed'][:,:,1,:]),dim=2),
            'doa': torch.stack((target['doa'][:,:,2,:],target['doa'][:,:,0,:],target['doa'][:,:,1,:]),dim=2)
        }
        target_flipped321 = {
            'sed': torch.stack((target['sed'][:,:,2,:],target['sed'][:,:,1,:],target['sed'][:,:,0,:]),dim=2),
            'doa': torch.stack((target['doa'][:,:,2,:],target['doa'][:,:,1,:],target['doa'][:,:,0,:]),dim=2)
        }

        loss_sed1 = self.losses_pit[0].calculate_loss(pred['sed'], target['sed'])
        loss_doa1 = self.losses_pit[1].calculate_loss(pred['doa'], target['doa'])
        loss_sed2 = self.losses_pit[0].calculate_loss(pred['sed'], target_flipped132['sed'])
        loss_doa2 = self.losses_pit[1].calculate_loss(pred['doa'], target_flipped132['doa'])
        loss_sed3 = self.losses_pit[0].calculate_loss(pred['sed'], target_flipped213['sed'])
        loss_doa3 = self.losses_pit[1].calculate_loss(pred['doa'], target_flipped213['doa'])
        loss_sed4 = self.losses_pit[0].calculate_loss(pred['sed'], target_flipped231['sed'])
        loss_doa4 = self.losses_pit[1].calculate_loss(pred['doa'], target_flipped231['doa'])
        loss_sed5 = self.losses_pit[0].calculate_loss(pred['sed'], target_flipped312['sed'])
        loss_doa5 = self.losses_pit[1].calculate_loss(pred['doa'], target_flipped312['doa'])
        loss_sed6 = self.losses_pit[0].calculate_loss(pred['sed'], target_flipped321['sed'])
        loss_doa6 = self.losses_pit[1].calculate_loss(pred['doa'], target_flipped321['doa'])

        loss1 = loss_sed1 + loss_doa1
        loss2 = loss_sed2 + loss_doa2
        loss3 = loss_sed3 + loss_doa3
        loss4 = loss_sed4 + loss_doa4
        loss5 = loss_sed5 + loss_doa5
        loss6 = loss_sed6 + loss_doa6
        
        #pdb.set_trace()
        min_loss = torch.min(torch.min(torch.min(torch.min(torch.min(loss1,loss2),loss3),loss4),loss5),loss6)
        loss_sed = (loss_sed1 * (loss1 == min_loss) + loss_sed2 * (loss2 == min_loss) +  loss_sed3 * (loss3 == min_loss) + loss_sed4 * (loss4 == min_loss) + loss_sed5 * (loss5 == min_loss) + loss_sed6 * (loss6 == min_loss)).mean()
        loss_doa = (loss_doa1 * (loss1 == min_loss) + loss_doa2 * (loss2 == min_loss) +  loss_doa3 * (loss3 == min_loss) + loss_doa4 * (loss4 == min_loss) + loss_doa5 * (loss5 == min_loss) + loss_doa6 * (loss6 == min_loss)).mean()
        return loss_sed, loss_doa