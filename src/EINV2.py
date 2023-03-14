import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

def init_layer(layer, nonlinearity='leaky_relu'):
    '''
    Initialize a layer
    '''
    #pdb.set_trace()
    classname = layer.__class__.__name__
    if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(layer.weight, 1.0, 0.02)
        nn.init.constant_(layer.bias, 0.0)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, 
                kernel_size=(3,3), stride=(1,1), padding=(1,1),
                dilation=1, bias=False):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                    out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=out_channels, 
                    out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.init_weights()
        
    def init_weights(self):
        for layer in self.double_conv:
            init_layer(layer)
        
    def forward(self, x):
        #pdb.set_trace()
        x = self.double_conv(x)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, pos_len, d_model=512, pe_type='t', dropout=0.0):
        """ Positional encoding using sin and cos

        Args:
            pos_len: positional length
            d_model: number of feature maps
            pe_type: 't' | 'f' , time domain, frequency domain
            dropout: dropout probability
        """
        super().__init__()
        
        self.pe_type = pe_type
        pe = torch.zeros(pos_len, d_model)
        pos = torch.arange(0, pos_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = 0.1 * torch.sin(pos * div_term)
        pe[:, 1::2] = 0.1 * torch.cos(pos * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2) # (N, C, T)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        # x is (N, C, T, F) or (N, C, T) or (N, C, F)
        if x.ndim == 4:
            if self.pe_type == 't':
                pe = self.pe.unsqueeze(3)
                x += pe[:, :, :x.shape[2]]
            elif self.pe_type == 'f':
                pe = self.pe.unsqueeze(2)
                x += pe[:, :, :, :x.shape[3]]
        elif x.ndim == 3:
            x += self.pe[:, :, :x.shape[2]]
        return self.dropout(x)

class mACCDOA(nn.Module):
    def __init__(self,n_track=2,n_class=13):
        super().__init__()
        
        self.n_track = n_track
        self.n_class = n_class
        
        self.activation = nn.Tanh()
        self.accdoa = nn.Linear(1024,n_track*n_class*3)
        
        init_layer(self.accdoa)
        
    def forward(self, f_sed,f_doa): 
        #print("f_sed f_doa {} {}".format(f_sed.shape,f_doa.shape))
        
        x = torch.cat((f_sed,f_doa),dim=2)
        x = self.activation(self.accdoa(x))
        
        #print("x {}".format(x.shape))
        return x.reshape(x.shape[0],x.shape[1],self.n_track,self.n_class,3)

class SED_DOA(nn.Module):
    def __init__(self, n_track=3,n_class=13):
        super().__init__()
        
        self.n_track = n_track
        
        if n_track < 1 : 
            raise Exception("ERORR:SED_DOA:: {} < 1".format(n_track))
        
        self.sed=[]
        self.doa=[]
        
        self.sed_act = nn.Sigmoid()
        self.doa_act = nn.Tanh()
        
        
        for i in range(n_track) :
            self.sed.append(nn.Linear(512, n_class, bias=True))
            self.doa.append(nn.Linear(512, 3, bias=True))
        
        for i in range(n_track) : 
            init_layer(self.sed[i])
            init_layer(self.doa[i])

                
    def forward(self, f_sed, f_doa):
        
        sed = self.sed[0](f_sed)
        doa = self.doa[0](f_doa)
        
        sed = torch.unsqueeze(sed,2)
        doa = torch.unsqueeze(doa,2)
        
        #print("SED_DOA : {}".format(sed.shape))
        
        for i in range(1,self.n_track) : 
            t_sed = self.sed[i](f_sed)
            t_doa = self.doa[i](f_doa)
            
            t_sed = torch.unsqueeze(t_sed,2)
            t_doa = torch.unsqueeze(t_doa,2)
            
            sed = torch.cat((sed,t_sed),2)
            doa = torch.cat((doa,t_doa),2)
              
            #print("SED_DOA : {}".format(sed.shape))
            
        return {"sed":sed,"doa":doa}

"""
Time domain of T -> int(T/4)
"""

class EINV2(nn.Module):
    def __init__(self,
    in_channels=4,
    n_track=1,
    out_format="mACCDOA",
    n_transformer_head = 8,
    n_transformer_layer=2,
    dropout=0.0
    ):
        self.out_format = out_format
        super().__init__()
        self.pe_enable = True  # Ture | False

        self.dropout = nn.Dropout(p=dropout)
       
        self.downsample_ratio = 2 ** 2
        self.sed_conv_block1 = nn.Sequential(
            DoubleConv(in_channels=4, out_channels=64),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.sed_conv_block2 = nn.Sequential(
            DoubleConv(in_channels=64, out_channels=128),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.sed_conv_block3 = nn.Sequential(
            DoubleConv(in_channels=128, out_channels=256),
            nn.AvgPool2d(kernel_size=(1, 2)),
        )
        self.sed_conv_block4 = nn.Sequential(
            DoubleConv(in_channels=256, out_channels=512),
            nn.AvgPool2d(kernel_size=(1, 2)),
        )

        self.doa_conv_block1 = nn.Sequential(
            DoubleConv(in_channels=in_channels, out_channels=64),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.doa_conv_block2 = nn.Sequential(
            DoubleConv(in_channels=64, out_channels=128),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.doa_conv_block3 = nn.Sequential(
            DoubleConv(in_channels=128, out_channels=256),
            nn.AvgPool2d(kernel_size=(1, 2)),
        )
        self.doa_conv_block4 = nn.Sequential(
            DoubleConv(in_channels=256, out_channels=512),
            nn.AvgPool2d(kernel_size=(1, 2)),
        )

        self.stitch = nn.ParameterList([
           nn.Parameter(torch.FloatTensor(64, 2, 2).uniform_(0.1, 0.9)),
           nn.Parameter(torch.FloatTensor(128, 2, 2).uniform_(0.1, 0.9)),
           nn.Parameter(torch.FloatTensor(256, 2, 2).uniform_(0.1, 0.9)),
        ])

        if self.pe_enable:
            self.sed_pe = PositionalEncoding(pos_len=100, d_model=512, pe_type='t', dropout=dropout)
            self.doa_pe = PositionalEncoding(pos_len=100, d_model=512, pe_type='t', dropout=dropout)

        decoder_layer1 = nn.TransformerDecoderLayer(
            d_model=512, 
            nhead=n_transformer_head
        )
        decoder_layer2 = nn.TransformerDecoderLayer(d_model=512,
         nhead=n_transformer_head
         )
        self.trans_decoder_sed_doa = nn.TransformerDecoder(decoder_layer1, num_layers=n_transformer_layer)
        self.trans_decoder_doa_sed = nn.TransformerDecoder(decoder_layer2, num_layers=n_transformer_layer)

        self.fc_sed_track1 = nn.Linear(512, 12, bias=True)
        self.fc_sed_track2 = nn.Linear(512, 12, bias=True)
        self.fc_sed_track3 = nn.Linear(512, 12, bias=True)
        self.fc_doa_track1 = nn.Linear(512, 3, bias=True)
        self.fc_doa_track2 = nn.Linear(512, 3, bias=True)
        self.fc_doa_track3 = nn.Linear(512, 3, bias=True)
        self.final_act_sed = nn.Sequential() # nn.Sigmoid()
        self.final_act_doa = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.init_weight()

        if out_format=="SED_DOA":
            self.format = SED_DOA(
                n_track=n_track
            )
        elif out_format == "mACCDOA":
            self.format = mACCDOA(
                n_track=n_track
            )
        else : 
            raise Exception("ERROR::EINV2::unknown format {}".format(out_format))
    
    def init_weight(self):

        init_layer(self.fc_sed_track1)
        init_layer(self.fc_sed_track2)
        init_layer(self.fc_sed_track3)
        init_layer(self.fc_doa_track1)
        init_layer(self.fc_doa_track2)
        init_layer(self.fc_doa_track3)

    def forward(self, x):
        """
        x: (n_batch, n_channel,n_time, n_feature)
        """

        n_time = x.shape[2]

        #pdb.set_trace()
        x_sed = x[:, :4] #[32,4,161,256]
        x_doa = x
        #x_doa = x[:, 4:]        #[32,7,161,256]   
        #pdb.set_trace()
        
        # CNN
        x_sed = self.sed_conv_block1(x_sed) #[32,64,80,128]
        x_doa = self.doa_conv_block1(x_doa) #[32,64,80,128]
        x_sed = torch.einsum('c, nctf -> nctf', self.stitch[0][:, 0, 0], x_sed) + \
           torch.einsum('c, nctf -> nctf', self.stitch[0][:, 0, 1], x_doa)
        x_doa = torch.einsum('c, nctf -> nctf', self.stitch[0][:, 1, 0], x_sed) + \
           torch.einsum('c, nctf -> nctf', self.stitch[0][:, 1, 1], x_doa)

        x_sed = self.sed_conv_block2(x_sed) #[32,128,40,64] 
        x_doa = self.doa_conv_block2(x_doa) #[32,128,40,64]
        x_sed = torch.einsum('c, nctf -> nctf', self.stitch[1][:, 0, 0], x_sed) + \
           torch.einsum('c, nctf -> nctf', self.stitch[1][:, 0, 1], x_doa)
        x_doa = torch.einsum('c, nctf -> nctf', self.stitch[1][:, 1, 0], x_sed) + \
           torch.einsum('c, nctf -> nctf', self.stitch[1][:, 1, 1], x_doa)

        x_sed = self.sed_conv_block3(x_sed) #[32,256,40,32]
        x_doa = self.doa_conv_block3(x_doa) #[32,256,40,32]
        x_sed = torch.einsum('c, nctf -> nctf', self.stitch[2][:, 0, 0], x_sed) + \
           torch.einsum('c, nctf -> nctf', self.stitch[2][:, 0, 1], x_doa)
        x_doa = torch.einsum('c, nctf -> nctf', self.stitch[2][:, 1, 0], x_sed) + \
           torch.einsum('c, nctf -> nctf', self.stitch[2][:, 1, 1], x_doa)

        x_sed = self.sed_conv_block4(x_sed) #[32,512,40,16]
        x_doa = self.doa_conv_block4(x_doa) #[32,512,40,16]

        x_sed = x_sed.mean(dim=3) # (N, C, T)
        x_doa = x_doa.mean(dim=3) # (N, C, T) [32,512,40]

        # Transformer
        if self.pe_enable:
            x_doa = self.sed_pe(x_doa)
            x_sed = self.doa_pe(x_sed)
        x_sed = x_sed.permute(2, 0, 1) # (T, N, C)
        x_doa = x_doa.permute(2, 0, 1) # (T, N, C)
        
        x_sed_trans = self.trans_decoder_sed_doa(x_sed, x_doa)
        x_doa_trans = self.trans_decoder_doa_sed(x_doa, x_sed)

        ## Iterpolate
        if self.out_format == "mACCDOA" : 
            x_sed_trans = torch.nn.functional.interpolate(x_sed_trans.permute(1,2,0),
            size = (n_time),
            mode="linear",
            align_corners=False
            )
            x_doa_trans = torch.nn.functional.interpolate(x_doa_trans.permute(1,2,0),size = (n_time),
            mode="linear",
            align_corners=False
            )
        else :
            raise Exception("ERORR::EINV2::Unimplemented out_format : {}".format(self.out_format))
 
        ## 
        x_sed_trans = x_sed_trans.transpose(1, 2)
        x_doa_trans = x_doa_trans.transpose(1, 2)


        output = self.format(x_sed_trans,x_doa_trans)




        
        return output
