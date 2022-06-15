import torch
import torch.nn as nn
import torch.nn.functional as F

class TDNN(nn.Module):
    
    def __init__(
        self, 
        input_dim=23, 
        output_dim=512,
        context_size=5,
        stride=1,
        dilation=1,
        batch_norm=False,
        dropout_p=0.2
    ):
        super().__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
      
        self.kernel = nn.Linear(input_dim*context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)
        
    def forward(self, x):
        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1)
        x = F.unfold(
            x, 
            (self.context_size, self.input_dim), 
            stride=(1,self.input_dim), 
            dilation=(self.dilation,1)
        )
        # N, input_dim*context_size, new_t = x.shape
        x = x.transpose(1,2)
        x = self.kernel(x.float())
        x = self.nonlinearity(x)
        if self.dropout_p:
            x = self.drop(x)
        if self.batch_norm:
            x = x.transpose(1,2)
            x = self.bn(x)
            x = x.transpose(1,2)
        return x


class XVector(nn.Module):
    def __init__(self, embedding_dim, n_mels):
        super().__init__()
        self.tdnn1 = TDNN(input_dim=n_mels, output_dim=512, context_size=5, dilation=1,dropout_p=0.5)
        self.tdnn2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=2,dropout_p=0.5)
        self.tdnn3 = TDNN(input_dim=512, output_dim=512, context_size=2, dilation=3,dropout_p=0.5)
        self.tdnn4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1,dropout_p=0.5)
        self.tdnn5 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1,dropout_p=0.5)
        #### Frame levelPooling
        self.segment6 = nn.Linear(1024, 512)
        self.segment7 = nn.Linear(512, embedding_dim)

    def forward(self, x):
        x = x[:,0,:,:]
        x = x.transpose(1,2)

        tdnn1_out = self.tdnn1(x)
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        tdnn5_out = self.tdnn5(tdnn4_out)
        ### Stat Pool
        mean = torch.mean(tdnn5_out,1)
        std = torch.std(tdnn5_out,1)
        stat_pooling = torch.cat((mean,std),1)
        segment6_out = self.segment6(stat_pooling)
        x_vec = self.segment7(segment6_out)
        return x_vec