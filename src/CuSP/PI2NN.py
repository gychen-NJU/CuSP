import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import os

from collections import OrderedDict
from torch.optim.lr_scheduler import StepLR

from .me_forward import MEForward

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y
    
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attn_probs = self.softmax(attn_scores)
        out = torch.matmul(attn_probs, V)
        return out + x  
    
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, use_batchnorm=True, use_attention=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim) if use_batchnorm else nn.Identity()
        self.activation = nn.GELU()
        self.shortcut = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.attention = SelfAttention(out_dim) if use_attention else nn.Identity()

        nn.init.xavier_uniform_(self.linear.weight)
        if use_batchnorm:
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)
        if isinstance(self.shortcut, nn.Linear):
            nn.init.xavier_uniform_(self.shortcut.weight)

    def forward(self, x):
        identify = self.shortcut(x)
        x = self.linear(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.attention(x)
        return x+identify

class ConvResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, use_batchnorm=True, groups=1, use_attention=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, groups=groups),
            nn.BatchNorm1d(out_channels) if use_batchnorm else nn.Identity(),
            nn.GELU(),
            ChannelAttention(out_channels) if use_attention else nn.Identity()
        )
        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, groups=groups),
                nn.BatchNorm1d(out_channels) if use_batchnorm else nn.Identity()
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)
        

class FCN(nn.Module):
    def __init__(self, layers, use_attention=True, use_batchnorm=True, use_residual=True):
        super(FCN, self).__init__()
        self.depth = len(layers) - 1
        self.activation = nn.GELU()
        self.use_attention = use_attention
        self.use_batchnorm = use_batchnorm
        self.use_residual = use_residual

        layer_list = []
        for i in range(self.depth-1):
            if self.use_residual and (0<i<self.depth-2):
                block = ResidualBlock(
                    layers[i], layers[i+1],
                    use_batchnorm=self.use_batchnorm,
                    use_attention=self.use_attention
                )
                layer_list.append(('res_block_%d' % i, block))
            else:
                layer_list.append(('layer_%d' % i, nn.Linear(layers[i], layers[i+1])))
                torch.nn.init.xavier_uniform_(layer_list[-1][1].weight)
                if self.use_batchnorm:
                    layer_list.append(('bn_%d' % i, nn.BatchNorm1d(layers[i+1])))
                layer_list.append(('activation_%d' % i, self.activation))
                if self.use_attention:
                    layer_list.append(('attn_%d' % i, SelfAttention(layers[i+1])))

        layer_list.append(('layer_%d' % (self.depth-1), nn.Linear(layers[-2], layers[-1])))
        torch.nn.init.xavier_uniform_(layer_list[-1][1].weight)
        layer_list.append(('sigmoid', nn.Sigmoid()))

        self.layers = nn.Sequential(OrderedDict(layer_list))

    def forward(self, x):
        return self.layers(x)

class InversionNet(nn.Module):
    def __init__(self, layers=[3 * 32, 8], use_attention=True, use_batchnorm=True, use_residual=True,use_kqv=False):
        super(InversionNet, self).__init__()
        self.layers = layers
        self.conv1d = nn.Conv1d
        self.pool = nn.MaxPool1d(2, 2)
        self.flatten = nn.Flatten()
        self.fcn = FCN(layers, use_attention=use_kqv, use_batchnorm=use_batchnorm, use_residual=use_residual)
        
        conv_blocks = []
        channels = [4, 16, 16, 16, 32, 32, 32]
        for i in range(6):
            if use_residual:
                if i in [1,2,4,5]:
                    block = ConvResidualBlock(
                        in_channels=channels[i],
                        out_channels=channels[i+1],
                        kernel_size=3,
                        padding=1,
                        groups = 4 if i!=0 else 1,
                        use_batchnorm=use_batchnorm,
                        use_attention=use_attention and (i+1) % 2 == 0
                    )
                else:
                    block = nn.Sequential(
                        nn.Conv1d(
                            channels[i],channels[i+1],
                            kernel_size=3,
                            padding = 0,
                            groups = 4 if i!=0 else 1,
                        ),
                        nn.BatchNorm1d(channels[i+1]),
                        nn.GELU(),
                        ChannelAttention(channels[i+1]) if use_attention and (i+1) % 2 == 0 else nn.Identity()
                    )
                conv_blocks.append(('res_block_%d' % (i+1), block))
            else:
                conv = self.conv1d(
                    in_channels=channels[i], 
                    out_channels=channels[i+1], 
                    kernel_size=3,
                    padding=1 if i in [1,2,4,5] else 0,
                    groups = 4 if i!=0 else 1
                )
                conv_blocks.append(('conv%d' % (i+1), conv))
                if use_batchnorm:
                    conv_blocks.append(('bn%d' % (i+1), nn.BatchNorm1d(channels[i+1])))
                conv_blocks.append(('relu%d' % (i+1), nn.ReLU()))
                if use_attention and (i+1) % 2 == 0:
                    conv_blocks.append(('channel_attn%d' % (i+1), ChannelAttention(channels[i+1])))

        net = [
            *conv_blocks,
            # ('pool', self.pool),
            ('flatten', self.flatten),
            ('fcn', self.fcn)
        ]
        self.net = nn.Sequential(OrderedDict(net))

    def forward(self, x):
        return self.net(x)
    
class PI2NN(nn.Module):
    def __init__(self,
                 hidden_layers=[64]*4,
                 input_size=6,
                 output_size=8,
                 device=None,
                 **kwargs
                 ):
        """
        PI2NN: Physics-Informed Polarization Inversion Network
        
        Input:
            hidden_layers: list of hidden layers
            input_size: input size
            output_size: output size
            device: device
            **kwargs: keyword arguments
        """
        super(PI2NN, self).__init__()
        self.device = torch.device(device) if (device is not None) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layers = [(input_size-4)*32] + hidden_layers + [output_size]
        use_batchnorm = kwargs.get('use_batchnorm', True)
        use_attention = kwargs.get('use_attention', True)
        use_residual = kwargs.get('use_residual', False)
        # self.net = RNetPlus().to(self.device)
        self.net = InversionNet(layers=self.layers, use_batchnorm=use_batchnorm, use_attention=use_attention, use_residual=use_residual).to(self.device)
        self._set_bounds(kwargs.get('bounds',None))
        self._set_norm_scale(kwargs.get('norm_scale',None))

        self.criterion = nn.MSELoss()
        # self.optimizer = torch.optim.NAdam(self.net.parameters(), lr=1.e-3, betas=(0.9, 0.999), weight_decay=1.e-4)
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     self.optimizer,
        #     max_lr=1.e-2,
        #     steps_per_epoch=100,
        #     epochs=1000,
        #     pct_start=0.1,
        #     div_factor=100,
        #     final_div_factor=10000,
        #     anneal_strategy='cos',
        #     max_momentum=0.95,
        #     base_momentum=0.85,
        #     )
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=1.e-3, betas=(0.9, 0.999), weight_decay=1.e-4)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.9)
        self.wall_time = 0.
        self.loss_info = dict(
            loss_list = list(),
            loss_ivs_list = list(),
            loss_dat_list = list(),
            loss_val_list = list(),             
        )
        self.train_info = dict()

        forward_params = kwargs.get('forward_params', None)
        if forward_params is None:
            self.forward_model = None
        else:
            wavebands = torch.tensor(forward_params.pop('wavebands'), dtype=torch.float32, device=self.device)
            landeG = forward_params.pop('landeG',2.5)
            lambda0 = forward_params.pop('lambda0', 630.25)
            wing = forward_params.pop('wing', None)
            self.forward_model = MEForward(wavebands,landeG,lambda0,wing)
            self.forward_model_dense = MEForward(
                torch.linspace(wavebands[0],wavebands[-1],100),
                landeG,
                lambda0,
                wing
            )

    def _set_bounds(self, bounds):
        if bounds is None:
            bounds = [
                [1.e-4,5.e-2],
                [-7.e3,7.e3],
                [1.,1000.],
                [0.1,10.],
                [0.4,0.6],
                [5.,5000.],
                [0.,np.pi],
                [0.,np.pi]
            ]
        self.bounds = bounds
    
    def _set_norm_scale(self, norm_scale):
        if norm_scale is None:
            norm_scale = ['log','linear','log','log','linear','linear','linear','linear']
        self.norm_scale = norm_scale
    
    def normalize(self, params,**kwargs):
        bounds = self.bounds
        norm_scale = self.norm_scale
        if len(norm_scale) != len(bounds):
            raise ValueError('norm_scale must be the same length as bounds')
        if len(bounds) != params.size(1):
            raise ValueError(f'bounds must have the same dimension as params in dim1. bounds has {len(bounds)}, while params has {params.size(1)}')
        norm_params = []
        for pi,bound,scale in zip(params.split(1,dim=1),bounds,norm_scale):
            xmin,xmax = bound
            if scale == 'log':
                xi = (torch.log(pi)-np.log(xmin))/(np.log(xmax)-np.log(xmin))
            elif scale == 'linear':
                xi = (pi-xmin)/(xmax-xmin)
            else:
                raise ValueError(f'unknown scale: {scale}, only log and linear are supported')
            norm_params.append(xi)
        return torch.cat(norm_params,dim=1)
    
    def denormalize(self, params,**kwargs):
        bounds = self.bounds
        norm_scale = self.norm_scale
        if len(norm_scale) != len(bounds):
            raise ValueError('norm_scale must be the same length as bounds')
        denorm_params = []
        for pi,bound,scale in zip(params.split(1,dim=1),bounds,norm_scale):
            xmin,xmax = bound
            if scale == 'log':
                xi = torch.exp(pi*(np.log(xmax)-np.log(xmin))+np.log(xmin))
            elif scale == 'linear':
                xi = pi*(xmax-xmin)+xmin
            else:
                raise ValueError(f'unknown scale: {scale}, only log and linear are supported')
            denorm_params.append(xi)
        return torch.cat(denorm_params,dim=1)

    def chi2(self,iquv_obs,iquv_syn):
        wights = torch.tensor([1,5,5,3.5],device=self.device,dtype=iquv_obs[0].dtype)
        sigmas = torch.tensor([0.118]+[0.204]*3,device=self.device,dtype=iquv_obs[0].dtype)
        F = iquv_obs.size(1)*iquv_obs.size(2)-8
        chi2   = torch.sum((iquv_obs-iquv_syn)**2/sigmas[None,:,None]**2*wights[None,:,None]**2,dim=(-1,-2)).unsqueeze(1)
        # loss = self.criterion(chi2/F,torch.zeros_like(chi2))
        return chi2/F
    
    def inversion_loss(self,data, target=None, dense_spectrum=False):
        output = self.net(data)
        # Bmag = output[:,-3:-2].clone()
        if dense_spectrum:
            with torch.no_grad():
                output = self.denormalize(output)
                target = self.denormalize(target)
                data_con = self.forward_model_dense(*target.T[:,:,None]).permute(1,0,2).detach()
                data_syn = self.forward_model_dense(*output.T[:,:,None]).permute(1,0,2)
            chi2 = self.chi2(data_con, data_syn)
        else:
            with torch.no_grad():
                output = self.denormalize(output)
                data_syn = self.forward_model(*output.T[:,:,None])
                data_syn = data_syn.permute(1,0,2)  
            chi2 = self.chi2(data, data_syn)
        # loss_ivs = self.criterion(chi2*Bmag,torch.zeros_like(chi2))
        loss_ivs = self.criterion(chi2,torch.zeros_like(chi2))
        return loss_ivs
    
    def train(self,
              max_iter=1000,
              training_set = dict(),
              validation_set = dict(),
              **kwargs
              ):
        """
        Train the network

        Input:
        ======
            max_iter: int
                maximum number of iterations
            training_set: dict
                inputs: np.ndarray
                    inputs
                targets: np.ndarray
                    targets
                total_size: int
                    total size of training set
            validation_set: dict
                inputs: np.ndarray
                    inputs
                targets: np.ndarray
                    targets
            **kwargs: keyword arguments
                batch_size: int
                    batch size
                print_interval: int
                    print interval
                save_interval: int
                    save interval
                save_path: str
                    save path
                save_name: str
                    save name
                learning_rate: float
                    learning rate
                betas: tuple
                    betas
                step_size: int
                    step size
                gamma: float
                    gamma
        """
        self.train_info = kwargs
        BS = kwargs.get('batch_size', 1024)
        PI = kwargs.get('print_interval', 10)
        SI = kwargs.get('save_interval', max_iter)
        SP = kwargs.get('save_path', './PI2NN_models/')
        SN = kwargs.get('save_name', 'model')
        lr = kwargs.get('learning_rate', None)
        BT = kwargs.get('betas',None)
        SS = kwargs.get('step_size', None)
        GM = kwargs.get('gamma', 0.9)
        DP = kwargs.get('do_physics_informed', False)
        DS = kwargs.get('dense_spectrum', False)
        wp = kwargs.get('weight_physcis',1.0)

        # data setup
        time_start = time.time()
        data_ipt = training_set.get('inputs',None)
        do_data_driven = True
        if data_ipt is None:
            total_size = training_set.get('total_size',1000000)
            if self.forward_model is None:
                raise ValueError('When no training data is provided, a forward model must be provided')
            print(f'No training data provided, generating random data with total size {total_size}')
            data_tgt = torch.rand(total_size,len(self.bounds))
            data_ipt = self.forward_model(*self.denormalize(data_tgt).T[:,:,None])
            data_ipt = data_ipt.permute(1,0,2).detach()
            if not os.path.exists(SP):
                os.makedirs(SP, exist_ok=True)
            np.savez(
                os.path.join(SP,f'{SN}_training_data.npz'),
                inputs=data_ipt.detach().cpu().numpy(),
                targets=data_tgt.detach().cpu().numpy()
                )
        else:
            data_ipt = torch.from_numpy(data_ipt).to(self.device)
            if training_set.get('targets',None) is None:
                do_data_driven = False
                data_tgt = torch.rand(data_ipt.size(0),len(self.bounds))
            else:
                data_tgt = torch.from_numpy(training_set.get('targets')).to(self.device)
                data_tgt = self.normalize(data_tgt)

        if validation_set:
            val_ipt = validation_set.get('inputs')
            val_tgt = validation_set.get('targets')
            if isinstance(val_ipt, np.ndarray):
                val_ipt = torch.from_numpy(val_ipt).to(self.device)
            if isinstance(val_tgt, np.ndarray):
                val_tgt = torch.from_numpy(val_tgt).to(self.device)

        # optimizer setup
        if lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        if BT is not None:
            for param_group in self.optimizer.param_groups:
                param_group['betas'] = BT
        if SS is not None:
            self.scheduler = StepLR(self.optimizer, step_size=SS, gamma=GM)
        wall_time = 0.

        # Create dataset and dataloader for batch training
        train_dataset = torch.utils.data.TensorDataset(data_ipt, data_tgt)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BS,
            shuffle=True
        )

        if validation_set:
            val_dataset = torch.utils.data.TensorDataset(val_ipt, val_tgt)
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=BS,
                shuffle=False
            )

        for epoch in range(max_iter):
            self.net.train()
            loss_dat = 0
            loss_ivs = 0
            loss_val = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                if do_data_driven:
                    # self.optimizer.zero_grad()
                    output = self.net(data)
                    iloss_dat = self.criterion(output, target)
                    # iloss_dat = self.criterion(self.denormalize(output), self.denormalize(target).detach())
                    # iloss_dat.backward()
                    # self.optimizer.step()
                    loss_dat += iloss_dat.item()
                else:
                    iloss_dat = 0

                if DP:
                    # self.optimizer.zero_grad()
                    # output = self.net(data)
                    # output = self.denormalize(output)
                    # data_syn = self.forward_model(*output.T[:,:,None])
                    # data_syn = data_syn.permute(1,0,2)  
                    # iloss_ivs = self.inversion_loss(data, data_syn)
                    iloss_ivs = self.inversion_loss(data, target, DS)
                    loss_ivs += iloss_ivs.item()
                    # iloss_ivs.backward()
                    # self.optimizer.step()
                else:
                    iloss_ivs = 0
                self.optimizer.zero_grad()
                iloss = iloss_dat + iloss_ivs*wp
                iloss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0,norm_type=2)
                self.optimizer.step()
            loss_dat = loss_dat / (batch_idx+1)
            loss_ivs = loss_ivs / (batch_idx+1)
            self.loss_info['loss_list'].append(loss_dat+loss_ivs)
            self.loss_info['loss_dat_list'].append(loss_dat)
            if DP:
                self.loss_info['loss_ivs_list'].append(loss_ivs)
            if validation_set:
                self.net.eval()
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(val_loader):
                        data, target = data.to(self.device), target.to(self.device)
                        output = self.net(data)
                        loss_val += self.criterion(output, target).item()
                loss_val = loss_val / (batch_idx+1)
                self.loss_info['loss_val_list'].append(loss_val)
            self.scheduler.step()
            wall_time = time.time() - time_start
            self.train_info['wall_time'] = wall_time
            if ((epoch+1) % PI == 0) or (epoch == 0) or (epoch+1==max_iter):
                text = f"Epoch {epoch+1:6d}/{max_iter} | Loss: {loss_dat+loss_ivs:.4e} | Data Loss: {loss_dat:.4e}"
                if DP:
                    text += f" | IVS Loss: {loss_ivs:.4e}"
                if validation_set:
                    text += f" | Validation Loss: {loss_val:.4e}"
                text += f" | lr: {self.scheduler.get_last_lr()[0]:.4e}"
                text += f" | Time: {wall_time/60:8.3f}m"
                print(text)

            if ((epoch+1) % SI == 0) or (epoch == 0) or (epoch+1==max_iter):
                if not os.path.exists(SP):
                    os.makedirs(SP, exist_ok=True)
                torch.save(self, os.path.join(SP,f'{SN}_{epoch+1:06d}.pkl'))
            if (epoch>2) and (loss_dat+loss_ivs<np.nanmin(self.loss_info['loss_list'][:-2])):
                torch.save(self.net, os.path.join(SP,f'{SN}_best.pkl'))
    
    def forward(self, x):
        y = self.net(x)
        y = self.denormalize(y)
        return y