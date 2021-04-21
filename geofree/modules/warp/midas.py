from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from splatting import splatting_function

# pretty much like eval_re_midas but b,c,h,w format and module which should be
# finetunable

def render_forward(src_ims, src_dms,
                   R, t,
                   K_src_inv,
                   K_dst,
                   alpha=None):
    # R: b,3,3
    # t: b,3
    # K_dst: b,3,3
    # K_src_inv: b,3,3
    t = t[...,None]

    #######

    assert len(src_ims.shape) == 4 # b,c,h,w
    assert len(src_dms.shape) == 3 # b,h,w
    assert src_ims.shape[2:4] == src_dms.shape[1:3], (src_ims.shape,
                                                      src_dms.shape)

    x = np.arange(src_ims.shape[3])
    y = np.arange(src_ims.shape[2])
    coord = np.stack(np.meshgrid(x,y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:,:,[0]]), -1) # z=1
    coord = coord.astype(np.float32)
    coord = torch.as_tensor(coord, dtype=K_dst.dtype, device=K_dst.device)
    coord = coord[None] # b,h,w,3

    D = src_dms[:,:,:,None,None] # b,h,w,1,1

    points = K_dst[:,None,None,...]@(R[:,None,None,...]@(D*K_src_inv[:,None,None,...]@coord[:,:,:,:,None])+t[:,None,None,:,:])
    points = points.squeeze(-1)

    new_z = points[:,:,:,[2]].clone().permute(0,3,1,2) # b,1,h,w
    points = points/torch.clamp(points[:,:,:,[2]], 1e-8, None)

    flow = points - coord
    flow = flow.permute(0,3,1,2)[:,:2,...]

    if alpha is not None:
        # used to be 50 but this is unstable even if we subtract the maximum
        importance = alpha/new_z
        #importance = importance-importance.amin((1,2,3),keepdim=True)
        importance = importance.exp()
    else:
        # use heuristic to rescale import between 0 and 10 to be stable in
        # float32
        importance = 1.0/new_z
        importance_min = importance.amin((1,2,3),keepdim=True)
        importance_max = importance.amax((1,2,3),keepdim=True)
        importance=(importance-importance_min)/(importance_max-importance_min+1e-6)*10-10
        importance = importance.exp()

    input_data = torch.cat([importance*src_ims, importance], 1)
    output_data = splatting_function("summation", input_data, flow)

    num = output_data[:,:-1,:,:]
    nom = output_data[:,-1:,:,:]

    #rendered = num/(nom+1e-7)
    rendered = num/nom.clamp(min=1e-8)
    return rendered



class Midas(nn.Module):
    def __init__(self):
        super().__init__()
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")

        self.midas = midas

        # parameters to reproduce the provided transform
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        mean = mean.reshape(1,3,1,1)
        self.register_buffer("mean", mean)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        std = std.reshape(1,3,1,1)
        self.register_buffer("std", std)
        self.__height = 384
        self.__width = 384
        self.__keep_aspect_ratio = True
        self.__resize_method = "upper_bound"
        self.__multiple_of = 32

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def resize(self, x):
        assert len(x.shape)==4
        assert x.shape[1]==3, x.shape

        width, height = self.get_size(
            x.shape[3], x.shape[2]
        )
        x = torch.nn.functional.interpolate(
            x,
            size=(height,width),
            mode="bicubic",
            align_corners=False,
        )
        return x

    def __call__(self, x, clamp=True, out_size="original"):
        assert len(x.shape)==4, x.shape
        assert x.shape[1]==3, x.shape
        assert -1.0 <= x.min() <= x.max() <= 1.0

        # replace provided transform by differentiable one supporting batches
        if out_size == "original":
            out_size = x.shape[2:4]
        # to [0,1]
        x = (x+1.0)/2.0
        # resize
        x = self.resize(x)
        # normalize (x-mean)/std
        x = (x - self.mean)/self.std
        # prepare = transpose to (b,c,h,w)

        x = self.midas(x)

        if out_size is not None:
            x = torch.nn.functional.interpolate(
                x.unsqueeze(1),
                size=out_size,
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)

        if clamp:
            # negative values due to resizing
            x = x.clamp(min=1e-8)

        return x

    def scaled_depth(self, x, points, return_inverse_depth=False):
        b,c,h,w = x.shape
        assert c==3, c
        b_,_,c_ = points.shape
        assert b==b_
        assert c_==3

        dm = self(x)

        xys = points[:,:,:2]
        xys = xys.round().to(dtype=torch.long)
        xys[:,:,0] = xys[:,:,0].clamp(min=0,max=w-1)
        xys[:,:,1] = xys[:,:,1].clamp(min=0,max=h-1)
        indices = xys[:,:,1]*w+xys[:,:,0] # b,N
        flatdm = torch.zeros(b,h*w, dtype=dm.dtype, device=dm.device)
        flatz = points[:,:,2] # b,N
        flatdm.scatter_(dim=1, index=indices, src=flatz)
        sparse_dm = flatdm.reshape(b,h,w)
        mask = (sparse_dm<1e-3).to(dtype=sparse_dm.dtype,
                                  device=sparse_dm.device)

        #error = (1-mask)*(dm[i]-(scale*dm+offset))**2
        N = (1-mask).sum(dim=(1,2)) # b
        m_sparse_dm = (1-mask)*sparse_dm # was mdm
        m_sparse_dm[(1-mask)>0] = 1.0/m_sparse_dm[(1-mask)>0] # we align disparity
        m_dm = (1-mask)*dm # was mmi
        s = ((m_dm*m_sparse_dm).sum(dim=(1,2))-1/N*m_sparse_dm.sum(dim=(1,2))*m_dm.sum(dim=(1,2))) / ((m_dm**2).sum(dim=(1,2))-1/N*(m_dm.sum(dim=(1,2))**2))
        c = 1/N*(m_sparse_dm.sum(dim=(1,2))-s*m_dm.sum(dim=(1,2)))

        scaled_dm = s[:,None,None]*dm + c[:,None,None]
        scaled_dm = scaled_dm.clamp(min=1e-8)
        if not return_inverse_depth:
            scaled_dm[scaled_dm!=0] = 1.0/scaled_dm[scaled_dm!=0] # disparity to depth

        return scaled_dm

    def fixed_scale_depth(self, x, return_inverse_depth=False, scale=[0.18577382, 0.93059154]):
        b,c,h,w = x.shape
        assert c==3, c

        dm = self(x)
        dmmin = dm.amin(dim=(1,2), keepdim=True)
        dmmax = dm.amax(dim=(1,2), keepdim=True)
        scaled_dm = (dm-dmmin)/(dmmax-dmmin)*(scale[1]-scale[0])+scale[0]

        if not return_inverse_depth:
            scaled_dm[scaled_dm!=0] = 1.0/scaled_dm[scaled_dm!=0] # disparity to depth

        return scaled_dm

    def warp(self, x, points, R, t, K_src_inv, K_dst):
        src_dms = self.scaled_depth(x, points)
        wrp = render_forward(src_ims=x, src_dms=src_dms,
                             R=R, t=t,
                             K_src_inv=K_src_inv, K_dst=K_dst)
        return wrp, src_dms

    def warp_features(self, f, x, points, R, t, K_src_inv, K_dst,
                      no_depth_grad=False, boltzmann_factor=None):
        b,c,h,w = f.shape

        context = torch.no_grad() if no_depth_grad else nullcontext()
        with context:
            src_dms = self.scaled_depth(x, points)

            # rescale depth map to feature map size
            src_dms = torch.nn.functional.interpolate(
                src_dms.unsqueeze(1),
                size=(h,w),
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)


        # rescale intrinsics to feature map size
        K_dst = K_dst.clone()
        K_dst[:,0,:] *= f.shape[3]/x.shape[3]
        K_dst[:,1,:] *= f.shape[2]/x.shape[2]
        K_src_inv = K_src_inv.clone()
        K_src_inv[:,0,0] /= f.shape[3]/x.shape[3]
        K_src_inv[:,1,1] /= f.shape[3]/x.shape[3]

        wrp = render_forward(src_ims=f, src_dms=src_dms,
                             R=R, t=t,
                             K_src_inv=K_src_inv, K_dst=K_dst,
                             alpha=boltzmann_factor)
        return wrp, src_dms
