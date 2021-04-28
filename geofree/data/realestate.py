from math import sqrt
import os
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from geofree.data.read_write_model import read_model


class PRNGMixin(object):
    """Adds a prng property which is a numpy RandomState which gets
    reinitialized whenever the pid changes to avoid synchronized sampling
    behavior when used in conjunction with multiprocessing."""

    @property
    def prng(self):
        currentpid = os.getpid()
        if getattr(self, "_initpid", None) != currentpid:
            self._initpid = currentpid
            self._prng = np.random.RandomState()
        return self._prng


def load_sparse_model_example(root, img_dst, img_src, size):
    """
    Parameters
        root        folder containing directory sparse with points3D.bin,
                    images.bin and cameras.bin, and directory images
        img_dst     filename of image in images to be used as destination
        img_src     filename of image in images to be used as source
        size        size to resize image and parameters to. If None nothing is
                    done, otherwise it should be in (h,w) format
    Returns
        example     dictionary containing
            dst_img     destination image as (h,w,c) array in range (-1,1)
            src_img     source image as (h,w,c) array in range (-1,1)
            src_points  sparse set of 3d points for source image as (N,3) array
                        with (:,:2) being pixel coordinates and (:,2) depth values
            K           3x3 camera intrinsics
            K_inv       inverse of camera intrinsics
            R_rel       relative rotation mapping from source to destination
                        coordinate system
            t_rel       relative translation mapping from source to destination
                        coordinate system
    """
    # load sparse model
    model = os.path.join(root, "sparse")
    try:
        cameras, images, points3D = read_model(path=model, ext=".bin")
    except Exception as e:
        raise Exception(f"Failed to load sparse model {model}.") from e


    # load camera parameters and image size
    cam = cameras[1]
    h = cam.height
    w = cam.width
    params = cam.params
    K = np.array([[params[0], 0.0, params[2]],
                  [0.0, params[1], params[3]],
                  [0.0,       0.0,       1.0]])

    # find keys of desired dst and src images
    key_dst = [k for k in images.keys() if images[k].name==img_dst]
    assert len(key_dst)==1, (img_dst, key_dst)
    key_src = [k for k in images.keys() if images[k].name==img_src]
    assert len(key_src)==1, (img_src, key_src)
    keys = [key_dst[0], key_src[0]]

    # load extrinsics
    Rs = np.stack([images[k].qvec2rotmat() for k in keys])
    ts = np.stack([images[k].tvec for k in keys])

    # load sparse 3d points to be able to estimate scale
    sparse_points = [None, None]
    #for i in range(len(keys)):
    for i in [1]: # only need it for source
        key = keys[i]
        xys = images[key].xys
        p3D = images[key].point3D_ids
        pmask = p3D > 0
        # if verbose: print("Found {} 3d points in sparse model.".format(pmask.sum()))
        xys = xys[pmask]
        p3D = p3D[pmask]
        worlds = np.stack([points3D[id_].xyz for id_ in p3D]) # N, 3
        # project to current view
        worlds = worlds[..., None] # N,3,1
        pixels = K[None,...]@(Rs[i][None,...]@worlds+ts[i][None,...,None])
        pixels = pixels.squeeze(-1) # N,3

        # instead of using provided xys, one could also project pixels, ie
        # xys ~ pixels[:,:2]/pixels[:,[2]]
        points = np.concatenate([xys, pixels[:,[2]]], axis=1)
        sparse_points[i] = points

        # code to convert to sparse depth map
        # xys = points[:,:2]
        # xys = np.round(xys).astype(np.int)
        # xys[:,0] = xys[:,0].clip(min=0,max=w-1)
        # xys[:,1] = xys[:,1].clip(min=0,max=h-1)
        # indices = xys[:,1]*w+xys[:,0]
        # flatdm = np.zeros(h*w)
        # flatz = pixels[:,2]
        # np.put_along_axis(flatdm, indices, flatz, axis=0)
        # sparse_dm = flatdm.reshape(h,w)

    # load images
    im_root = os.path.join(root, "images")
    im_paths = [os.path.join(im_root, images[k].name) for k in keys]
    ims = list()
    for path in im_paths:
        im = Image.open(path)
        ims.append(im)

    if size is not None and (size[0] != h or size[1] != w):
        # resize
        ## K
        K[0,:] = K[0,:]*size[1]/w
        K[1,:] = K[1,:]*size[0]/h
        ## points
        points[:,0] = points[:,0]*size[1]/w
        points[:,1] = points[:,1]*size[0]/h
        ## img
        for i in range(len(ims)):
            ims[i] = ims[i].resize((size[1],size[0]),
                                   resample=Image.LANCZOS)


    for i in range(len(ims)):
        ims[i] = np.array(ims[i])/127.5-1.0


    # relative camera
    R_dst = Rs[0]
    t_dst = ts[0]
    R_src_inv = Rs[1].transpose(-1,-2)
    t_src = ts[1]
    R_rel = R_dst@R_src_inv
    t_rel = t_dst-R_rel@t_src
    K_inv = np.linalg.inv(K)

    # collect results
    example = {
        "dst_img": ims[0],
        "src_img": ims[1],
        "src_points": sparse_points[1],
        "K": K,
        "K_inv": K_inv,
        "R_rel": R_rel,
        "t_rel": t_rel,
    }

    return example

def pad_points(points, N):
    padded = -1*np.ones((N,3), dtype=points.dtype)
    padded[:points.shape[0],:] = points
    return padded

class RealEstate10KCustomTest(data.Dataset):
    def __init__(self, size=None, max_points=16384):
        self.size = size
        self.max_points = max_points

        self.frames_file = "data/realestate_custom_frames.txt"
        self.sparse_dir = "data/realestate_sparse"
        self.split = "test"

        with open(self.frames_file, "r") as f:
            frames = f.read().splitlines()

        seq_data = dict()
        for line in frames:
            seq,a,b,c = line.split(",")
            assert not seq in seq_data
            seq_data[seq] = [a,b,c]

        # sequential list of seq, label, dst, src
        # where label is used to disambiguate different warping scenarios
        # 0: small forward movement
        # 1: large forward movement
        # 2: small backward movement (reverse of 0)
        # 3: large backward movement (reverse of 1)
        frame_data = list()
        for seq in sorted(seq_data.keys()):
            abc = seq_data[seq]
            frame_data.append([seq, 0, abc[1], abc[0]]) # b|a
            frame_data.append([seq, 1, abc[2], abc[0]]) # c|a
            frame_data.append([seq, 2, abc[0], abc[1]]) # a|b
            frame_data.append([seq, 3, abc[0], abc[2]]) # a|c

        self.frame_data = frame_data

    def __len__(self):
        return len(self.frame_data)

    def __getitem__(self, index):
        seq, label, img_dst, img_src = self.frame_data[index]
        root = os.path.join(self.sparse_dir, self.split, seq)

        example = load_sparse_model_example(
            root=root, img_dst=img_dst, img_src=img_src, size=self.size)

        for k in example:
            example[k] = example[k].astype(np.float32)

        example["src_points"] = pad_points(example["src_points"],
                                           self.max_points)
        example["seq"] = seq
        example["label"] = label
        example["dst_fname"] = img_dst
        example["src_fname"] = img_src

        return example


class RealEstate10KSparseTrain(data.Dataset, PRNGMixin):
    def __init__(self, size=None, max_points=16384):
        self.size = size
        self.max_points = max_points

        self.sparse_dir = "data/realestate_sparse"
        self.split = "train"
        self.sequence_dir = os.path.join(self.sparse_dir, self.split)
        with open("data/realestate_train_sequences.txt", "r") as f:
            self.sequences = f.read().splitlines()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]
        root = os.path.join(self.sequence_dir, seq)
        frames = sorted([fname for fname in os.listdir(os.path.join(root, "images")) if fname.endswith(".png")])
        segments = self.prng.choice(3, 2, replace=False)
        if segments[0] < segments[1]: # forward
            if segments[1]-segments[0] == 1: # small
                label = 0
            else:
                label = 1 # large
        else: # backward
            if segments[1]-segments[0] == 1: # small
                label = 2
            else:
                label = 3
        n = len(frames)
        dst_indices = list(range(segments[0]*n//3, (segments[0]+1)*n//3))
        src_indices = list(range(segments[1]*n//3, (segments[1]+1)*n//3))
        dst_index = self.prng.choice(dst_indices)
        src_index = self.prng.choice(src_indices)
        img_dst = frames[dst_index]
        img_src = frames[src_index]

        example = load_sparse_model_example(
            root=root, img_dst=img_dst, img_src=img_src, size=self.size)

        for k in example:
            example[k] = example[k].astype(np.float32)

        example["src_points"] = pad_points(example["src_points"],
                                           self.max_points)
        example["seq"] = seq
        example["label"] = label
        example["dst_fname"] = img_dst
        example["src_fname"] = img_src

        return example


class RealEstate10KSparseCustom(data.Dataset):
    def __init__(self, frame_data, split, size=None, max_points=16384):
        self.size = size
        self.max_points = max_points

        self.sparse_dir = "data/realestate_sparse"
        self.split = split
        self.frame_data = frame_data

    def __len__(self):
        return len(self.frame_data)

    def __getitem__(self, index):
        seq, label, img_dst, img_src = self.frame_data[index]
        root = os.path.join(self.sparse_dir, self.split, seq)

        example = load_sparse_model_example(
            root=root, img_dst=img_dst, img_src=img_src, size=self.size)

        for k in example:
            example[k] = example[k].astype(np.float32)

        example["src_points"] = pad_points(example["src_points"],
                                           self.max_points)
        example["seq"] = seq
        example["label"] = label
        example["dst_fname"] = img_dst
        example["src_fname"] = img_src

        return example
