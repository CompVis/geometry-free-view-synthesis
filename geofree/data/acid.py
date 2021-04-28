import os
import numpy as np
from torch.utils.data import Dataset

from geofree.data.realestate import PRNGMixin, load_sparse_model_example, pad_points


class ACIDSparseBase(Dataset, PRNGMixin):
    def __init__(self):
        self.sparse_dir = "data/acid_sparse"

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


class ACIDSparseTrain(ACIDSparseBase):
    def __init__(self, size=None, max_points=16384):
        super().__init__()
        self.size = size
        self.max_points = max_points

        self.split = "train"
        self.sequence_dir = os.path.join(self.sparse_dir, self.split)
        with open("data/acid_train_sequences.txt", "r") as f:
            self.sequences = f.read().splitlines()


class ACIDSparseValidation(ACIDSparseBase):
    def __init__(self, size=None, max_points=16384):
        super().__init__()
        self.size = size
        self.max_points = max_points

        self.split = "validation"
        self.sequence_dir = os.path.join(self.sparse_dir, self.split)
        with open("data/acid_validation_sequences.txt", "r") as f:
            self.sequences = f.read().splitlines()


class ACIDSparseTest(ACIDSparseBase):
    def __init__(self, size=None, max_points=16384):
        super().__init__()
        self.size = size
        self.max_points = max_points

        self.split = "test"
        self.sequence_dir = os.path.join(self.sparse_dir, self.split)
        with open("data/acid_test_sequences.txt", "r") as f:
            self.sequences = f.read().splitlines()


class ACIDCustomTest(Dataset):
    def __init__(self, size=None, max_points=16384):
        self.size = size
        self.max_points = max_points

        self.frames_file = "data/acid_custom_frames.txt"
        self.sparse_dir = "data/acid_sparse"
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
