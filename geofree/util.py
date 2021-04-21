import os, hashlib
import requests
from tqdm import tqdm
import numpy as np
import torch
import os
from omegaconf import OmegaConf

from geofree.models.transformers.geogpt import GeoTransformer


URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1",
    "re_first_stage": "https://heibox.uni-heidelberg.de/f/6db3e2beebe34c1e8d06/?dl=1",
    "re_depth_stage": "https://heibox.uni-heidelberg.de/f/c5e51b91377942d7be18/?dl=1",
    "re_impl_depth_config": "https://heibox.uni-heidelberg.de/f/234e2e21a6414690b663/?dl=1",
    "re_impl_depth": "https://heibox.uni-heidelberg.de/f/740100d4cdbd46d4bb39/?dl=1",
    "re_impl_nodepth_config": "https://heibox.uni-heidelberg.de/f/21989b42cea544bbbb9e/?dl=1",
    "re_impl_nodepth": "https://heibox.uni-heidelberg.de/f/b909c3d4ac7143209387/?dl=1",
}

CKPT_MAP = {
    "vgg_lpips": "geofree/lpips/vgg.pth",
    "re_first_stage": "geofree/re_first_stage/last.ckpt",
    "re_depth_stage": "geofree/re_depth_stage/last.ckpt",
    "re_impl_depth_config": "geofree/re_impl_depth/config.yaml",
    "re_impl_depth": "geofree/re_impl_depth/last.ckpt",
    "re_impl_nodepth_config": "geofree/re_impl_nodepth/config.yaml",
    "re_impl_nodepth": "geofree/re_impl_nodepth/last.ckpt",
}
CACHE = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
CKPT_MAP = dict((k, os.path.join(CACHE, CKPT_MAP[k])) for k in CKPT_MAP)

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a",
    "re_first_stage": "b8b999aba6b618757329c1f114e9f5a5",
    "re_depth_stage": "ab35861b9050476d02fa8f4c8f761d61",
    "re_impl_depth_config": "e75df96667a3a6022ca2d5b27515324b",
    "re_impl_depth": "144dcdeb1379760d2f1ae763cabcac85",
    "re_impl_nodepth_config": "351c976463c4740fc575a7c64a836624",
    "re_impl_nodepth": "b6646db26a756b80840aa2b6aca924d8",
}


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_local_path(name, root=None, check=False):
    path = CKPT_MAP[name]
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        assert name in URL_MAP, name
        print("Downloading {} from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


def pretrained_models(model="re_impl_nodepth"):
    assert model.startswith("re_"), "not implemented"

    config_path = get_local_path(model+"_config")
    config = OmegaConf.load(config_path)
    config.model.params.first_stage_config.params["ckpt_path"] = get_local_path("re_first_stage")
    config.model.params.depth_stage_config.params["ckpt_path"] = get_local_path("re_depth_stage")

    ckpt_path = get_local_path(model)

    model = GeoTransformer(**config.model.params)
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model = model.eval()
    print(f"Restored model from {ckpt_path}")
    return model
