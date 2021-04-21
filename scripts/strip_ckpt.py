# remove unnecessary stuff from checkpoints which is not required for inference
import sys, torch

def keep(k):
    blacklist = {"_midas", "depth_stage", "cond_stage_model",
                 "depth_stage_model", "first_stage_model",
                 "loss"}
    return not k.split(".")[0] in blacklist

if __name__ == "__main__":
    inckpt = sys.argv[1]
    outckpt = sys.argv[2]

    ckpt = torch.load(inckpt, map_location="cpu")
    new_ckpt = {"state_dict": dict(
        (k, ckpt["state_dict"][k]) for k in ckpt["state_dict"] if keep(k))}

    print(list(new_ckpt["state_dict"].keys()))
    print(f"Before: {len(ckpt['state_dict'].keys())}")
    print(f"After: {len(new_ckpt['state_dict'].keys())}")

    torch.save(new_ckpt, outckpt)
