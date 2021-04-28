#!/usr/bin/env python
import sys, os
import argparse
import pygame
import torch
import numpy as np
from PIL import Image
from splatting import splatting_function
from torch.utils.data.dataloader import default_collate
from geofree import pretrained_models
import imageio

torch.set_grad_enabled(False)

from geofree.modules.warp.midas import Midas

from tkinter.filedialog import askopenfilename
from tkinter import Tk


def to_surface(x, text=None):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = x.transpose(1,0,2)
    x = (x+1.0)*127.5
    x = x.clip(0, 255).astype(np.uint8)
    if text is not None:
        from PIL import ImageDraw, ImageFont
        fontsize=22
        try:
            font = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSans-BoldItalic.ttf", fontsize)
        except OSError:
            font = ImageFont.load_default()
        margin = 8

        x = x.transpose(1,0,2)
        pos = (margin, x.shape[0]-fontsize-margin//2)
        x = x.astype(np.float)
        x[x.shape[0]-fontsize-margin:,:,:] *= 0.5
        x = x.astype(np.uint8)

        img = Image.fromarray(x)
        ImageDraw.Draw(img).text(pos, f'{text}', (255, 255, 255), font=font) # coordinates, text, color, font
        x = np.array(img)
        x = x.transpose(1,0,2)
    return pygame.surfarray.make_surface(x), x.transpose(1,0,2)


def render_forward(src_ims, src_dms,
                   Rcam, tcam,
                   K_src,
                   K_dst):
    Rcam = Rcam.to(device=src_ims.device)[None]
    tcam = tcam.to(device=src_ims.device)[None]

    R = Rcam
    t = tcam[...,None]
    K_src_inv = K_src.inverse()

    assert len(src_ims.shape) == 4
    assert len(src_dms.shape) == 3
    assert src_ims.shape[1:3] == src_dms.shape[1:3], (src_ims.shape,
                                                      src_dms.shape)

    x = np.arange(src_ims[0].shape[1])
    y = np.arange(src_ims[0].shape[0])
    coord = np.stack(np.meshgrid(x,y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:,:,[0]]), -1) # z=1
    coord = coord.astype(np.float32)
    coord = torch.as_tensor(coord, dtype=K_src.dtype, device=K_src.device)
    coord = coord[None] # bs, h, w, 3

    D = src_dms[:,:,:,None,None]

    points = K_dst[None,None,None,...]@(R[:,None,None,...]@(D*K_src_inv[None,None,None,...]@coord[:,:,:,:,None])+t[:,None,None,:,:])
    points = points.squeeze(-1)

    new_z = points[:,:,:,[2]].clone().permute(0,3,1,2) # b,1,h,w
    points = points/torch.clamp(points[:,:,:,[2]], 1e-8, None)

    src_ims = src_ims.permute(0,3,1,2)
    flow = points - coord
    flow = flow.permute(0,3,1,2)[:,:2,...]

    alpha = 0.5
    importance = alpha/new_z
    importance_min = importance.amin((1,2,3),keepdim=True)
    importance_max = importance.amax((1,2,3),keepdim=True)
    importance=(importance-importance_min)/(importance_max-importance_min+1e-6)*10-10
    importance = importance.exp()

    input_data = torch.cat([importance*src_ims, importance], 1)
    output_data = splatting_function("summation", input_data, flow)

    num = torch.sum(output_data[:,:-1,:,:], dim=0, keepdim=True)
    nom = torch.sum(output_data[:,-1:,:,:], dim=0, keepdim=True)

    rendered = num/(nom+1e-7)
    rendered = rendered.permute(0,2,3,1)[0,...]
    return rendered

def normalize(x):
    return x/np.linalg.norm(x)

def cosd(x):
    return np.cos(np.deg2rad(x))

def sind(x):
    return np.sin(np.deg2rad(x))

def look_to(camera_pos, camera_dir, camera_up):
  camera_right = normalize(np.cross(camera_up, camera_dir))
  R = np.zeros((4, 4))
  R[0,0:3] = normalize(camera_right)
  R[1,0:3] = normalize(np.cross(camera_dir, camera_right))
  R[2,0:3] = normalize(camera_dir)
  R[3,3] = 1
  trans_matrix = np.array([[1.0, 0.0, 0.0, -camera_pos[0]],
                           [0.0, 1.0, 0.0, -camera_pos[1]],
                           [0.0, 0.0, 1.0, -camera_pos[2]],
                           [0.0, 0.0, 0.0,            1.0]])
  tmp = R@trans_matrix
  return tmp[:3,:3], tmp[:3,3]

def rotate_around_axis(angle, axis):
    axis = normalize(axis)
    rotation = np.array([[cosd(angle)+axis[0]**2*(1-cosd(angle)),
                          axis[0]*axis[1]*(1-cosd(angle))-axis[2]*sind(angle),
                          axis[0]*axis[2]*(1-cosd(angle))+axis[1]*sind(angle)],
                         [axis[1]*axis[0]*(1-cosd(angle))+axis[2]*sind(angle),
                          cosd(angle)+axis[1]**2*(1-cosd(angle)),
                          axis[1]*axis[2]*(1-cosd(angle))-axis[0]*sind(angle)],
                         [axis[2]*axis[0]*(1-cosd(angle))-axis[1]*sind(angle),
                          axis[2]*axis[1]*(1-cosd(angle))+axis[0]*sind(angle),
                          cosd(angle)+axis[2]**2*(1-cosd(angle))]])
    return rotation


class Renderer(object):
    def __init__(self, model, device):
        self.model = pretrained_models(model=model)
        self.model = self.model.to(device=device)
        self._active = False
        # rough estimates for min and maximum inverse depth values on the
        # training datasets
        if model.startswith("re"):
            self.scale = [0.18577382, 0.93059154]
        else:
            self.scale = [1e-8, 0.75]

    def init(self,
             start_im,
             example,
             show_R,
             show_t):
        self._active = True
        self.step = 0

        batch = self.batch = default_collate([example])
        batch["R_rel"] = show_R[None,...]
        batch["t_rel"] = show_t[None,...]

        _, cdict, edict = self.model.get_xce(batch)
        for k in cdict:
            cdict[k] = cdict[k].to(device=self.model.device)
        for k in edict:
            edict[k] = edict[k].to(device=self.model.device)

        quant_d, quant_c, dc_indices, embeddings = self.model.get_normalized_c(
            cdict, edict, fixed_scale=True, scale=self.scale)

        start_im = start_im[None,...].to(self.model.device).permute(0,3,1,2)
        quant_c, c_indices = self.model.encode_to_c(c=start_im)
        cond_rec = self.model.cond_stage_model.decode(quant_c)

        self.current_im = cond_rec.permute(0,2,3,1)[0]
        self.current_sample = c_indices

        self.quant_c = quant_c # to know shape
        # for sampling
        self.dc_indices = dc_indices
        self.embeddings = embeddings

    def __call__(self):
        if self.step < self.current_sample.shape[1]:
            z_start_indices = self.current_sample[:, :self.step]
            temperature=None
            top_k=250
            callback=None
            index_sample = self.model.sample(z_start_indices, self.dc_indices,
                                             steps=1,
                                             temperature=temperature if temperature is not None else 1.0,
                                             sample=True,
                                             top_k=top_k if top_k is not None else 100,
                                             callback=callback if callback is not None else lambda k: None,
                                             embeddings=self.embeddings)
            self.current_sample = torch.cat((index_sample,
                                             self.current_sample[:,self.step+1:]),
                                            dim=1)

            sample_dec = self.model.decode_to_img(self.current_sample,
                                                  self.quant_c.shape)
            self.current_im = sample_dec.permute(0,2,3,1)[0]
            self.step += 1

        if self.step >= self.current_sample.shape[1]:
            self._active = False

        return self.current_im

    def active(self):
        return self._active

    def reconstruct(self, x):
        x = x.to(self.model.device).permute(0,3,1,2)
        quant_c, c_indices = self.model.encode_to_c(c=x)
        x_rec = self.model.cond_stage_model.decode(quant_c)
        return x_rec.permute(0,2,3,1)


def load_as_example(path, model="re"):
    size = [208, 368]
    im = Image.open(path)
    w,h = im.size
    if np.abs(w/h - size[1]/size[0]) > 0.1:
        print(f"Center cropping {path} to AR {size[1]/size[0]}")
        if w/h < size[1]/size[0]:
            # crop h
            left = 0
            right = w
            top = h/2 - size[0]/size[1]*w/2
            bottom = h/2 + size[0]/size[1]*w/2
        else:
            # crop w
            top = 0
            bottom = h
            left = w/2 - size[1]/size[0]*h
            right = w/2 + size[1]/size[0]*h
        im = im.crop(box=(left, top, right, bottom))

    im = im.resize((size[1],size[0]),
                   resample=Image.LANCZOS)
    im = np.array(im)/127.5-1.0
    im = im.astype(np.float32)

    example = dict()
    example["src_img"] = im
    if model.startswith("re"):
        example["K"] = np.array([[184.0, 0.0, 184.0],
                                 [0.0, 184.0, 104.0],
                                 [0.0, 0.0, 1.0]], dtype=np.float32)
    elif model.startswith("ac"):
        example["K"] = np.array([[200.0, 0.0, 184.0],
                                 [0.0, 200.0, 104.0],
                                 [0.0, 0.0, 1.0]], dtype=np.float32)
    else:
        raise NotImplementedError()
    example["K_inv"] = np.linalg.inv(example["K"])

    ## dummy data not used during inference
    example["dst_img"] = np.zeros_like(example["src_img"])
    example["src_points"] = np.zeros((1,3), dtype=np.float32)

    return example


if __name__ == "__main__":
    helptxt = "What's up, BD-maniacs?\n\n"+"\n".join([
        "{: <12} {: <24}".format("key(s)", "action"),
        "="*37,
        "{: <12} {: <24}".format("wasd", "move around"),
        "{: <12} {: <24}".format("arrows", "look around"),
        "{: <12} {: <24}".format("m", "enable looking with mouse"),
        "{: <12} {: <24}".format("space", "render with transformer"),
        "{: <12} {: <24}".format("q", "quit"),
    ])
    parser = argparse.ArgumentParser(description=helptxt,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('path', type=str, nargs='?', default=None,
                        help='path to image or directory from which to select '
                        'image. Default example is used if not specified.')
    parser.add_argument('--model', choices=["re_impl_nodepth", "re_impl_depth",
                                            "ac_impl_nodepth", "ac_impl_depth"],
                        default="re_impl_nodepth",
                        help='pretrained model to use.')
    parser.add_argument('--video', type=str, nargs='?', default=None,
                        help='path to write video recording to. (no recording if unspecified).')
    opt = parser.parse_args()
    print(helptxt)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("Warning: Running on CPU---sampling might take a while...")
        device = torch.device("cpu")
    midas = Midas().eval().to(device)
    # init transformer
    renderer = Renderer(model=opt.model, device=device)

    if opt.path is None:
        try:
            import importlib.resources as pkg_resources
        except ImportError:
            import importlib_resources as pkg_resources

        example_name = "artist.jpg" if opt.model.startswith("re") else "beach.jpg"
        with pkg_resources.path("geofree.examples", example_name) as path:
            example = load_as_example(path, model=opt.model)
    else:
        path = opt.path
        if not os.path.isfile(path):
            Tk().withdraw()
            path = askopenfilename(initialdir=sys.argv[1])
        example = load_as_example(path, model=opt.model)

    ims = example["src_img"][None,...]
    K = example["K"]

    # compute depth for preview
    dms = [None]
    for i in range(ims.shape[0]):
        midas_in = torch.tensor(ims[i])[None,...].permute(0,3,1,2).to(device)
        scaled_idepth = midas.fixed_scale_depth(midas_in,
                                                return_inverse_depth=True,
                                                scale=renderer.scale)
        dms[i] = 1.0/scaled_idepth[0].cpu().numpy()

    # now switch to pytorch
    src_ims = torch.tensor(ims, dtype=torch.float32)
    src_dms = torch.tensor(dms, dtype=torch.float32)
    K = torch.tensor(K, dtype=torch.float32)

    src_ims = src_ims.to(device=device)
    src_dms = src_dms.to(device=device)
    K = K.to(device=device)

    K_cam = K.clone().detach()

    RENDERING = False
    DISPLAY_REC = True
    if DISPLAY_REC:
        rec_ims = renderer.reconstruct(src_ims)

    # init pygame
    b,h,w,c = src_ims.shape
    pygame.init()
    display = (w, h)
    surface = pygame.display.set_mode(display)
    clock = pygame.time.Clock()

    # init camera
    camera_pos = np.array([0.0, 0.0, 0.0])
    camera_dir = np.array([0.0, 0.0, 1.0])
    camera_up = np.array([0.0, 1.0, 0.0])
    CAM_SPEED = 0.025
    CAM_SPEED_YAW = 0.5
    CAM_SPEED_PITCH = 0.25
    MOUSE_SENSITIVITY = 0.02
    USE_MOUSE = False
    if opt.model.startswith("ac"):
        CAM_SPEED *= 0.1
        CAM_SPEED_YAW *= 0.5
        CAM_SPEED_PITCH *= 0.5

    if opt.video is not None:
        writer = imageio.get_writer(opt.video, fps=40)

    step = 0
    step_PHASE = 0
    while True:
        ######## Boring stuff
        clock.tick(40)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            if opt.video is not None:
                writer.close()
            pygame.quit()
            quit()

        ######### Camera
        camera_yaw = 0
        camera_pitch = 0
        if keys[pygame.K_a]:
            camera_pos += CAM_SPEED*normalize(np.cross(camera_dir, camera_up))
        if keys[pygame.K_d]:
            camera_pos -= CAM_SPEED*normalize(np.cross(camera_dir, camera_up))
        if keys[pygame.K_w]:
            camera_pos += CAM_SPEED*normalize(camera_dir)
        if keys[pygame.K_s]:
            camera_pos -= CAM_SPEED*normalize(camera_dir)
        if keys[pygame.K_PAGEUP]:
            camera_pos -= CAM_SPEED*normalize(camera_up)
        if keys[pygame.K_PAGEDOWN]:
            camera_pos += CAM_SPEED*normalize(camera_up)

        if keys[pygame.K_LEFT]:
            camera_yaw += CAM_SPEED_YAW
        if keys[pygame.K_RIGHT]:
            camera_yaw -= CAM_SPEED_YAW
        if keys[pygame.K_UP]:
            camera_pitch -= CAM_SPEED_PITCH
        if keys[pygame.K_DOWN]:
            camera_pitch += CAM_SPEED_PITCH

        if USE_MOUSE:
            dx, dy = pygame.mouse.get_rel()
            if not RENDERING:
                camera_yaw -= MOUSE_SENSITIVITY*dx
                camera_pitch += MOUSE_SENSITIVITY*dy

        if keys[pygame.K_PLUS]:
            CAM_SPEED += 0.1
            print(CAM_SPEED)
        if keys[pygame.K_MINUS]:
            CAM_SPEED -= 0.1
            print(CAM_SPEED)

        if keys[pygame.K_m]:
            if not USE_MOUSE:
                pygame.mouse.set_visible(False)
                pygame.event.set_grab(True)
                USE_MOUSE = True
            else:
                pygame.mouse.set_visible(True)
                pygame.event.set_grab(False)
                USE_MOUSE = False

        # adjust for yaw and pitch
        rotation = np.array([[cosd(-camera_yaw), 0.0, sind(-camera_yaw)],
                             [0.0, 1.0, 0.0],
                             [-sind(-camera_yaw), 0.0, cosd(-camera_yaw)]])
        camera_dir = rotation@camera_dir

        rotation = rotate_around_axis(camera_pitch, np.cross(camera_dir,
                                                             camera_up))
        camera_dir = rotation@camera_dir

        show_R, show_t = look_to(camera_pos, camera_dir, camera_up) # look from pos in direction dir
        show_R = torch.as_tensor(show_R, dtype=torch.float32)
        show_t = torch.as_tensor(show_t, dtype=torch.float32)

        ############# /Camera
        ###### control rendering
        if keys[pygame.K_SPACE]:
            RENDERING = True
            renderer.init(wrp_im, example, show_R, show_t)

        PRESSED = False
        if any(keys[k] for k in [pygame.K_a, pygame.K_d, pygame.K_w,
                                 pygame.K_s]):
            RENDERING = False

        # display
        if not RENDERING:
            with torch.no_grad():
                wrp_im = render_forward(src_ims, src_dms,
                                        show_R, show_t,
                                        K_src=K,
                                        K_dst=K_cam)
        else:
            with torch.no_grad():
                wrp_im = renderer()

        text = "Sampling" if renderer._active else None
        image, frame = to_surface(wrp_im, text)
        surface.blit(image, (0,0))
        pygame.display.flip()
        if opt.video is not None:
            writer.append_data(frame)

        step +=1
