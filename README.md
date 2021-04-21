# Geometry-Free View Synthesis: Transformers and no 3D Priors
![teaser](assets/firstpage.jpg)

[**Geometry-Free View Synthesis: Transformers and no 3D Priors**](https://compvis.github.io/geometry-free-view-synthesis/)<br/>
[Robin Rombach](https://github.com/rromb)\*,
[Patrick Esser](https://github.com/pesser)\*,
[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer)<br/>
\* equal contribution

[arXiv](https://arxiv.org/abs/2104.07652) | [BibTeX](#bibtex)

### Interactive Scene Exploration Results

[RealEstate10K](https://google.github.io/realestate10k/):<br/>
<a href="assets/realestate_short.mp4">![realestate](assets/realestate_preview.gif)</a><br/>
Videos: [short (2min)](assets/realestate_short.mp4) / [long (12min)](assets/realestate_long.mp4)

[ACID](https://infinite-nature.github.io/):<br/>
<a href="assets/acid_short.mp4">![acid](assets/acid_preview.gif)</a><br/>
Videos: [short (2min)](assets/acid_short.mp4) / [long (9min)](assets/acid_long.mp4)

### Demo

#### Installation

The demo requires building a PyTorch extension. If you have a sane development
environment with PyTorch, g++ and nvcc, you can simply

```
pip install git+https://github.com/CompVis/geometry-free-view-synthesis#egg=geometry-free-view-synthesis
```

If you run into problems and have a GPU with compute capability below 8, you
can also use the provided conda environment:

```
git clone https://github.com/CompVis/geometry-free-view-synthesis
conda env create -f geometry-free-view-synthesis/environment.yaml
conda activate geofree
pip install geometry-free-view-synthesis/
```

#### Running

After [installation](#installation), running

```
braindance.py
```

will start the demo on [a sample scene](http://walledoffhotel.com/rooms.html).
Explore the scene interactively using the `WASD` keys to move and `arrow keys` to
look around. Once positioned, hit the `space bar` to render the novel view with
GeoGPT.

You can move again with WASD keys. Mouse control can be activated with the m
key. Run `braindance.py <folder to select image from/path to image>` to run the
demo on your own images. By default, it uses the `re-impl-nodepth` (trained on
RealEstate without explicit transformation and no depth input) which can be
changed with the `--model` flag. The corresponding checkpoints will be
downloaded the first time they are required. Specify an output path using
`--video path/to/vid.mp4` to record a video.

```
> braindance.py -h
usage: braindance.py [-h] [--model {re_impl_nodepth,re_impl_depth}] [--video [VIDEO]] [path]

What's up, BD-maniacs?

key(s)       action                  
=====================================
wasd         move around             
arrows       look around             
m            enable looking with mouse
space        render with transformer 
q            quit                    

positional arguments:
  path                  path to image or directory from which to select image. Default example is used if not specified.

optional arguments:
  -h, --help            show this help message and exit
  --model {re_impl_nodepth,re_impl_depth}
                        pretrained model to use.
  --video [VIDEO]       path to write video recording to. (no recording if unspecified).
```

## BibTeX

```
@misc{rombach2021geometryfree,
      title={Geometry-Free View Synthesis: Transformers and no 3D Priors}, 
      author={Robin Rombach and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2104.07652},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
