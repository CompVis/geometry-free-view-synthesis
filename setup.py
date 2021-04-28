from setuptools import setup, find_packages

setup(
    name='geometry-free-view-synthesis',
    version='0.0.1',
    description='Geometry-Free View Synthesis: Transformers and no 3D Priors',
    url='https://github.com/CompVis/geometry-free-view-synthesis',
    author='Robin Rombach, Patrick Esser, Bjorn Ommer',
    author_email='robin.rombach@iwr.uni-heidelberg.de',
    packages=find_packages(),
    package_data={"geofree": ["examples/*"]},
    include_package_data=True,
    scripts=['scripts/braindance.py'],
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'tqdm',
        'omegaconf>=2.0.0',
        'pytorch-lightning>=1.0.8',
        'pygame',
        'splatting @ git+https://github.com/pesser/splatting@1427d7c4204282d117403b35698d489e0324287f#egg=splatting',
        'einops',
        'importlib-resources',
        'imageio',
        'imageio-ffmpeg',
        'test-tube'
    ],
)
