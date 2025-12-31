You can set up the environment by following the instructions below:

```bash

# Lazy way to install dependencies
conda create --name swift --file requirements.txt

# Or funnier way ;)
conda create -n swift python=3.8 -y
conda activate swift

conda install -y pytorch torchvision torchaudio torchmetrics -c pytorch

# need to install the correct torchvision version
pip3 install torchvision==0.15.2

# install openclip module
pip install open_clip_torch

# install OpenAI CLIP
pip install git+https://github.com/openai/CLIP.git

# for retrieving images from urls
pip install img2dataset==1.2.0

```