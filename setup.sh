conda remove -n "h2i" --all
conda create -n "h2i" python=3.9
conda activate h2i
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.44.2
pip install diffusers==0.30.3


