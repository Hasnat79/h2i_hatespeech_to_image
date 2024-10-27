conda remove -n "h2i" --all
conda create -n "h2i" python=3.9
conda activate h2i
pip install diffusers==0.31.0 transformers==4.46.0 accelerate==1.0.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install protobuf
pip install sentencepiece


