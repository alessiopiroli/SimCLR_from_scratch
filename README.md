# SimCLR from scratch
<div style="display: flex; gap: 5px;">
<a href="https://arxiv.org/abs/2002.05709">
<img src="https://img.shields.io/badge/Arxiv-Paper-green" alt="Arxiv Paper" />
</a>
</div>

###
Implementing ["A Simple Framework for Contrastive Learning of Visual Representations"](https://arxiv.org/abs/2002.05709) from scratch.

### Clone and install dependencies
``` 
git clone https://github.com/alessiopiroli/SimCLR_from_scratch
pip install -r requirements.txt 
```

### Train 
``` 
python train.py simclr/config/simclr_config.yml
```


### Results
> UMAP visualization after ~1500 steps showing class-wise clustering of the learned embeddings



https://github.com/user-attachments/assets/6f005bbc-a869-4952-b730-96bd4ec9796a

