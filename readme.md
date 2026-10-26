# Acknowledgement
Our codebase directly uses some of the code from earlier open-source repositories. I put the most important ones here. I would like to thank their setup and hope this repositories further facilitates open researches and code sharing. Some code is modified by LLMs, but it has been thoroughly checked by me!

[PanDerm](https://github.com/SiyuanYan1/PanDerm)

[Derm1m](https://github.com/mlfoundations/open_clip)

[Derm1m](https://github.com/SiyuanYan1/Derm1M)

[Pretrained Derm1m base-weights (panderm)](https://huggingface.co/redlessone/DermLIP_PanDerm-base-w-PubMed-256/blob/b079489efd7ec1f8f9c5553c117ab8d4335f99c3/README.md)

[Pretrained Derm1m Clip Base weights](https://huggingface.co/redlessone/DermLIP_ViT-B-16)

Vision Tx Modification [1](https://github.com/SiyuanYan1/Derm1M/blob/main/src/open_clip/transformer.py), [2](https://github.com/SiyuanYan1/Derm1M/blob/main/src/CAE/models/modeling_finetune.py)


[MONET](https://github.com/suinleelab/MONET)

[Monet Base Weights](https://aimslab.cs.washington.edu/MONET/weight_clip.pt)

[Model Modification](https://github.com/suinleelab/MONET/blob/main/src/clip/model.py)



# User Instruction
Users may have to download the missing files from the original source GitHubs for the models. This repository contains all the modified files in the exact directories compatible with the base repos. 
# Background
# Package instructions
# Our Codes
### Panderm Models:  
Model Definition: PanDerm/classification/script/custom_models  panderm_custom_classifier.py

Test Model: PanDerm/classification/ classification_panderm_experiment.ipynb

Training (cleaned) Model: PanDerm/classification/ training_panderm.ipynb
