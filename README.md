# Experiments of TTA

* **Implimentation**： Here are the commands for testing performance on Imagenet1K:<br><br>1. prepare Imagenet1K dataset using imagenet1k.py  <br><br>2. python3 deepfeature/clip_model/run.py    <br><br>3. python3 debug.py


For ease of verification, we provide the training log on ImageNet-1K. Please refer to imagenet1k_clip_0.001_10sample_1.txt, lines 23,693–23,695, where the reported TNR@TPR95 reaches 83%, consistent with the result reported in the paper.
