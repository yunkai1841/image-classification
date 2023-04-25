# image-classification
Traning model for classification images using PyTorch

## My experiment
classify Human vs Stable-diffusion
人間が描いた絵とAIが描いた絵の識別

I trained AlexNet and ResNet aim to detect AI drawn picture on the Internet. 

### Dataset
Human picture: Danbooru2021 dataset

AI picture: Downloaded from pixiv.net
[yunkai1841/pixiv-scraper](https://github.com/yunkai1841/pixiv-scraper)

### Result
Accuracy is ~70%. 
Pictures drawn by humans are classified as AI depending on how they are drawn.
It is better to find a new method to detect AI picture. 

## scripts
| script              | description                              |
| :------------------ | :--------------------------------------- |
| train.py            | train AlexNet model                      |
| train_resnet.py     | train ResNet model                       |
| train_imbalanced.py | train AlexNet model with imbalanced data |
| eval.py             | test AlexNet model                       |
| choose_file.py      | choose file from directory               |
| img_resize.py       | resize image                             |


## data
All train*.py is using ImageFolder class to load data.
So, you need to prepare data like this.  
You can put subdirectory under class directory.

```
data
├── class1
│   ├── img1.jpg
│   ├── img2.jpg
│   ├── img3.jpg
│   └── ...
└── class2
    ├── img1.jpg
    ├── img2.jpg
    ├── img3.jpg
    └── ...
```

## train
1. Prepare data
2. Fix path in train*.py
3. Run train*.py

DataParallel is not recommended because load data is slower than training,
so it can not be faster than single GPU.  
Maybe, it is effective when you store data in M.2 SSD.
