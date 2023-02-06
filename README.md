# image-classification
Traning model for classification images using PyTorch


## scripts
| script              | description                              |
| :------------------ | :--------------------------------------- |
| train.py            | train AlexNet model                      |
| eval.py             | test AlexNet model                       |
| train_resnet.py     | train ResNet model                       |
| train_imbalanced.py | train AlexNet model with imbalanced data |
| choose_file.py      | choose file from directory               |
| img_resize.py       | resize image                             |


## data
Using ImageFolder to load data.
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