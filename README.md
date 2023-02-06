# image-classification
Traning model for classification images using PyTorch


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