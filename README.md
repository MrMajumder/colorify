# 🖌️ Colorify - Black and White Images
Implementation of [_Let there be Color!_](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/en/)
by Satoshi Iizuka, Edgar Simo-Serra and Hiroshi Ishikawa, as L4T2 Machine Learning Project, CSE, BUET.

_Colorized 100 years old photo of Książ Castle:_

![Colorized Książ Castle, Poland](colorized/ksiaz-castle.png "Colorized Książ Castle, Poland")

[More images](colorized/colorized.md)

_Other colorized images:_

![Colorized images](colorized/results.png "Sample of colorized images")

## Model Architecture
![Model architecture](colorized/model.png "Parts of the model")

## 📄 Dataset Used
[Places365-Standard](http://places2.csail.mit.edu/download-private.html) 

## 🗃️ Requirements
Code is written in Python 3.7. [Here](requirements.txt) are all requirements, hit the following to install:
```bash
py -3.7 -m pip install -r requirements.txt
```

## ➡️ Usage
Run `main.py` to use the architecture for training and testing. The hyperparameters are hardcoded in this file. 

The default hyperparameters are:
```c
batch size = 32
epoch = 45
learning rate = 0.001
```

the `main.py` file takes 2 arguments:
```
--model : Path to pretrained .pt model
--mode  : train: to train, test: to test, none: train+test
```

We are running it on 10 classes of the [Places365-Standard](http://places2.csail.mit.edu/download-private.html)  dataset. The classes we are using are similar (garden/tree/outdoor) images. These are:
```
botanical_garden  house            roof_garden       zen_garden
cottage           japanese_garden  topiary_garden
formal_garden     lawn             vegetable_garden
```

Total information of train, validation, and test dataset:
```
Train images        : 40960
Validation images   : 1280
Test images         : 960
```

### 🎯 Training
To train the model from scratch, use the following command:
```bash
py -3.7 main.py --mode train
```

To train the model from a pretrained state, use the following command:
```bash
py -3.7 main.py --mode train --model model/places10/ColorifyNet-current.pt
```

### 🧪 Testing
To test the model, use the following command:
```bash
py -3.7 main.py --mode test --model model/places10/ColorifyNet-best.pt
```

## ℹ️ Project Information and Presentation Slides
📃 Slide 1 - [Colorify - ML Project Idea](https://docs.google.com/presentation/d/12aq2LDsEImlDOwldLfFMpNNX0tQObyeZ/edit?usp=sharing&ouid=103881186940791324229&rtpof=true&sd=true)
📃 Slide 2 - [Colorify - Update 1 (Architecture Implemented, Training Started)](https://docs.google.com/presentation/d/1DRkCzScaaBJJcdsakL554Utj6rA-FBcl/edit?usp=sharing&ouid=103881186940791324229&rtpof=true&sd=true)

## 🧑‍💻 Contributors
1. [Sourov Jajodia](https://github.com/Sourov72)
2. [Shafayat Hossain Majumder](https://github.com/MrMajumder)
