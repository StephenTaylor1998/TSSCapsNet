<h1 align="center"> Time Series Signal CapsNet </h1>

# 1.0 Getting Started

## 1.1 Installation

Python3 and Tensorflow 2.x are required and should be installed on the host machine following the [official guide](https://www.tensorflow.org/install). Good luck with it!

1. Clone this repository
   ```bash
   git clone https://github.com/StephenTerror/CapsNet.git
   ```
2. Install the required packages
   ```bash
   pip3 install -r tutorial/requirements.txt
   ```
Peek inside the requirements file if you have everything already installed. Most of the dependencies are common libraries.

## 1.2 run model 

### with terminal args

   ```bash
  # CapsuleNet
  python main.py --arch CapsNet --model-name CapsNet --data-name CIFAR10 --initial-epoch 0 --select-gpu 0 --test True
   ```
   ```bash
  # ResNet 
  # (will be implement soon)   :)
  # know please use this
  python train_resnet.py
   ```

### train without args

   ```bash
   python train.py
   ```

### test without args

   ```bash
   python test.py
   ```

# Citation
Use this bibtex if you enjoyed this repository and you want to cite it:

```
@article{mazzia2021efficient,
  title={Efficient-CapsNet: Capsule Network withSelf-Attention Routing},
  author={Mazzia, Vittorio and Salvetti, Francesco and Chiaberge, Marcello},
  year={2021},
  journal={arXiv preprint arXiv:2101.12491},
}
```
