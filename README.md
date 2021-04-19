<h1 align="center"> Time Series Signal CapsNet </h1>

# 1.0 Getting Start

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

## 1.2 Run Model 

### With terminal args
   GOTO "./utils/argparse.py" FOR DETAILS.
   ```bash
   # CapsuleNet
   python main.py --arch CapsNet --model-name CapsNet --data-name MNIST --initial-epoch 0 --select-gpu 0 --test True --heterogeneous False --optimizer Adam
   # Efficient_CapsNet
   python main.py --arch Efficient_CapsNet --model-name Efficient_CapsNet --data-name MNIST --initial-epoch 0 --select-gpu 0 --test True --heterogeneous False --optimizer Adam
   # TSSCapsNet
   python main.py --arch TSSCapsNet --model-name DWT_FPN_MNIST --data-name MNIST --initial-epoch 0 --select-gpu 0 --test True --heterogeneous False --optimizer Adam
   # ETCModel
   python main.py --arch ETCModel --model-name RESNET_DWT50_Tiny --data-name CIFAR10 --initial-epoch 0 --select-gpu 0 --test True --heterogeneous False --optimizer Adam
  
   ```


### Without Args

   ```bash
   # Tips: Choose Paramters You Want in Before Running!
   # train model
   python train.py
   # test model
   python test.py
   # ResNet 
   python train_caps_without_decoder.py
   ```

# Citation
Use this bibtex if you enjoyed this repository and you want to cite it:

```
@article{mazzia2021efficient,
  title={Efficient-CapsNet: Capsule Network withSelf-Attention RoutingTiny},
  author={Mazzia, Vittorio and Salvetti, Francesco and Chiaberge, Marcello},
  year={2021},
  journal={arXiv preprint arXiv:2101.12491},
}
```
