mkdir "log"

python main.py --arch ETCModel --model-name RESNET18 \
--data-name CIFAR10 --initial-epoch 0 --select-gpu 1 --test True \
>>./log/RESNET18.txt &

python main.py --arch ETCModel --model-name RESNET34 \
--data-name CIFAR10 --initial-epoch 0 --select-gpu 2 --test True \
>>./log/RESNET34.txt &

python main.py --arch ETCModel --model-name RESNET50 \
--data-name CIFAR10 --initial-epoch 0 --select-gpu 3 --test True \
>>./log/RESNET50.txt &

python main.py --arch ETCModel --model-name RESNET18_DWT_Tiny \
--data-name CIFAR10 --initial-epoch 0 --select-gpu 4 --test True \
>>./log/RESNET18_DWT_Tiny.txt &

python main.py --arch ETCModel --model-name RESNET34_DWT_Tiny \
--data-name CIFAR10 --initial-epoch 0 --select-gpu 5 --test True \
>>./log/RESNET34_DWT_Tiny.txt &

python main.py --arch ETCModel --model-name RESNET50_DWT_Tiny \
--data-name CIFAR10 --initial-epoch 0 --select-gpu 6 --test True \
>>./log/RESNET50_DWT_Tiny.txt &

python main.py --arch ETCModel --model-name RESNET18_DWT_Tiny_Half \
--data-name CIFAR10 --initial-epoch 0 --select-gpu 7 --test True \
>>./log/RESNET18_DWT_Tiny_Half.txt &

python main.py --arch ETCModel --model-name RESNET34_DWT_Tiny_Half \
--data-name CIFAR10 --initial-epoch 0 --select-gpu 8 --test True \
>>./log/RESNET34_DWT_Tiny_Half.txt &

python main.py --arch ETCModel --model-name RESNET50_DWT_Tiny_Half \
--data-name CIFAR10 --initial-epoch 0 --select-gpu 9 --test True \
>>./log/RESNET50_DWT_Tiny_Half.txt &

echo "[INFO]Starting!"
