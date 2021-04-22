mkdir "log"

python main.py --arch ETCModel --model-name DWT_Tiny_Half_R50_Tiny_FPN_CIFAR \
--data-name CIFAR10 --initial-epoch 0 --select-gpu 0 --test True \
>>./log/DWT_Tiny_Half_R50_Tiny_FPN_CIFAR.txt &


echo "[INFO]Starting!"