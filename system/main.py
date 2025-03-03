import copy
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
from test import *
from flcore.servers.servercp import FedCP
from flcore.trainmodel.models import *

from utils.mem_utils import MemReporter

warnings.simplefilter("ignore")
torch.manual_seed(0)

# hyper-params for AG News
vocab_size = 98635
max_len=200

hidden_dim=32

def run(args):

    time_list = []
    reporter = MemReporter()
    args.modle_name = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if args.modle_name == "cnn":
            if args.dataset[:5] == "mnist":
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif args.dataset[:5] == "cifar":
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            # elif args.dataset[:13] =="tiny-imagenet":
            #     args.model ==FedAvgResNet18(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif args.modle_name == "resnet":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
            # args.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False).to(args.device)

        elif args.modle_name == "fastText":
            args.model = fastText(hidden_dim=hidden_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(args.device)
        
        else:
            raise NotImplementedError


        if args.algorithm == "FedCP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, args.head)
            server = FedCP(args, i)
        else:
            raise NotImplementedError
            
        server.train(args)
        
        # torch.cuda.empty_cache()

        time_list.append(time.time()-start)

        reporter.report()

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    print("All done!")


if __name__ == "__main__":
    import torch
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-dp','--difference_privacy', type=bool, default=False)
    parser.add_argument('-dpl','--difference_privacy_layer', type=str, default="model.head")
    parser.add_argument('-dpl2', '--difference_privacy_layer2', type=str, default="model.feature_extractor")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-pre', "--pretrain", type=bool, default=False)
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-mn', "--model_name", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=5)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=1000)
    parser.add_argument('-ls', "--local_steps", type=int, default=1)
    parser.add_argument('-algo', "--algorithm", type=str, default="FedCP")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
   
    parser.add_argument('-lam', "--lamda", type=float, default=0.0)



    args = parser.parse_args()
    print(args.difference_privacy)
    print(args.device_id)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id


    # 检查 CUDA 是否可用
    if torch.cuda.is_available():
        print("CUDA 可用")
        print(f"可用的 GPU 数量: {torch.cuda.device_count()}")
        print(f"当前 GPU 设备: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"CUDA 版本: {torch.version.cuda}")
        # 如果你想指定哪块GPU, 比如使用 args.device_id:
        # device = torch.device(f"cuda:{args.device_id}")
        device = torch.device("cuda:0")  # 默认为 0 号 GPU
    else:
        print("CUDA 不可用")
        device = torch.device("cpu")

    # 如果本意是：只有在用户 args.device 指定 "cuda" 时才尝试用 GPU，
    # 若不可用则自动使用 CPU，可以改为：
    '''
    if args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("\ncuda is not available.\n自动切换到 CPU...")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    '''

    print(f"最终使用的计算设备: {device}")
    args.device = device  # 若需要在后续使用

    run(args)