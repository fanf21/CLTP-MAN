import argparse

import torch

from trainer import trainer_controllerMem


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="./", help="Directory containing logging file")
    parser.add_argument("--loader_num_workers", default=0, type=int)#4
    parser.add_argument("--dataset_name", default="ETH", type=str)
    parser.add_argument("--delim", default="\t")

    parser.add_argument("--skip", default=1, type=int)
    parser.add_argument("--th", default =8.0, type=float)
    # 6.5:800,7:

    parser.add_argument("--cuda", default=True)
    parser.add_argument("--batch_size", type=int, default=32) # 32
    parser.add_argument("--learning_rate", type=int, default=0.0001)
    parser.add_argument("--max_epochs", type=int, default=100) # 600

    parser.add_argument("--past_len", type=int, default=8)
    parser.add_argument("--future_len", type=int, default=12)#12
    parser.add_argument("--best_k", type=int, default=20)  # 1
    parser.add_argument("--dim_embedding_key", type=int, default=48)

    parser.add_argument("--model_ae", default='pretrained_models/model_AE/ETH')
    #parser.add_argument("--dataset_file", default="kitti_dataset.json", help="dataset file")
    parser.add_argument("--info", type=str, default='', help='Name of training. '
                                                             'It will use in tensorboard log and test folder')
    return parser.parse_args()


def main(config):
    print('Start training writing controller')
    t = trainer_controllerMem.Trainer(config)
    t.fit()

if __name__ == "__main__":

    config = parse_config()
    main(config)
