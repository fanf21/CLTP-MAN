import argparse
from trainer import trainer_TP

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="./", help="Directory containing logging file")
    parser.add_argument("--loader_num_workers", default=4, type=int)

    parser.add_argument("--delim", default="\t")
    parser.add_argument("--skip", default=1, type=int)
    parser.add_argument("--best_k", default=20, type=int)
    parser.add_argument("--th", default=8.0, type=float)
    parser.add_argument("--CL_flag",  default=True, help='whether continual learning')
    # 进行持续学习时，以['ETH','STU','ZARA']为列，首先预训练ETHautoencoder，然后训练controller得到ETH_ctr
    parser.add_argument("--task_order",  default=['ETH','STU','ZARA'], help='task order')
    parser.add_argument("--model", default='pretrained_models/model_controller/ETH_ctr')# MODEL CONTROLLER

    parser.add_argument("--cuda", default=True)
    parser.add_argument("--batch_size", type=int, default=32)#
    parser.add_argument("--learning_rate", type=int, default=0.0001)#0.0001
    parser.add_argument("--max_epochs", type=int, default=600)#600

    parser.add_argument("--past_len", type=int, default=8)
    parser.add_argument("--future_len", type=int, default=12)#12
    #parser.add_argument("--preds", type=int, default=20)
    parser.add_argument("--dim_embedding_key", type=int, default=48)
#16 /30
    # MODEL CONTROLLER

    parser.add_argument("--saved_memory", default=True) #True

    parser.add_argument("--info", type=str, default='', help='Name of training. '
                                                             'It will be used in tensorboard log and test folder')
    return parser.parse_args()


def main(config):
    t = trainer_TP.Trainer(config)
    print('start training IRM')
    t.fit()


if __name__ == "__main__":
    config = parse_config()
    main(config)
