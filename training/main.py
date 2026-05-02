import os
import argparse
from solver import Solver
import torch
import numpy as np
import random
from torch.utils.data import ConcatDataset


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def main(config):
    # path for models
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    
    # import data loader
    if config.dataset == "daicwoz":
        from data_loader.daicwoz_loader import get_audio_loader
        
    if config.dataset == "edaic":
        from data_loader.edaic_loader import get_audio_loader
        
    if config.dataset == "hearing":
        from data_loader.hearing_loader import get_audio_loader
        
    if config.dataset == "eatd":
        from data_loader.eatd_loader import get_audio_loader
        
    if config.dataset == "avec2014_both":
        from data_loader.avec2014_loader_both import get_audio_loader
        
    if config.dataset == "avec2014_freeform":
        from data_loader.avec2014_loader_freeform import get_audio_loader
        
    if config.dataset == "avec2014_northwind":
        from data_loader.avec2014_loader_northwind import get_audio_loader
        

    # get data loder
    print("train loader...")
    train_loader = get_audio_loader(
        config.data_path,
        config.batch_size,
        split="train",
        input_length=config.input_length,
        num_workers=config.num_workers,
    )
    print("valid loader...")
    valid_loader = get_audio_loader(
        config.data_path,
        1,
        split="valid",
        input_length=config.input_length,
        num_workers=config.num_workers,
    )
    print("test loader...")
    test_loader = get_audio_loader(
        config.data_path,
        1,
        split="test",
        input_length=config.input_length,
        num_workers=config.num_workers,
    )
    solver = Solver(train_loader, valid_loader, test_loader, config)
    solver.train()
    solver.final_test()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--dataset",
        type=str,
        default="edaic",
        choices=["daicwoz", "edaic", "hearing", "eatd", "avec2014_both", "avec2014_freeform", "avec2014_northwind"],
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="hubert",
        choices=["CNN235.5k", "speechatt", "ast", "wav2vec", "hubert", "hubert_large", "whisper", "whisper_large"],
    )
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3) # 1e-3
    parser.add_argument("--use_tensorboard", type=int, default=1)
    parser.add_argument("--model_save_path", type=str, default="../models")
    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--log_step", type=int, default=1)
    parser.add_argument("--input_length", type=int, default=16000*10) #기존 16000*10
    parser.add_argument("--map_num", type=int, default=2)
    parser.add_argument("--pad_num", type=int, default=100)
    parser.add_argument("--prompt_len", type=float, default=0)
    parser.add_argument("--training_mode", type=str, default="", choices=["LP", "FT", "RP"])
    parser.add_argument(
        "--reprog_front",
        type=str,
        default="None",
        choices=["None", "uni_noise", "condi", "skip"],
    )

    config = parser.parse_args()

    print(config)
    main(config)
