# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fwang1412@gmail.com
@Time       : 2020/12/10 0:30
@Description: 
"""
import json
import argparse

from src.train.run_cross_validation import cross_validation
from src.train.run_train_test import train_and_test
from src.train.run_pretraining import pretrain

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True, help='experiment to run')
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    args = parser.parse_args()

    # cross validation on wikitables
    if args.exp == 'cross_validation':
        config = json.load(open(args.config))
        cross_validation(config)

    # train and test on webquerytable
    elif args.exp == 'train_test':
        config = json.load(open(args.config))
        train_and_test(config)

    # pretrain
    elif args.exp == 'pretrain':
        config = json.load(open(args.config))
        pretrain(config)
