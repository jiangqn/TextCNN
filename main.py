import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='train', choices=['train', 'test', 'predict'])
parser.add_argument('--gpu', type=int, default=0, choices=[i for i in range(8)])
parser.add_argument('--config', type=str, default='config.yaml')

args = parser.parse_args()

config = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
config['gpu'] = args.gpu

if args.task == 'train':
    from src.train_text_cnn import train_text_cnn
    train_text_cnn(config)
elif args.task == 'test':
    from src.test_text_cnn import test_text_cnn
    test_text_cnn(config)