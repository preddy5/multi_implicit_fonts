
from argparse import ArgumentParser

parser = ArgumentParser(description='Change.')

parser.add_argument('--version', type=str)
parser.add_argument('--config', type=str, default='vanilla')
parser.add_argument('--local_supervision', action='store_true', default=False)

args = parser.parse_args()
