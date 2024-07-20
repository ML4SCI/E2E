from models.resnet import *
from utils import *


args = get_args_parser()

set_seed(args.seed)

train(args)
    