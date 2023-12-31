import argparse
import itertools
from pathlib import Path
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms

import net
from sampler import InfiniteSamplerWrapper

# Set some configurations
cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated

def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

def adjust_learning_rate(optimizer, iteration_count):
    """Adjust learning rate based on iteration count"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Argument parsing
parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, help='Directory path to COCO2014 data-set')
parser.add_argument('--style_dir', type=str, help='Directory path to Wikiart data-set')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# Training options
parser.add_argument('--training_mode', default='art', help='Artistic or Photo-realistic')
parser.add_argument('--save_dir', default='./experiments', help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs', help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--ccp_weight', type=float, default=5.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--tau', type=float, default=0.07)
parser.add_argument('--num_s', type=int, default=8, help='number of sampled anchor vectors')
parser.add_argument('--num_l', type=int, default=3, help='number of layers to calculate CCPL')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
args = parser.parse_args()

# Device selection
device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

# Load pre-trained models
decoder = net.decoder if args.training_mode == 'art' else nn.Sequential(*list(net.decoder.children())[10:])
vgg = net.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31]) if args.training_mode == 'art' else nn.Sequential(*list(vgg.children())[:18])
network = net.Net(vgg, decoder, args.training_mode)
network.train()
network.to(device)

# Data transformations
content_tf = train_transform()
style_tf = train_transform()

# Load datasets
content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

# Create data iterators
content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

# Optimizer setup
optimizer = torch.optim.Adam(itertools.chain(network.decoder.parameters(), network.SCT.parameters(), network.mlp.parameters()), lr=args.lr)

# Training loop
for i in tqdm(range(args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)

    # Calculate losses
    loss_c, loss_s, loss_ccp = network(content_images, style_images, args.tau, args.num_s, args.num_l)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss_ccp = args.ccp_weight * loss_ccp
    loss = loss_c + loss_s + loss_ccp 

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log losses
    writer.add_scalar('loss_content', loss_c.item(), i + 1)
    writer.add_scalar('loss_style', loss_s.item(), i + 1)
    writer.add_scalar('loss_ccp', loss_ccp.item(), i + 1)

    # Save models
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = net.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'decoder_iter_{:d}.pth.tar'.format(i + 1))
    
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.SCT.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'sct_iter_{:d}.pth.tar'.format(i + 1))                                      
writer.close()
