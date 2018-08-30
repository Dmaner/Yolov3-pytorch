import argparse
from torchvision import transforms
import os
from Darknet import *
from datasets import MyTraindataset
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--image_folder', type=str, default='data/samples', help='path to dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of each image batch')
parser.add_argument('--model_config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('--data_config_path', type=str, default='data/obj.data', help='path to data config file')
parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
parser.add_argument('--class_path', type=str, default='data/obj.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between saving model weights')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='directory where model checkpoints are saved')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args()
print(opt)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options

def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names

device = torch.device('cuda' if torch.cuda.is_available() and opt.use_cuda else 'cpu')

os.makedirs('output', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

classes = load_classes(opt.class_path)

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
train_path = data_config['train']

# Get hyper parameters
hyperparams = read_cfg(opt.model_config_path)[0]
learning_rate = float(hyperparams['learning_rate'])
momentum = float(hyperparams['momentum'])
decay = float(hyperparams['decay'])
burn_in = int(hyperparams['burn_in'])

# Initiate model
model = Darknet(opt.model_config_path,img_size=opt.img_size)
#model.load_weights(opt.weights_path)
model.apply(weights_init_normal)

model = model.to(device)
model.train()


# Get dataloader
dataloader = DataLoader(
    MyTraindataset(train_path),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu)

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=0, weight_decay=decay)

for epoch in range(opt.epochs):
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = torch.FloatTensor(imgs).to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()

        print('[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f]' %
                                    (epoch, opt.epochs, batch_i, len(dataloader),
                                    model.losses['x'], model.losses['y'], model.losses['w'],
                                    model.losses['h'], model.losses['conf'], model.losses['cls'],
                                    loss.item(), model.losses['recall']))

        model.seen += imgs.size(0)

    if epoch % opt.checkpoint_interval == 0:
        model.save_weights('%s/%d.weights' % (opt.checkpoint_dir, epoch))
