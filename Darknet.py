import torch
import torch.nn as nn
import numpy as np
from function import build_targets
from collections import defaultdict
def read_cfg(cfgfile):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

def create_modules(module_defs):
    """
        Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

        elif module_def['type'] == 'upsample':
            upsample = nn.Upsample(scale_factor=int(module_def['stride']),
                                   mode='nearest')
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def["layers"].split(',')]
            filters = sum([output_filters[layer_i] for layer_i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def['classes'])
            img_height = int(hyperparams['height'])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_height)
            modules.add_module('yolo_%d' % i, yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list

class EmptyLayer(nn.Module):
    """
    prepare for shortcut layer / route layer
    """
    def __init__(self):
        super(EmptyLayer, self).__init__()

class YOLOLayer(nn.Module):
    """Detection layer"""
    def __init__(self, anchors, num_classes, img_size):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size
        self.ignore_thres = 0.5 # nms的阈值
        self.lambda_coord = 1   # λ

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, prediction, target = None):
        batch_size = prediction.size(0)
        grid_size = prediction.size(2)
        stride = self.img_size/grid_size
        self.device = torch.device('cuda' if prediction.is_cuda else 'cpu')

        scale_anchors = [(a[0] / stride, a[1] / stride) for a in self.anchors]

        # I don't know the usage of contiguous()
        prediction = prediction.view(batch_size,
                                     self.num_anchors,
                                     self.bbox_attrs,
                                     grid_size,
                                     grid_size).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        Centerx = torch.sigmoid(prediction[..., 0])  # Center x
        Centery = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        prediction = prediction.view(batch_size,
                                     self.num_anchors*grid_size*grid_size,
                                     self.bbox_attrs)

        # Add center offset
        grid_len = np.arange(grid_size)
        a,b = np.meshgrid(grid_len,grid_len)

        x_offset = torch.FloatTensor(a).view(-1, 1).to(self.device)
        y_offset = torch.FloatTensor(b).view(-1, 1).to(self.device)

        x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, self.num_anchors).view(-1, 2).unsqueeze(0)
        prediction[..., :2] += x_y_offset

        # log space transform height and the width
        anchors = torch.FloatTensor(scale_anchors).to(self.device)

        anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
        prediction[..., 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

        # Softmax the class scores ps: Now change it to sigmoid
        prediction[..., 5: 5 + self.num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + self.num_classes]))

        # Add offset and scale with anchors
        pred_bbox = prediction[..., :4].view(batch_size, self.num_anchors,grid_size,grid_size,4)

        # train
        if target is not None:
            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(pred_bbox.cpu().data,
                                                                                        target.cpu().data,
                                                                                        scale_anchors,
                                                                                        self.num_anchors,
                                                                                        self.num_classes,
                                                                                        grid_size,
                                                                                        self.ignore_thres)

            recall = float(nCorrect / nGT) if nGT else 1
            mask = torch.FloatTensor(mask).to(self.device)
            conf_mask = conf_mask.to(self.device)
            cls_mask = mask.unsqueeze(-1).repeat(1, 1, 1, 1, self.num_classes).to(self.device)

            # Handle target variables
            tx = torch.FloatTensor(tx).to(self.device)
            ty = torch.FloatTensor(ty).to(self.device)
            tw = torch.FloatTensor(tw).to(self.device)
            th = torch.FloatTensor(th).to(self.device)
            tconf = torch.FloatTensor(tconf).to(self.device)
            tcls = torch.FloatTensor(tcls).to(self.device)

            # Mask outputs to ignore non-existing objects
            loss_x = self.lambda_coord * self.bce_loss(Centerx * mask, tx * mask)
            loss_y = self.lambda_coord * self.bce_loss(Centery * mask, ty * mask)
            loss_w = self.lambda_coord * self.mse_loss(w * mask, tw * mask) / 2
            loss_h = self.lambda_coord * self.mse_loss(h * mask, th * mask) / 2
            loss_conf = self.bce_loss(conf * conf_mask, tconf * conf_mask)
            loss_cls = self.bce_loss(pred_cls * cls_mask, tcls * cls_mask)
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return loss, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item(), recall

        # test
        else:
            prediction[..., :4] *= stride
            return prediction

class Darknet(nn.Module):
    """YOLOv3 object detection model"""
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = read_cfg(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ['x', 'y', 'w', 'h', 'conf', 'cls', 'recall']

    def forward(self, x, targets=None):
        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample']:
                x = module(x)
            elif module_def['type'] == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'yolo':
                # Train phase: get loss
                if is_training:
                    x, *losses = module[0](x, targets)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module(x)
                output.append(x)
            layer_outputs.append(x)

        self.losses['recall'] /= 3
        return sum(output) if is_training else torch.cat(output, 1)

    def load_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        #Open the weights file
        fp = open(weights_path, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)   # First five are header values

        # Needed to write header when saving weights
        self.header_info = header

        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)         # The rest are weights
        fp.close()

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                if module_def['batch_normalize']:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel() # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_weights(self, path, cutoff=-1):

        fp = open(path, 'wb')
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

# model = Darknet('config/yolov3.cfg')
# input = torch.sigmoid(torch.rand(1, 3, 416, 416).float())
# # model.net_info["height"] = 416
# predictions = model(input)
# print(predictions.shape)
# # print(predictions.shape)
