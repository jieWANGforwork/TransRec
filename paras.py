

import torchvision.transforms as transforms
from parameters import parse_args


args = parse_args()
'''
All pre-trained models expect input images normalized in the same way, 
i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. 
The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
'''

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


bert_finetuneset={
'encoder.layer.10.attention.self.query.weight',
'encoder.layer.10.attention.self.query.bias',
'encoder.layer.10.attention.self.key.weight',
'encoder.layer.10.attention.self.key.bias',
'encoder.layer.10.attention.self.value.weight',
'encoder.layer.10.attention.self.value.bias',
'encoder.layer.10.attention.output.dense.weight',
'encoder.layer.10.attention.output.dense.bias',
'encoder.layer.10.attention.output.LayerNorm.weight',
'encoder.layer.10.attention.output.LayerNorm.bias',
'encoder.layer.10.intermediate.dense.weight',
'encoder.layer.10.intermediate.dense.bias',
'encoder.layer.10.output.dense.weight',
'encoder.layer.10.output.dense.bias',
'encoder.layer.10.output.LayerNorm.weight',
'encoder.layer.10.output.LayerNorm.bias',
'encoder.layer.11.attention.self.query.weight',
'encoder.layer.11.attention.self.query.bias',
'encoder.layer.11.attention.self.key.weight',
'encoder.layer.11.attention.self.key.bias',
'encoder.layer.11.attention.self.value.weight',
'encoder.layer.11.attention.self.value.bias',
'encoder.layer.11.attention.output.dense.weight',
'encoder.layer.11.attention.output.dense.bias',
'encoder.layer.11.attention.output.LayerNorm.weight',
'encoder.layer.11.attention.output.LayerNorm.bias',
'encoder.layer.11.intermediate.dense.weight',
'encoder.layer.11.intermediate.dense.bias',
'encoder.layer.11.output.dense.weight',
'encoder.layer.11.output.dense.bias',
'encoder.layer.11.output.LayerNorm.weight',
'encoder.layer.11.output.LayerNorm.bias',
'pooler.dense.weight',
'pooler.dense.bias',
'rel_pos_bias.weight',
'classifier.weight',
'classifier.bias'}



resnet_finetuneset = {
'layer3.0.conv1.weight',
'layer3.0.bn1.weight',
'layer3.0.bn1.bias',
'layer3.0.conv2.weight',
'layer3.0.bn2.weight',
'layer3.0.bn2.bias',
'layer3.0.downsample.0.weight',
'layer3.0.downsample.1.weight',
'layer3.0.downsample.1.bias',
'layer3.1.conv1.weight',
'layer3.1.bn1.weight',
'layer3.1.bn1.bias',
'layer3.1.conv2.weight',
'layer3.1.bn2.weight',
'layer3.1.bn2.bias',
'layer4.0.conv1.weight',
'layer4.0.bn1.weight',
'layer4.0.bn1.bias',
'layer4.0.conv2.weight',
'layer4.0.bn2.weight',
'layer4.0.bn2.bias',
'layer4.0.downsample.0.weight',
'layer4.0.downsample.1.weight',
'layer4.0.downsample.1.bias',
'layer4.1.conv1.weight',
'layer4.1.bn1.weight',
'layer4.1.bn1.bias',
'layer4.1.conv2.weight',
'layer4.1.bn2.weight',
'layer4.1.bn2.bias',
'fc.weight',
'fc.bias',
}