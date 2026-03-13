from torch.nn import Module, MaxPool2d, Sequential

class ResNet(Module):
    def __init__(self, encoder_name):
        super(ResNet, self).__init__()
        if encoder_name == "resnet18":
            from torchvision.models import resnet18
            resnet = resnet18(weights='DEFAULT')
        elif encoder_name == "resnet34":
            from torchvision.models import resnet34
            resnet = resnet34(weights='DEFAULT')
        else:
            raise NotImplementedError
        self.inc = Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.pool = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = []
        x = self.inc(x)
        skip.append(x)
        x = self.encoder1(self.pool(x))
        skip.append(x)
        x = self.encoder2(x)
        skip.append(x)
        x = self.encoder3(x)
        skip.append(x)
        x = self.encoder4(x)
        return x, skip[::-1]
