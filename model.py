import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from efficientnet_pytorch import EfficientNet

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv2d_1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv2d_3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.conv2d_4 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        self.dropout_1 = nn.Dropout(0.5)
        self.dropout_2 = nn.Dropout(0.5)
        self.dropout_3 = nn.Dropout(0.5)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.fc_1 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = F.relu(x)

        x = self.conv2d_2(x)
        x = F.relu(x)
        x = self.dropout_1(x)

        x = self.conv2d_3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout_2(x)

        x = self.conv2d_4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout_3(x)

        x = self.avgpool(x)

        x = self.flat(x)
        x = self.fc_1(x)
        return x

class MaskClassification(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


class AgeGenderClassfication(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)

        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)

        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)
        self.fc9 = nn.Linear(2622, num_classes)

    def forward(self, x):
        """ Pytorch forward
        Args:
            x: input image (224x224)
        Returns: class logits
        """
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5, self.training)

        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.5, self.training)

        x = F.relu(self.fc8(x))
        x = F.dropout(x, 0.5, self.training)

        return self.fc9(x)


class DCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(512, 256)
        self.fc_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn4(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.bn5(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        
        x = self.flat(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc_bn(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class Ensemble(nn.Module):
    def __init__(self, mask_num_classes, age_gender_num_classes):
        super().__init__()
        self.mask_n_cl = mask_num_classes
        self.age_gende_n_cl = age_gender_num_classes

        self.MaskModel = MaskClassification(mask_num_classes)
        self.AgeGenderModel = AgeGenderClassfication(age_gender_num_classes)
        self.Dcnn = DCNN(age_gender_num_classes)

    def forward(self, x):
        mask_out = self.MaskModel(x)
        age_gender_out = self.Dcnn(x)

        return mask_out, age_gender_out

class DensNet(nn.Module):
    def __init__(self, num_classes=1000, num_channels=3):
        super().__init__()
        preloaded = torchvision.models.densenet121(pretrained=True)
        self.features = preloaded.features
        self.features.conv0 = nn.Conv2d(num_channels, 64, 7, 2, 3)
        self.classifier = nn.Linear(1024, num_classes, bias=True)
        del preloaded
        
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out

class CNN_Model(nn.Module):
    def __init__(self, num_classes, rate=0.2):
        super(CNN_Model, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3')
        self.dropout = nn.Dropout(rate)
        self.output_layer = nn.Linear(in_features=1000, out_features=num_classes, bias=True)

    def forward(self, inputs):
        output = self.output_layer(self.dropout(self.model(inputs)))
        return output