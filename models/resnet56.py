import tensorflow as tf
from tensorflow.keras import layers, models

# 3x3 Convolution
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return layers.Conv2D(out_planes, kernel_size=3, strides=stride, padding='same', use_bias=False)

# Basic block
class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, inplanes, midplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, midplanes, stride)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.conv2 = conv3x3(midplanes, planes)
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

        self.shortcut = models.Sequential()
        if stride != 1 or inplanes != planes:
            if stride != 1:
                self.shortcut.add(layers.Lambda(lambda x: tf.image.resize(x, [x.shape[1]//2, x.shape[2]//2])))
            self.shortcut.add(layers.Conv2D(planes, kernel_size=1, strides=stride, use_bias=False))

    def call(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu2(out)

        return out

# ResNet Model
class ResNet(tf.keras.Model):
    def __init__(self, block, num_layers, sparsity, num_classes=10):
        super(ResNet, self).__init__()
        assert (num_layers - 2) % 6 == 0, 'Depth should be 6n+2'
        n = (num_layers - 2) // 6

        self.conv1 = layers.Conv2D(16, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()

        # Create layers based on the BasicBlock
        self.layer1 = self._make_layer(block, 16, 16, n, stride=1)
        self.layer2 = self._make_layer(block, 16, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 32, 64, n, stride=2)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def _make_layer(self, block, inplanes, planes, blocks_num, stride):
        layers = []
        layers.append(block(inplanes, inplanes, planes, stride))
        for _ in range(1, blocks_num):
            layers.append(block(planes, planes, planes))
        return models.Sequential(layers)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

def resnet_56(sparsity, num_classes=10):
    return ResNet(BasicBlock, 56, sparsity, num_classes)

if __name__ == '__main__':
    sparsity = [0.0] * 3  # Adjust this based on your sparsity needs
    model = resnet_56(sparsity, num_classes=10)

    # Compile and use the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Model summary
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()