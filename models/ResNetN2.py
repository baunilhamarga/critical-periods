# Adapted from https://github.com/pytorch/vision/blob/v0.4.0/torchvision/models/resnet.py
import math
from tensorflow import keras
from tensorflow.keras import layers

kaiming_normal = keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')

def conv3x3(x, out_planes, stride=1, name=None):
    x = layers.ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
    return layers.Conv2D(filters=out_planes, kernel_size=3, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=name)(x)

def basic_block(x, planes, stride=1, downsample=None, name=None):
    identity = x

    out = conv3x3(x, planes, stride=stride, name=f'{name}.conv1')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(out)
    out = layers.ReLU(name=f'{name}.relu1')(out)

    out = conv3x3(out, planes, name=f'{name}.conv2')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)

    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)

    out = layers.Add(name=f'{name}.add')([identity, out])
    out = layers.ReLU(name=f'{name}.relu2')(out)

    return out

def make_layer(x, planes, blocks, stride=1, name=None):
    downsample = None
    inplanes = x.shape[3]
    # print(f"stride: {stride} | inplanes: {inplanes} | planes: {planes}")
    if stride != 1 or inplanes != planes:
        downsample = [
            layers.Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=f'{name}.0.downsample.0'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.0.downsample.1'),
        ]

    x = basic_block(x, planes, stride, downsample, name=f'{name}.0')
    for i in range(1, blocks):
        x = basic_block(x, planes, name=f'{name}.{i}')

    return x

def resnet(x, blocks_per_layer, num_classes=10): # Número de classes: Atualizado para 10, correspondendo ao CIFAR-10.
    # x = layers.ZeroPadding2D(padding=3, name='conv1_pad')(x)
    # x = layers.Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False, kernel_initializer=kaiming_normal, name='conv1')(x)
    # x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
    # x = layers.ReLU(name='relu1')(x)
    # x = layers.ZeroPadding2D(padding=1, name='maxpool_pad')(x)
    # x = layers.MaxPool2D(pool_size=3, strides=2, name='maxpool')(x)
    
    # Primeira camada de convolução: Ajustes na primeira camada de convolução são essenciais para adaptar o modelo à natureza das imagens do CIFAR-10, que são significativamente menores que as do ImageNet. Ao usar um kernel de tamanho 3 e um stride de 1, a rede começa com uma operação menos agressiva de redução de dimensão espacial, permitindo que mais detalhes da imagem pequena sejam preservados nas camadas iniciais.
    # Remoção do MaxPooling inicial: No código adaptado, removemos a camada de MaxPooling que seguia a primeira convolução no modelo original. Isso é feito para evitar reduzir prematuramente a resolução espacial da entrada, que já é bastante baixa (32x32 pixels). Em redes projetadas para ImageNet, o MaxPooling é útil para reduzir a resolução após uma convolução inicial grande (7x7 com stride 2), mas essa etapa se torna desnecessária e potencialmente prejudicial para imagens do tamanho do CIFAR-10.

    # Ajusta a camada de entrada e a primeira convolução para o tamanho da imagem CIFAR-10  
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, use_bias=False, kernel_initializer=kaiming_normal, name='conv1', padding = 'same')(x)  
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)  
    # x = layers.ReLU(name='relu1')(x)  

    x = make_layer(x, 64, blocks_per_layer[0], name='layer1')
    x = make_layer(x, 128, blocks_per_layer[1], stride=2, name='layer2')
    x = make_layer(x, 256, blocks_per_layer[2], stride=2, name='layer3')
    x = make_layer(x, 512, blocks_per_layer[3], stride=2, name='layer4')

    # x = layers.GlobalAveragePooling2D(name='avgpool')(x)
    # initializer = keras.initializers.RandomUniform(-1.0 / math.sqrt(512), 1.0 / math.sqrt(512))
    # x = layers.Dense(units=num_classes, kernel_initializer=initializer, bias_initializer=initializer, name='fc')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(units=num_classes, activation=keras.activations.softmax)(x)

    return x

def resnet18(x, **kwargs):
    return resnet(x, [2, 2, 2, 2], **kwargs)

def resnet34(x, **kwargs):
    return resnet(x, [3, 4, 6, 3], **kwargs)
