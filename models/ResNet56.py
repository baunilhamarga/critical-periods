import tensorflow as tf
import argparse
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Flatten

class BasicBlock(tf.keras.layers.Layer):
    
    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.weight_decay = 1e-4
        self.batch_norm_momentum = 0.99
        self._kernel_regularizer = tf.keras.regularizers.l2(self.weight_decay)

        self.conv1 = Conv2D(planes, 3, strides=stride, padding='same', use_bias=False, kernel_regularizer=self._kernel_regularizer)
        self.bn1 = BatchNormalization(-1, self.batch_norm_momentum)
        self.conv2 = Conv2D(planes, 3, strides=1, padding='same', use_bias=False, kernel_regularizer=self._kernel_regularizer)
        self.bn2 = BatchNormalization(-1, self.batch_norm_momentum)

        if stride != 1 or in_planes != planes:
            self.shortcut = lambda x : tf.pad(
                tf.nn.avg_pool2d(x, (2, 2), strides=(1, 2, 2, 1), padding='SAME'),
                [[0, 0], [0, 0], [0, 0], [(planes - in_planes) // 2] * 2])
        else:
            self.shortcut = lambda x: x

    def call(self, x):
        out = tf.nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = tf.nn.relu(out)
        return out

class ResNet(tf.keras.Model):
    # defines a resnet model
    # resnet20: num_block=[3,3,3]
    # resnet32: num_block=[5,5,5]
    # resnet44: num_block=[7,7,7]
    # resnet56: num_block=[9,9,9]
    # resnet110: num_block=[18,18,18]
    # resnet1202: num_block=[200,200,200]

    def __init__(self, input_shape=(32,32,3), num_blocks=[3,3,3]):
        super(ResNet, self).__init__()
        self.weight_decay = 1e-4
        self.batch_norm_momentum = 0.99

        self.input_layer = tf.keras.layers.Input(input_shape)
        self._kernel_regularizer = tf.keras.regularizers.l2(self.weight_decay)
        self.in_planes = 16

        self.conv1 = Conv2D(16, 3, padding='same', use_bias=False, kernel_regularizer=self._kernel_regularizer)
        self.bn1 = BatchNormalization(-1,self.batch_norm_momentum)

        self.layer1 = self._make_layer(16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(64, num_blocks[2], stride=2)

        self.flatten = Flatten()
        self.linear = Dense(10, name="logits", kernel_regularizer=self._kernel_regularizer)

        self.out = self.call(self.input_layer)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes

        return layers

    def call(self, images):
        out = tf.nn.relu(self.bn1(self.conv1(images)))
        for l in self.layer1: out = l(out)
        for l in self.layer2: out = l(out)
        for l in self.layer3: out = l(out)
        out = tf.math.reduce_mean(out,axis=[1,2])
        out = self.flatten(out)
        out = self.linear(out)
        return out
   
if __name__ == '__main__': 
    # Model setup based on the script provided
    MODEL_NAME = "ResNet56"
    NUM_BLOCKS = [9, 9, 9]  # Configuration for ResNet56

    # Initialize the model
    model = ResNet(num_blocks=NUM_BLOCKS)

    # Build the model (this initializes the layers)
    model.build(input_shape=(None, 32, 32, 3))

    # Print the model summary
    print(f"Initializing {MODEL_NAME} with blocks {NUM_BLOCKS}")
    model.summary()

    # Optionally, pass a dummy input to check if the model works as expected
    dummy_input = tf.random.normal((1, 32, 32, 3))  # 1 example, 32x32 image, 3 channels
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")
    # Model setup based on the script provided
    MODEL_NAME = "ResNet56"
    NUM_BLOCKS = [9, 9, 9]  # Configuration for ResNet56

    # Initialize the model
    model = ResNet(num_blocks=NUM_BLOCKS)

    # Build the model (this initializes the layers)
    model.build(input_shape=(None, 32, 32, 3))

    # Print the model summary
    print(f"Initializing {MODEL_NAME} with blocks {NUM_BLOCKS}")
    model.summary()

    # Optionally, pass a dummy input to check if the model works as expected
    dummy_input = tf.random.normal((1, 32, 32, 3))  # 1 example, 32x32 image, 3 channels
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")
    # Model setup based on the script provided
    MODEL_NAME = "ResNet56"
    NUM_BLOCKS = [9, 9, 9]  # Configuration for ResNet56

    # Initialize the model
    model = ResNet(num_blocks=NUM_BLOCKS)

    # Build the model (this initializes the layers)
    model.build(input_shape=(None, 32, 32, 3))

    # Print the model summary
    print(f"Initializing {MODEL_NAME} with blocks {NUM_BLOCKS}")
    model.summary()

    # Optionally, pass a dummy input to check if the model works as expected
    dummy_input = tf.random.normal((1, 32, 32, 3))
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")
    # Model setup based on the script provided
    MODEL_NAME = "ResNet56"
    NUM_BLOCKS = [9, 9, 9]  # Configuration for ResNet56

    # Initialize the model
    model = ResNet(num_blocks=NUM_BLOCKS)

    # Build the model (this initializes the layers)
    model.build(input_shape=(None, 32, 32, 3))

    # Print the model summary
    print(f"Initializing {MODEL_NAME} with blocks {NUM_BLOCKS}")
    model.summary()

    # Example dummy input to ensure the model is built properly
    dummy_input = tf.random.normal((1, 32, 32, 3))
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")
