import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, Input

class BaseModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(BaseModel, self).__init__()
        self.num_classes = num_classes
    
    def call(self, inputs, training=False):
        raise NotImplementedError("Subclasses should implement this method.")

# Define CustomModel1
# model_3 from https://www.kaggle.com/code/devsubhash/cifar-10-image-classification-using-cnn
class CustomModel1(BaseModel):
    def __init__(self, num_classes=10):
        super(CustomModel1, self).__init__(num_classes)
        
        # First block
        self.conv1a = Conv2D(64, (4, 4), activation='relu', padding='same')
        self.bn1a = BatchNormalization()
        self.conv1b = Conv2D(64, (4, 4), activation='relu', padding='same')
        self.bn1b = BatchNormalization()
        self.pool1 = MaxPooling2D(pool_size=(2, 2))
        self.drop1 = Dropout(0.2)
        
        # Second block
        self.conv2a = Conv2D(128, (4, 4), activation='relu', padding='same')
        self.bn2a = BatchNormalization()
        self.conv2b = Conv2D(128, (4, 4), activation='relu', padding='same')
        self.bn2b = BatchNormalization()
        self.pool2 = MaxPooling2D(pool_size=(2, 2))
        self.drop2 = Dropout(0.25)
        
        # Third block
        self.conv3a = Conv2D(128, (4, 4), activation='relu', padding='same')
        self.bn3a = BatchNormalization()
        self.conv3b = Conv2D(128, (4, 4), activation='relu', padding='same')
        self.bn3b = BatchNormalization()
        self.pool3 = MaxPooling2D(pool_size=(2, 2))
        self.drop3 = Dropout(0.35)
        
        # Fully connected layers
        self.flatten = Flatten()
        self.fc1 = Dense(256, activation='relu')
        self.bn_fc1 = BatchNormalization()
        self.drop_fc = Dropout(0.5)
        
        # Output layer
        self.fc2 = Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1a(inputs)
        x = self.bn1a(x, training=training)
        x = self.conv1b(x)
        x = self.bn1b(x, training=training)
        x = self.pool1(x)
        x = self.drop1(x, training=training)

        x = self.conv2a(x)
        x = self.bn2a(x, training=training)
        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = self.pool2(x)
        x = self.drop2(x, training=training)

        x = self.conv3a(x)
        x = self.bn3a(x, training=training)
        x = self.conv3b(x)
        x = self.bn3b(x, training=training)
        x = self.pool3(x)
        x = self.drop3(x, training=training)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn_fc1(x, training=training)
        x = self.drop_fc(x, training=training)

        return self.fc2(x)

# Define a function to dynamically select and create models
def build_model(model_type, input_shape=(32, 32, 3), num_classes=10, compile=False):
    if model_type == 'CustomModel1':
        model = CustomModel1(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type {model_type}")

    inputs = Input(shape=input_shape)
    outputs = model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    if compile:
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
