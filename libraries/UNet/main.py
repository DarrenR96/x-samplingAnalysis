from multiprocessing import pool
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

# Different PoolTypes:
# "trainable" : filter with trainable weights
# "MaxPool" : Max value over filter window 
# "AvgPool" : Average Value over filter window 
# "Gaussain" : Gaussain Kernel

poolType = ["trainable", "MaxPool", "AvgPool", "Gaussian"]

class UNetBlockDownSample(layers.Layer):
    def __init__(self, numFilters, size=4, strides=(2, 2), padding='same',poolType='trainable', **kwargs):
        super().__init__()
        # Class variables
        self.numFilters = numFilters
        self.size = size
        self.strides = strides
        self.padding = padding
        self.poolType = poolType
        
        # Layers
        self.conv1 = layers.Conv2D(self.numFilters, self.size, (1,1), "same")
        self.relu1 = layers.LeakyReLU()
        self.conv2 = layers.Conv2D(self.numFilters, self.size, (1,1), "same")
        self.relu2 = layers.LeakyReLU()
        self.convP = layers.Conv2D(self.numFilters, self.size, (1,1), "same")
        self.reluP = layers.LeakyReLU()

        self.add = layers.Add()

        # Choosing downsample layer type
        if self.poolType == "trainable":
            self.outputConv = layers.Conv2D(self.numFilters, self.size, (2,2), "same")
            self.outputRelu = layers.LeakyReLU()
        if self.poolType == "MaxPool":
            self.outputPool = layers.MaxPooling2D((4,4),(2,2), padding="same")
        if self.poolType == "AvgPool":
            self.outputPool = layers.AveragePooling2D((4,4),(2,2), padding="same")
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'numFilters' : self.numFilters,
            'size' : self.size,
            'strides' : self.strides,
            'padding' : self.padding
        })
        return config
    
    def call(self, inputs, training=False):
        # Preceeding layer tensor processing
        x_0 = self.conv1(inputs,training=training)
        x_0 = self.relu1(x_0,training=training)
        x_0 = self.conv2(x_0,training=training)
        x_0 = self.relu2(x_0, training=training)
        x_1 = self.convP(inputs,training=training)
        x_1 = self.reluP(x_1,training=training)
        x = self.add([x_0, x_1], training=training)

        # Downsample layer tensor processing
        if self.poolType == "trainable":
            x = self.outputConv(x,training=training)
            x = self.outputRelu(x,training=training)
        if self.poolType == "MaxPool":
            x = self.outputPool(x, training=training)
        if self.poolType == "AvgPool":
            x = self.outputPool(x, training=training)
        return x


        

class UNetBlockUpSample(layers.Layer):
    def __init__(self, numFilters, size=4, strides=(2, 2), padding='same', upsampleType='trainable', **kwargs):
        super().__init__()
        self.numFilters = numFilters
        self.size = size
        self.strides = strides
        self.padding = padding
        self.upsampleType = upsampleType
        
        self.conv1 = layers.Conv2D(self.numFilters, self.size, (1,1), "same")
        self.relu1 = layers.LeakyReLU()
        self.conv2 = layers.Conv2D(self.numFilters, self.size, (1,1), "same")
        self.relu2 = layers.LeakyReLU()
        self.convP = layers.Conv2D(self.numFilters, self.size, (1,1), "same")
        self.reluP = layers.LeakyReLU()
        self.add = layers.Add()

        if self.upsampleType == 'trainable':
            self.outputConv = layers.Conv2DTranspose(self.numFilters, self.size, (2,2), "same")
            self.outputRelu = layers.LeakyReLU()
        if self.upsampleType == 'Repeat':
            self.upSample = layers.UpSampling2D()
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'numFilters' : self.numFilters,
            'size' : self.size,
            'strides' : self.strides,
            'padding' : self.padding
        })
        return config
    
    def call(self, inputs, training=False):
        x_0 = self.conv1(inputs,training=training)
        x_0 = self.relu1(x_0,training=training)
        x_0 = self.conv2(x_0,training=training)
        x_0 = self.relu2(x_0, training=training)
        x_1 = self.convP(inputs,training=training)
        x_1 = self.reluP(x_1, training=training)
        x = self.add([x_0, x_1], training=training)

        if self.upsampleType == 'trainable':
            x = self.outputConv(x, training=training)
            x = self.outputRelu(x, training=training)
        if self.upsampleType == 'Repeat':
            x = self.upSample(x, training=training)
        return x

class UNet(keras.Model):
    def __init__(self, filters=[16,32,64,128,256,512], inputSize=192, outputSize=3, poolType='MaxPool', upsampleType='trainable'):
        super().__init__()
        self.filters = filters
        self.network = []
        self.outputSize = outputSize
        self.inputSize = inputSize

        for filter in self.filters:
            self.network.append(UNetBlockDownSample(filter, poolType=poolType))
        
        for filter in self.filters[::-1][1:]:
            self.network.append(UNetBlockUpSample(filter, upsampleType=upsampleType))

        self.finalConv2D = layers.Conv2DTranspose(3,4,(2,2),padding='same')
        self.finalRelu = layers.LeakyReLU()

    def call(self, x, training=False):
        xIn = x
        networkDown = []
        for count, filter in enumerate(self.network):
            if count < len(self.filters)-1:
                x = filter(x,training=training)
                networkDown.append(x)
            if count == len(self.filters)-1:
                x = filter(x,training=training)
            if count >= len(self.filters):
                x = filter(x,training=training)
                x = layers.Concatenate()([x, networkDown.pop()])

        x = self.finalConv2D(x,training=training)
        x = self.finalRelu(x,training=training)
        x = layers.Add()([xIn, x])
        return x
    
    def model(self):
        x = keras.Input(shape=(self.inputSize,self.inputSize,3))
        return keras.Model(inputs=[x], outputs=self.call(x))
