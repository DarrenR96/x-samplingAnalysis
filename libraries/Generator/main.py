from tensorflow.keras.utils import Sequence
import numpy as np
import random


def readYUV420PatchRange(name: str, resolution: tuple, range: tuple, upsampleUV: bool = True, patchDim=192, patchStart=(0,0)):
    height = resolution[0]
    width = resolution[1]
    bytesY = int(height * width)
    bytesUV = int(bytesY/4)
    Y = []
    U = []
    V = []
    with open(name,"rb") as yuvFile:
        startLocation = range[0]
        endLocation = range[1] + 1
        startLocationBytes = startLocation * (bytesY + 2*bytesUV)
        endLocationBytes = endLocation * (bytesY + 2*bytesUV)
        data = np.fromfile(yuvFile, np.uint8, endLocationBytes-startLocationBytes, offset=startLocationBytes).reshape(-1,bytesY + 2*bytesUV)
        Y = np.reshape(data[:, :bytesY], (-1, width, height))
        U = np.reshape(data[:, bytesY:bytesY+bytesUV], (-1, width//2, height//2))
        V = np.reshape(data[:, bytesY+bytesUV:bytesY+2*bytesUV], (-1, width//2, height//2))
    if upsampleUV:
        U = U.repeat(2, axis=1).repeat(2, axis=2)
        V = V.repeat(2, axis=1).repeat(2, axis=2)
    YUV = np.stack([Y,U,V],axis=-1)
    YUV = YUV[:, patchStart[0]:patchStart[0]+patchDim, patchStart[1]:patchStart[1]+patchDim, :].astype(np.float32)
    return YUV


class hrNetDataGenerator(Sequence):
    def __init__(self, degradedPaths, referencePaths, frames, batch_size, dim=(192,192,3), sourceDims=(1920,1080), shuffle=True):
        self.degradedPaths = degradedPaths
        self.referencePaths = referencePaths
        self.frames = frames
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.sourceDims = sourceDims
        self.on_epoch_end()

    def __len__(self):
        # Return number of batches in one epoch
        return int(np.floor(len(self.degradedPaths)/self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        _degradedPaths = []
        _referencePaths = []
        _frameRange = []
        _patchLoc = []

        for i in indexes:
            _degradedPaths.append(self.degradedPaths[i])
            _referencePaths.append(self.referencePaths[i])

            _upperLimitFrame = self.frames[i]
            _frame = random.randint(0 ,_upperLimitFrame - 3)
            _frameRange.append((_frame,_frame+2))

            _upperLimitWidth = self.sourceDims[0] - self.dim[0] - 1 
            _upperLimitHeight = self.sourceDims[1] - self.dim[1] - 1
            _patchLoc.append((random.randint(0,_upperLimitHeight),random.randint(0,_upperLimitWidth)))
        X, y = self.__data_generation(_degradedPaths,_referencePaths,_frameRange,_patchLoc)

        return X, y

    def on_epoch_end(self):
        # Update and shuffle epochs after every step
        self.indexes = np.arange(len(self.degradedPaths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self,_degradedPaths,_referencePaths,_frameRange,_patchLoc):
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], 3))
        y = np.empty((self.batch_size, self.dim[0], self.dim[1], 3))
        for i, (_deg, _ref, _range, _patch) in enumerate(zip(_degradedPaths,_referencePaths,_frameRange,_patchLoc)):
            X[i,] = readYUV420PatchRange(_deg,self.sourceDims,(_range[0]+1,_range[0]+1),True,self.dim[0],_patch)
            y[i,] = readYUV420PatchRange(_ref,self.sourceDims,(_range[0]+1,_range[0]+1),True,self.dim[0],_patch)
        return X, y
