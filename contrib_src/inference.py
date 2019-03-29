import caffe
import json
from processing import ImageProcessor
from modelhublib.model import ModelBase
import numpy as np


class Model(ModelBase):

    def __init__(self):
        # load config file
        config = json.load(open("model/config.json"))
        # get the image processor
        self._imageProcessor = ImageProcessor(config)
        # load the DL model
        self._model = caffe.Net('model/model.prototxt',
                        'model/model.caffemodel',
                        caffe.TEST)
        # load input and configure preprocessing
        self._transformer = caffe.io.Transformer({'data': self._model.blobs['data'].data.shape})
        self._transformer.set_mean('data', np.load('model/ilsvrc_2012_mean.npy').mean(1).mean(1))
        self._transformer.set_transpose('data', (2,0,1))
        self._transformer.set_channel_swap('data', (2,1,0))
        self._transformer.set_raw_scale('data', 255.0)

    def infer(self, input):
        # load preprocessed input
        inputAsNpArr = self._imageProcessor.loadAndPreprocess(input)
        # Run inference with caffe
        self._model.blobs['data'].reshape(1,3,224,224)
        self._model.blobs['data'].data[...] = self._transformer.preprocess('data', inputAsNpArr)
        results = self._model.forward()['prob']
        # postprocess results into output
        output = self._imageProcessor.computeOutput(results)
        return output
