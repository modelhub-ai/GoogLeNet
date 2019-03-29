from modelhublib.processor import ImageProcessorBase
import PIL
import numpy as np
import json
import skimage.io
import caffe

class ImageProcessor(ImageProcessorBase):

    def _preprocessBeforeConversionToNumpy(self, image):
        if isinstance(image, PIL.Image.Image):
            color = True
            image = skimage.img_as_float( image ).astype(np.float32)
            if image.ndim == 2:
                image = image[:, :, np.newaxis]
                if color:
                    image = np.tile(image, (1, 1, 3))
            elif image.shape[2] == 4:
                image = image[:, :, :3]
        else:
            raise IOError("Image Type not supported for preprocessing.")
        return image

    def _preprocessAfterConversionToNumpy(self, npArr):
        return npArr

    def computeOutput(self, inferenceResults):
        probs = np.squeeze(np.asarray(inferenceResults))
        with open("model/labels.json") as jsonFile:
            labels = json.load(jsonFile)
        result = []
        for i in range (len(probs)):
            obj = {'label': str(labels[str(i)]),
                    'probability': float(probs[i])}
            result.append(obj)
        return result
