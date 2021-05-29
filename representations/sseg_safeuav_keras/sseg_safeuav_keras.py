import numpy as np
import cv2
import warnings
from .safeuav import get_unet_MDCB_with_deconv_layers
from ..representation import Representation

warnings.filterwarnings("ignore")

def get_disjoint_prediction_fast(prediction_map):
    height, width, nChs = prediction_map.shape
    
    position = np.argmax(prediction_map, axis=2)
    values = np.max(prediction_map, axis=2)
    
    disjoint_map = np.zeros_like(prediction_map)
    
    xx, yy = np.meshgrid(np.arange(height), np.arange(width))
    
    disjoint_map[xx, yy, position.transpose()] =  values.transpose()
    
    return disjoint_map

class SSegSafeUAVKeras(Representation):
    def make(self, t:int) -> np.ndarray:
        orig_img = self.video[t]
        input_img = cv2.resize(orig_img, (self.width, self.height))
        img = (np.float32(input_img) / 255)[None]
        pred = self.model.predict(img)
        result = np.array(pred[0], dtype=np.float32)
        return result

    def makeImage(self, x:np.ndarray) -> np.ndarray:
        predicted_label = get_disjoint_prediction_fast(x)
        predicted_colormap = np.zeros((x.shape[0], x.shape[1], 3), dtype=np.uint8)
        label_indices = predicted_label.argmax(axis=2)

        CLASSES_BGR = ((0, 255, 0), (0, 127, 0), (0, 255, 255), (0, 127, 255), (255, 255, 255), (255, 0, 255), \
            (127, 127, 127), (255, 0, 0), (255, 255, 0), (63, 127, 127), (0, 0, 255), (0, 127, 127))
        for current_prediction_idx in range(self.NUM_CLASSES):
            predicted_colormap[np.nonzero(np.equal(label_indices,current_prediction_idx))] = \
                CLASSES_BGR[current_prediction_idx]
        return predicted_colormap

    def setup(self):
        if hasattr(self, "model"):
            return
        self.NUM_CLASSES = 12
        self.width = 1920
        self.height = 1080
        self.init_nb = 24
        weights_path = '/home/mihai/Public/Projects/ngc/video-representation-extractor/weights/EPOCH_13_TRAIN_loss_0.05107_dice_0.9654_VALID_loss_0.19600_dice_0.8877.hdf5'
        model = get_unet_MDCB_with_deconv_layers(input_shape=(self.height, self.width, 3), init_nb=self.init_nb, num_classes=self.NUM_CLASSES)
        model.load_weights(filepath=weights_path)
        self.model = model
