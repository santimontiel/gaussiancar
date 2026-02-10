import cv2
import numpy as np

from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from shapely.geometry import LineString


# Fix to use Shapely>2.0.0
class FixedMapExplorer(NuScenesMapExplorer):
    @staticmethod
    def mask_for_lines(lines: LineString, mask: np.ndarray) -> np.ndarray:
        """
        Convert a Shapely LineString back to an image mask ndarray.
        :param lines: List of shapely LineStrings to be converted to a numpy array.
        :param mask: Canvas where mask will be generated.
        :return: Numpy ndarray line mask.
        """
        if lines.geom_type == 'MultiLineString':
            for line in lines.geoms:
                coords = np.asarray(list(line.coords), np.int32)
                coords = coords.reshape((-1, 2))
                cv2.polylines(mask, [coords], False, 1, 2)
        else:
            coords = np.asarray(list(lines.coords), np.int32)
            coords = coords.reshape((-1, 2))
            cv2.polylines(mask, [coords], False, 1, 2)

        return mask

class FixedNuScenesMap(NuScenesMap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explorer = FixedMapExplorer(self)