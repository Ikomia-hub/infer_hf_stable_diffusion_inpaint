import logging
from ikomia.utils.tests import run_for_test
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def test(t, data_dict):
    logger.info(f"===== Test::{t.name} =====")
    t.set_parameters({"iteration": "2", "size":"3"})
    img = cv2.imread(data_dict["images"]["detection"]["coco"])[::-1]
    shape = np.shape(img)[:2]
    mask_example = np.zeros(shape, dtype="uint8")
    mask_example[:100,:100] = 1
    input_0 = t.get_input(0)
    input_0.set_image(img)

    graphic_input = t.get_input(2)
    graphic_input.init("HF SD", 0, *shape)
    graphic_input.add_object(0, 0, 0, "other", 1.0, 0, 0, 100, 100, mask_example, [0,0,0])
    return run_for_test(t)
