import logging
from ikomia.core import task
from ikomia.utils.tests import run_for_test


logger = logging.getLogger(__name__)


def test(t, data_dict):
    logger.info(f"===== Test::{t.name} =====")
    t.set_parameters({"iteration": 2, "size":3})
    img = cv2.imread(data_dict["images"]["detection"]["coco"])[::-1]
    input_0 = t.getInput(0)
    input_0.setImage(img)
    return run_for_test(t)
