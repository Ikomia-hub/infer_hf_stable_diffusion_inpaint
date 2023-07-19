# infer_hf_stable_diffusion_inpaint


## :rocket: Inference with Ikomia API

``` python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils import ik
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

sam = wf.add_task(ik.infer_segment_anything(
    model_name='vit_b',
    image_path='./Path/To/Your/Image'),
    auto_connect=True
)

sd_inpaint = wf.add_task(ik.infer_hf_stable_diffusion_inpaint(
    model_name = 'stabilityai/stable-diffusion-2-inpainting',
    prompt = 'Face of a yellow cat, high resolution, sitting on a park bench',
    negative_prompt = 'low quality',
    num_inference_steps = 100,
    guidance_scale = 7.5,
    num_images_per_prompt = 1,
    ),
    auto_connect=True
)

# Run directly on your image
wf.run_on(path='./Path/To/Your/Image')

display(sam.get_image_with_mask())
display(sd_inpaint.get_output(0).get_image())

```