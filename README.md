<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_hf_stable_diffusion_inpaint/main/icons/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_hf_stable_diffusion_inpaint</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_hf_stable_diffusion_inpaint">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_hf_stable_diffusion_inpaint">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_hf_stable_diffusion_inpaint/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_hf_stable_diffusion_inpaint.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

This algorithm proposes inference for stable diffusion inpainting using diffusion models from Hugging Face.

![Stable diffusion](https://raw.githubusercontent.com/Ikomia-hubinfer_hf_stable_diffusion_inpaint/main/icons/output.jpg)



## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()


sam  = wf.add_task(name = "infer_segment_anything", auto_connect=True)

sam.set_parameters({'model_name':'vit_b',
                    'input_box':'[204.8, 221.8, 769.7, 928.5]'
})

sd_inpaint = wf.add_task(name = "infer_hf_stable_diffusion_inpaint", auto_connect=True)

sd_inpaint.set_parameters({'prompt' :'dog, high resolution'})

# Run directly on your image
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_cat.jpg")

# Inspect your result
display(sam.get_image_with_mask())
display(sd_inpaint.get_output(0).get_image())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- **model_name** (str) - default 'stabilityai/stable-diffusion-2-inpainting': Name of the stable diffusion model. Other model available: 'runwayml/stable-diffusion-inpainting'
- **prompt** (str): Input prompt.
- **negative_prompt** (str, *optional*): The prompt not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
- **num_inference_steps** (int) - default '50': Number of denoising steps (minimum: 1; maximum: 500).
- **guidance_scale** (float) - default '7.5': Scale for classifier-free guidance (minimum: 1; maximum: 20).
- **num_images_per_prompt** (int) - default '1': Number of output.


```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()


sam  = wf.add_task(name = "infer_segment_anything", auto_connect=True)

sam.set_parameters({
        'model_name':'vit_b',         
        'input_box':'[204.8, 221.8, 769.7, 928.5]',                 
})

sd_inpaint = wf.add_task(name = "infer_hf_stable_diffusion_inpaint", auto_connect=True)

sd_inpaint.set_parameters({
                'prompt' :'dog, high resolution',
                'negative_prompt':'low quality',
                'num_inference_steps':'100',
                'guidance_scale':'7.5',
                'num_images_per_prompt':'1',
})

# Run directly on your image
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_cat.jpg")

# Inspect your result
display(sam.get_image_with_mask())
display(sd_inpaint.get_output(0).get_image())
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
import ikomia
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
sam  = wf.add_task(name = "infer_segment_anything", auto_connect=True)

sam.set_parameters({'model_name':'vit_b',
                    'input_box':'[204.8, 221.8, 769.7, 928.5]',
                    
})
sd_inpaint = wf.add_task(name = "infer_hf_stable_diffusion_inpaint", auto_connect=True)

sd_inpaint.set_parameters({'prompt' :'dog, high resolution'})

# Run on your image  
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_cat.jpg")

# Iterate over outputs
for output in sd_inpaint.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

## :fast_forward: Advanced usage 

Inpainting can be done from a graphic input (e.g. with Ikomia STUDIO), a semantic segmantation or a instance segmenation mask.
For more information on the infer_stable_diffusion_inpaint algorithm check out the blog post [Easy stable diffusion inpainting with Segment Anything Model (SAM)](https://www.ikomia.ai/blog/stable-diffusion-inpainting-with-segment-anything-model-sam-using-the-ikomia-api).
