# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import copy
from ikomia import core, dataprocess, utils
from diffusers import StableDiffusionInpaintPipeline
import torch
import numpy as np
import cv2
from skimage import img_as_float

# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferStableDiffusionInpaintingParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.model_name = "stabilityai/stable-diffusion-2-inpainting"
        self.prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
        self.num_inference_steps = 100
        self.guidance_scale = 7.5
        self.cuda = torch.cuda.is_available()
        self.negative_prompt = ""
        self.num_images_per_prompt = 1
        self.output = "Output resized to input size"

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = str(param_map["model_name"])
        self.prompt = param_map["prompt"]
        self.guidance_scale = float(param_map["guidance_scale"])
        self.num_inference_steps = int(param_map["num_inference_steps"])
        self.negative_prompt = param_map["negative_prompt"]
        self.num_images_per_prompt = int(param_map["num_images_per_prompt"])
        self.output = param_map["output"]
        self.cuda = utils.strtobool(param_map["cuda"])

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["model_name"] = str(self.model_name)
        param_map["prompt"] = str(self.prompt)
        param_map["guidance_scale"] = str(self.guidance_scale)
        param_map["num_inference_steps"] = str(self.num_inference_steps)
        param_map["cuda"] = str(self.cuda)
        param_map["negative_prompt"] = str(self.negative_prompt)
        param_map["num_images_per_prompt"] = str(self.num_images_per_prompt)
        param_map["output"] = str(self.output)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferStableDiffusionInpainting(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the process here   
        # Create parameters class
        if param is None:
            self.set_param_object(InferStableDiffusionInpaintingParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.device = torch.device("cpu")
        self.pipe = None

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        # Get parameters:
        param = self.get_param_object()

        # Get input:
        image_in = self.get_input(0)
        src_image = image_in.get_image()

        # Check input image format and generate binary mask
        src_ini = src_image
        h_ori ,w_ori , _ = src_image.shape

        if src_image.dtype == 'uint8':
            imagef = img_as_float(src_image)
        graph_input = self.get_input(1)
        self.create_graphics_mask(imagef.shape[1], imagef.shape[0], graph_input)
        binimg = self.get_graphics_mask(0)

        if binimg is not None:
            mask_image = cv2.resize(binimg, (512, 512))
        else:
            raise Exception("No graphic input set.")

        # Resize image
        image_input = src_image[:, :, :3]
        image_input = cv2.resize(image_input, (512, 512))

        # Load pipeline
        if param.update or self.pipe is None:
            self.device = torch.device("cuda") if param.cuda else torch.device("cpu")
            torch_tensor_dtype = torch.float16 if param.cuda else torch.float32
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                                    param.model_name,
                                    torch_dtype = torch_tensor_dtype,
                                )

            self.pipe = self.pipe.to(self.device)

        # Inference
        image = self.pipe(
                    prompt=param.prompt,
                    image=image_input,
                    mask_image=mask_image,
                    num_inference_steps=param.num_inference_steps,
                    guidance_scale = param.guidance_scale,
                    num_images_per_prompt  = param.num_images_per_prompt,
                    negative_prompt = param.negative_prompt,
                    ).images

        # Set output(s)
        image_numpy = np.array(image[0].resize((w_ori,h_ori)))
        if param.output == "Burned-in mask":
            Inpainted_img = self.apply_graphics_mask(src_ini, image_numpy, 0)
            output = self.get_output(0)
            output.set_image(Inpainted_img)
        else:
            output = self.get_output(0)
            output.set_image(image_numpy)

        if len(image) > 1:
            for i in range(1,len(image)):
                self.add_output(dataprocess.CImageIO())
                image_numpy = np.array(image[i].resize((w_ori,h_ori)))
                if param.output == "Burned-in mask":
                    Inpainted_img = self.apply_graphics_mask(src_ini, image_numpy, 0)
                    output = self.get_output(i)
                    output.set_image(Inpainted_img)
                else:
                    output = self.get_output(i)
                    output.set_image(image_numpy)

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()

# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferStableDiffusionInpaintingFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_stable_diffusion_inpainting"
        self.info.short_description = "Stable diffusion inpainting models from Hugging Face."
        self.info.description = "This plugin proposes inference for stable diffusion " \
                                "inpainting using diffusion models from Hugging Face."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Diffusion"
        self.info.version = "1.0.0"
        self.info.icon_path = "icons/icon.png"
        self.info.authors = "Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer."
        self.info.article = "High-Resolution Image Synthesis with Latent Diffusion Models"
        self.info.journal = "arXiv"
        self.info.year = 2021
        self.info.license = "MIT License"
        # URL of documentation
        self.info.documentation_link = "https://arxiv.org/pdf/2112.10752.pdf"
        # Code source repository
        self.info.repository = "https://github.com/Stability-AI/stablediffusion"
        # Keywords used for search
        self.info.keywords = "stable diffusion,inpainting,huggingface, Stability-AI"

    def create(self, param=None):
        # Create process object
        return InferStableDiffusionInpainting(self.info.name, param)