from detectron2.data.transforms.augmentation import Augmentation #, _transform_to_aug
from detectron2.data.transforms.transform import ExtentTransform, ResizeTransform, RotationTransform

import numpy as np
import os, json, cv2, random, torch
import torch.nn.functional as F
from fvcore.transforms.transform import (
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    Transform,
    TransformList,
)
from PIL import Image


class custom_ExtentTransform(Transform):
    """
    Extracts a subregion from the source image and scales it to the output size.
    The fill color is used to map pixels from the source rect that fall outside
    the source image.
    See: https://pillow.readthedocs.io/en/latest/PIL.html#PIL.ImageTransform.ExtentTransform
    """

    def __init__(self, src_rect, output_size, interp=Image.LINEAR, fill=0):
        """
        Args:
            src_rect (x0, y0, x1, y1): src coordinates
            output_size (h, w): dst image size
            interp: PIL interpolation methods
            fill: Fill color used when src_rect extends outside image
        """
        super().__init__()
        self._set_attributes(locals())

    def multi_slice_apply_image(self, img, interp=None):
        h, w = self.output_size
        img = img.astype('uint8')
        #print('inner talk111',img.shape,img.dtype)
        
        pil_image = Image.fromarray(img)
        #print('came upto thos', type(pil_image))
        pil_image = pil_image.transform(
            size=(w, h),
            method=Image.EXTENT,
            data=self.src_rect,
            resample=interp if interp else self.interp,
            fill=self.fill,
        )
        ret = np.asarray(pil_image)
        return ret
    
    def apply_image(self, img, interp=None):
        h, w = self.output_size
        if len(img.shape) > 2 and img.shape[2] == 1:
            pil_image = Image.fromarray(img[:, :, 0], mode="L")
            pil_image = pil_image.transform(
                size=(w, h),
                method=Image.EXTENT,
                data=self.src_rect,
                resample=interp if interp else self.interp,
                fill=self.fill,
            )
            ret = np.asarray(pil_image)
        else:
            #pil_image = Image.fromarray(img)
            #print('hahahohoh', img.shape)
            pil_image1 = self.multi_slice_apply_image(img[:,:,0:3],None)
            pil_image2 = self.multi_slice_apply_image(img[:,:,3:6],None)
            pil_image3 = self.multi_slice_apply_image(img[:,:,6:9],None)
            pil_image4 = self.multi_slice_apply_image(img[:,:,9:12],None)
            pil_image5 = self.multi_slice_apply_image(img[:,:,12:15],None)
            #pil_image6 = self.multi_slice_apply_image(img[:,:,15:18],None)
            #pil_image7 = self.multi_slice_apply_image(img[:,:,18:21],None)
            #pil_image8 = self.multi_slice_apply_image(img[:,:,21:24],None)
            #pil_image9 = self.multi_slice_apply_image(img[:,:,24:27],None)
            #ret =np.concatenate((pil_image1, pil_image2,pil_image3 ), axis=2)
            ret =np.concatenate((pil_image1, pil_image2,pil_image3,pil_image4,pil_image5 ), axis=2)
            #print('after combiing shape',ret.shape)
            ret = ret.astype('float32')
        
        if len(img.shape) > 2 and img.shape[2] == 1:
            ret = np.expand_dims(ret, -1)
        return ret

    def apply_coords(self, coords):
        # Transform image center from source coordinates into output coordinates
        # and then map the new origin to the corner of the output image.
        h, w = self.output_size
        x0, y0, x1, y1 = self.src_rect
        new_coords = coords.astype(np.float32)
        new_coords[:, 0] -= 0.5 * (x0 + x1)
        new_coords[:, 1] -= 0.5 * (y0 + y1)
        new_coords[:, 0] *= w / (x1 - x0)
        new_coords[:, 1] *= h / (y1 - y0)
        new_coords[:, 0] += 0.5 * w
        new_coords[:, 1] += 0.5 * h
        return new_coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation


class custom_RandomExtent(Augmentation):
    """
    Outputs an image by cropping a random "subrect" of the source image.
    The subrect can be parameterized to include pixels outside the source image,
    in which case they will be set to zeros (i.e. black). The size of the output
    image will vary with the size of the random subrect.
    """

    def __init__(self, scale_range, shift_range):
        """
        Args:
            output_size (h, w): Dimensions of output image
            scale_range (l, h): Range of input-to-output size scaling factor
            shift_range (x, y): Range of shifts of the cropped subrect. The rect
                is shifted by [w / 2 * Uniform(-x, x), h / 2 * Uniform(-y, y)],
                where (w, h) is the (width, height) of the input image. Set each
                component to zero to crop at the image's center.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        img_h, img_w = image.shape[:2]

        # Initialize src_rect to fit the input image.
        src_rect = np.array([-0.5 * img_w, -0.5 * img_h, 0.5 * img_w, 0.5 * img_h])

        # Apply a random scaling to the src_rect.
        src_rect *= np.random.uniform(self.scale_range[0], self.scale_range[1])

        # Apply a random shift to the coordinates origin.
        src_rect[0::2] += self.shift_range[0] * img_w * (np.random.rand() - 0.5)
        src_rect[1::2] += self.shift_range[1] * img_h * (np.random.rand() - 0.5)

        # Map src_rect coordinates into image coordinates (center at corner).
        src_rect[0::2] += 0.5 * img_w
        src_rect[1::2] += 0.5 * img_h

        return custom_ExtentTransform(
            src_rect=(src_rect[0], src_rect[1], src_rect[2], src_rect[3]),
            output_size=(int(src_rect[3] - src_rect[1]), int(src_rect[2] - src_rect[0])),
        )
