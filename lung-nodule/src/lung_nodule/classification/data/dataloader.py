"""Dataset and data loading utilities for lung nodule classification."""

from typing import List
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import pandas as pd

from lung_nodule.classification.data.transforms import (
    volumeTransform,
    rotateMatrixX,
    rotateMatrixY,
    rotateMatrixZ,
    sample_random_coordinate_on_sphere,
)


def worker_init_fn(worker_id):
    """
    A worker initialization method for seeding the numpy random
    state using different random seeds for all epochs and workers
    """
    seed = int(torch.utils.data.get_worker_info().seed) % (2**32)
    np.random.seed(seed=seed)


def clip_and_scale(npzarray, maxHU=400.0, minHU=-1000.0):
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.0
    npzarray[npzarray < 0] = 0.0
    return npzarray


def extract_patch(
    CTData,
    coord,
    srcVoxelOrigin,
    srcWorldMatrix,
    srcVoxelSpacing,
    output_shape=(64, 64, 64),
    voxel_spacing=(50.0 / 64, 50.0 / 64, 50.0 / 64),
    rotations=None,
    translations=None,
    coord_space_world=False,
    mode="2D",
):

    transform_matrix = np.eye(3)

    if rotations is not None:
        (zmin, zmax), (ymin, ymax), (xmin, xmax) = rotations

        # add random rotation
        angleX = np.multiply(np.pi / 180.0, np.random.randint(xmin, xmax, 1))[0]
        angleY = np.multiply(np.pi / 180.0, np.random.randint(ymin, ymax, 1))[0]
        angleZ = np.multiply(np.pi / 180.0, np.random.randint(zmin, zmax, 1))[0]

        transformMatrixAug = np.eye(3)
        transformMatrixAug = np.dot(
            transformMatrixAug, rotateMatrixX(np.cos(angleX), np.sin(angleX))
        )
        transformMatrixAug = np.dot(
            transformMatrixAug, rotateMatrixY(np.cos(angleY), np.sin(angleY))
        )
        transformMatrixAug = np.dot(
            transformMatrixAug, rotateMatrixZ(np.cos(angleZ), np.sin(angleZ))
        )

        transform_matrix = np.dot(transform_matrix, transformMatrixAug)

    if translations is not None:
        # add random translation
        radius = np.random.random_sample() * translations
        offset = sample_random_coordinate_on_sphere(radius=radius)
        offset = offset * (1.0 / srcVoxelSpacing)

        coord = np.array(coord) + offset

    # Normalize transform matrix
    thisTransformMatrix = transform_matrix

    thisTransformMatrix = (
        thisTransformMatrix.T
        / np.sqrt(np.sum(thisTransformMatrix * thisTransformMatrix, axis=1))
    ).T

    invSrcMatrix = np.linalg.inv(srcWorldMatrix)

    # world coord sampling
    if coord_space_world:
        overrideCoord = invSrcMatrix.dot(coord - srcVoxelOrigin)
    else:
        # image coord sampling
        overrideCoord = coord * srcVoxelSpacing
    overrideMatrix = (invSrcMatrix.dot(thisTransformMatrix.T) * srcVoxelSpacing).T

    patch = volumeTransform(
        CTData,
        srcVoxelSpacing,
        overrideMatrix,
        center=overrideCoord,
        output_shape=np.array(output_shape),
        output_voxel_spacing=np.array(voxel_spacing),
        order=1,
        prefilter=False,
    )

    if mode == "2D":
        # replicate the channel dimension
        patch = np.repeat(patch, 3, axis=0)

    else:
        patch = np.expand_dims(patch, axis=0)

    return patch


class CTCaseDataset(data.Dataset):
    """LUNA25 baseline dataset
            Args:
            data_dir (str): path to the nodule_blocks data directory
            dataset (pd.DataFrame): dataframe with the dataset information
            translations (bool): whether to apply random translations
            rotations (tuple): tuple with the rotation ranges
            size_px (int): size of the patch in pixels
            size_mm (int): size of the patch in mm
            mode (str): 2D or 3D
            patch_size (list): patch size for extraction coordinates

    """

    def __init__(
        self,
        data_dir: str,
        dataset: pd.DataFrame,
        translations: bool = None,
        rotations: tuple = None,
        size_px: int = 64,
        size_mm: int = 50,
        mode: str = "2D",
        patch_size: List[int] = None,
    ):

        self.data_dir = Path(data_dir)
        self.dataset = dataset
        self.patch_size = patch_size if patch_size is not None else [64, 128, 128]
        self.rotations = rotations
        self.translations = translations
        self.size_px = size_px
        self.size_mm = size_mm
        self.mode = mode


    def __getitem__(self, idx):  # caseid, z, y, x, label, radius

        pd_row = self.dataset.iloc[idx]

        label = pd_row.label

        annotation_id = pd_row.AnnotationID

        image_path = self.data_dir / "image" / f"{annotation_id}.npy"
        metadata_path = self.data_dir / "metadata" / f"{annotation_id}.npy"

        # numpy memory map data/image case file
        img = np.load(image_path, mmap_mode="r")
        metadata = np.load(metadata_path, allow_pickle=True).item()

        origin = metadata["origin"]
        spacing = metadata["spacing"]
        transform = metadata["transform"]

        translations = None
        if self.translations == True:
            radius = 2.5
            translations = radius if radius > 0 else None


        if self.mode == "2D":
            output_shape = (1, self.size_px, self.size_px)
        else:
            output_shape = (self.size_px, self.size_px, self.size_px)

        patch = extract_patch(
            CTData=img,
            coord=tuple(np.array(self.patch_size) // 2),
            srcVoxelOrigin=origin,
            srcWorldMatrix=transform,
            srcVoxelSpacing=spacing,
            output_shape=output_shape,
            voxel_spacing=(
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
            ),
            rotations=self.rotations,
            translations=translations,
            coord_space_world=False,
            mode=self.mode,
        )

        # ensure same datatype...
        patch = patch.astype(np.float32)

        # clip and scale...
        patch = clip_and_scale(patch)

        target = torch.ones((1,)) * label

        sample = {
            "image": torch.from_numpy(patch),
            "label": target.long(),
            "ID": annotation_id,
        }

        return sample

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        return fmt_str


def get_data_loader(
    data_dir,
    dataset,
    mode="2D",
    sampler=None,
    workers=0,
    batch_size=64,
    size_px=64,
    size_mm=70,
    rotations=None,
    translations=None,
    patch_size=None,
):

    data_set = CTCaseDataset(
        data_dir=data_dir,
        translations=translations,
        dataset=dataset,
        rotations=rotations,
        size_mm=size_mm,
        size_px=size_px,
        mode=mode,
        patch_size=patch_size,
    )

    shuffle = False
    if sampler == None:
        shuffle = (True,)

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        sampler=sampler,
        worker_init_fn=worker_init_fn,
    )

    return data_loader
