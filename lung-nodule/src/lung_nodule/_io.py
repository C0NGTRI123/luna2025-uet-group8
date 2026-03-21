"""Shared I/O utilities for SimpleITK image conversion."""

import numpy as np
import SimpleITK


def _transform(input_image, point):
    """Transform a point from index to physical space (reversed axis order)."""
    return np.array(
        list(
            reversed(
                input_image.TransformContinuousIndexToPhysicalPoint(
                    list(reversed(point))
                )
            )
        )
    )


def itk_image_to_numpy(input_image: SimpleITK.Image) -> tuple:
    """
    Convert a SimpleITK image to a numpy array with spatial metadata.

    Parameters
    ----------
    input_image : SimpleITK.Image
        The CT image to convert.

    Returns
    -------
    numpy_image : np.ndarray
        The image pixel data as a numpy array.
    header : dict
        Dictionary with keys 'origin', 'spacing', 'transform' as numpy arrays.
    """
    numpy_image = SimpleITK.GetArrayFromImage(input_image)
    numpy_origin = np.array(list(reversed(input_image.GetOrigin())))
    numpy_spacing = np.array(list(reversed(input_image.GetSpacing())))

    t_numpy_origin = _transform(input_image, np.zeros((numpy_image.ndim,)))
    t_numpy_matrix_components = [None] * numpy_image.ndim
    for i in range(numpy_image.ndim):
        v = [0] * numpy_image.ndim
        v[i] = 1
        t_numpy_matrix_components[i] = _transform(input_image, v) - t_numpy_origin
    numpy_transform = np.vstack(t_numpy_matrix_components).dot(np.diag(1 / numpy_spacing))

    header = {
        "origin": numpy_origin,
        "spacing": numpy_spacing,
        "transform": numpy_transform,
    }

    return numpy_image, header
