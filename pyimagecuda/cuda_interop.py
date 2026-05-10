from .image import ImageBase, ImageU8


def to_cupy(image: ImageBase):
    """
    Wraps a pyimagecuda Image or ImageU8 as a zero-copy CuPy ndarray.

    The returned ndarray shares GPU memory with the source image. The image
    must remain alive while the array is in use; CuPy holds a reference to
    the image via UnownedMemory's owner argument.

    Docs & Examples: https://offerrall.github.io/pyimagecuda/image/
    """
    try:
        import cupy
    except ImportError as e:
        raise ImportError(
            "CuPy is required for to_cupy(). "
            "Install it with: pip install cupy-cuda12x."
        ) from e

    is_u8 = isinstance(image, ImageU8)
    bytes_per_pixel = 4 if is_u8 else 16
    size_bytes = image.width * image.height * bytes_per_pixel

    unowned_mem = cupy.cuda.UnownedMemory(image.cuda_ptr, size_bytes, image)
    memptr = cupy.cuda.MemoryPointer(unowned_mem, 0)

    dtype = cupy.uint8 if is_u8 else cupy.float32
    return cupy.ndarray((image.height, image.width, 4), dtype=dtype, memptr=memptr)
