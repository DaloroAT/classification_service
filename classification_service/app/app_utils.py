import struct
import uuid
import numpy as np
from werkzeug.datastructures import FileStorage
import cv2


def filestorage2ndarray(file: FileStorage) -> np.ndarray:
    filestr = file.read()
    img_np = np.fromstring(filestr, np.uint8)
    img_np = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
    return img_np


def ndarray2bytes(array_np: np.ndarray) -> bytes:
    assert isinstance(array_np, np.ndarray)
    assert array_np.dtype == np.uint8

    h, w, *c = array_np.shape

    if len(c) == 1:
        c = c[0]
    elif len(c) == 0:
        c = 1
    else:
        raise ValueError("Not supported shape")

    shape = struct.pack(">III", h, w, c)
    array_bytes = shape + array_np.tobytes()
    return array_bytes


def bytes2ndarray(array_bytes: bytes) -> np.ndarray:
    assert isinstance(array_bytes, bytes)

    h, w, c = struct.unpack(">III", array_bytes[:12])
    array_np = np.frombuffer(array_bytes, dtype=np.uint8, offset=12).reshape(h, w, c)
    return array_np


def create_uuid():
    return str(uuid.uuid4())

