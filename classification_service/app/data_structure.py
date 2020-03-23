from typing import Dict, List, Optional

from werkzeug.datastructures import FileStorage
import numpy as np

from classification_service.app.app_utils import create_uuid, ndarray2bytes, bytes2ndarray, filestorage2ndarray


class UUID:

    def __init__(self, unique_id: Optional[str] = None):
        super().__init__()

        if unique_id is None:
            self.uuid = create_uuid()
        else:
            self.uuid = None
            self.from_str(unique_id)

    def __call__(self) -> str:
        return str(self.uuid)

    def __str__(self) -> str:
        return self.__call__()

    def from_str(self, unique_id: str) -> None:
        assert isinstance(unique_id, str)
        self.uuid = unique_id

    def empty_uuid(self) -> None:
        self.uuid = ""

    def is_empty(self) -> bool:
        if self.uuid == "":
            return True
        else:
            return False

    def to_str(self) -> str:
        return self.__str__()


class TaskStructure:

    def __init__(self):
        self._structure = self._get_structure_template()

    def _get_structure_template(self) -> Dict:
        fieldsname = self._get_fieldsname()
        structure = dict(zip(fieldsname, [""] * len(fieldsname)))
        structure["img_bytes"] = b''
        return structure

    @staticmethod
    def _get_fieldsname() -> List[str]:
        fieldsname = ["uuid",
                      "img_bytes",
                      "status",
                      "prediction"]
        return fieldsname

    def _flush_structure(self) -> None:
        self._structure = self._get_structure_template()

    def flush_img(self) -> None:
        self._structure["img_bytes"] = b''

    def from_img_filestorage(self, file: FileStorage) -> UUID:
        assert isinstance(file, FileStorage)
        img_np = filestorage2ndarray(file)
        return self.from_img_ndarray(img_np)

    def from_img_ndarray(self, img_np: np.ndarray) -> UUID:
        assert isinstance(img_np, np.ndarray)
        img_bytes = ndarray2bytes(img_np)
        return self.from_img_bytes(img_bytes)

    def from_img_bytes(self, img_bytes: bytes) -> UUID:
        assert isinstance(img_bytes, bytes)
        self._flush_structure()
        self._structure["img_bytes"] = img_bytes
        unique_id = UUID()
        self._structure["uuid"] = unique_id()
        self._structure["status"] = "added to queue for processing"
        return unique_id

    def load_redis_hgetall(self, redis_hgetall: Dict) -> UUID:
        if redis_hgetall == {}:
            empty_uuuid = UUID()
            empty_uuuid.empty_uuid()
            return empty_uuuid
        else:
            self._flush_structure()
            keys = self._get_fieldsname()

            for key in keys:
                raw_value = redis_hgetall.get(key.encode("utf-8"))
                if raw_value is None:
                    value = None
                else:
                    value = raw_value.decode("utf-8") if key != "img_bytes" else raw_value

                self._structure[key] = value

            return UUID(self._structure["uuid"])

    def get_img_bytes(self) -> bytes:
        return self._structure["img_bytes"]

    def get_img_ndarray(self) -> np.ndarray:
        return bytes2ndarray(self._structure["img_bytes"])

    def get_status(self) -> str:
        return self._structure["status"]

    def update_status(self, status_str: str) -> None:
        assert isinstance(status_str, str)
        self._structure["status"] = status_str

    def get_prediction(self) -> int:
        return self._structure["prediction"]

    def update_prediction(self, prediction_str: str) -> None:
        assert isinstance(prediction_str, str)
        self._structure["prediction"] = prediction_str

    def get_uuid(self) -> UUID:
        return UUID(self._structure["uuid"])

    def update_img_np(self, img_np: np.ndarray) -> None:
        assert isinstance(img_np, np.ndarray)
        self._structure["img_bytes"] = ndarray2bytes(img_np)

    def get_structure(self, include_image: bool = True) -> Dict:
        if include_image:
            return self._structure
        else:
            structure = self._structure.copy()
            structure.pop("img_bytes")
            return structure

    def to_redis_hmset(self) -> Dict:
        return self._structure

    @staticmethod
    def get_keyname_status() -> str:
        return "status"

    def __str__(self):
        return str(self._structure)
