from typing import List, Union, Dict
from pathlib import Path
import time

from torchvision.models import resnet18
import torch
from torch import Tensor
from torch import nn
import redis
import numpy as np
import cv2

from classification_service.utils import TaskStructure
from classification_service.network.transforms import EvalTransforms
from classification_service.config import config, redis_config


Predictions = List[str]


class Classifier(nn.Module):

    def __init__(self, imagenet_classes_path: Union[Path, str] = None):
        super().__init__()

        self.classes_num2word = self._get_classes_num2word(imagenet_classes_path)

        self.network = resnet18(pretrained=True)
        self.network.to(config.device)

        self.eval_transform = EvalTransforms()

    @staticmethod
    def _get_classes_num2word(imagenet_classes_path: Union[Path, str] = None) -> Dict[int, str]:
        imagenet_num_classes = 1000
        if imagenet_classes_path is not None:
            imagenet_classes_path = Path(imagenet_classes_path)
            assert imagenet_classes_path.exists()

            with open(imagenet_classes_path, 'r') as file_in:
                classes_words = file_in.read().splitlines()
            assert len(classes_words) == imagenet_num_classes
        else:
            classes_words = [str(k) for k in range(imagenet_num_classes)]

        classes_num2word = dict(zip(range(imagenet_num_classes), classes_words))
        return classes_num2word

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

    def classify_by_path(self, path_img: Union[Path, str]) -> Predictions:
        path_img = Path(path_img)
        assert path_img.exists()
        assert path_img.is_file()
        ext = [".png", ".jpg", ".jpeg"]
        if path_img.suffix not in ext:
            raise ValueError("Not supported format")
        else:
            image = cv2.imread(str(path_img))
            image_tensor = self.eval_transform(image).unsqueeze(0)
            return self.classify_tensor_batch(image_tensor)

    def classify_img_ndarray(self, image: np.ndarray) -> Predictions:
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        assert image.ndim == 3
        assert image.shape[2] == 3

        image_tensor = self.eval_transform(image).unsqueeze(0)
        return self.classify_tensor_batch(image_tensor)

    @torch.no_grad()
    def classify_tensor_batch(self, batch_images: Tensor) -> Predictions:
        assert batch_images.ndim == 4
        assert batch_images.size()[1] == 3

        self.network.eval()

        outputs = self(batch_images.to(config.device))
        outputs = torch.softmax(outputs, dim=1)
        probs, classes = torch.max(outputs, dim=1)

        probs = probs.tolist()
        classes = classes.tolist()
        predictions = [f"probability {round(p, 3)} - {self.classes_num2word[c]}" for p, c in zip(probs, classes)]
        return predictions

    def loop_based_redis(self, db_redis: redis.Redis) -> None:
        status_on_classifying = "image on classifying"
        status_completed = "classifying completed"

        while True:

            unique_ids_from_queue_head: List[bytes] = []
            # Get list of unique_ids for classification.
            # We use 'lpop' instead combination of 'lrange'+'ltrim' due to several instances of classifier can process
            # same batch of tasks.
            # Also 'lrange'+'ltrim' consumes more time than 'lpop' even with loop.
            for _ in range(config.BATCH_SIZE):
                unique_id_for_classification = db_redis.lpop(redis_config.REDIS_QUEUE_IMAGES)
                if unique_id_for_classification is not None:
                    unique_ids_from_queue_head.append(unique_id_for_classification)

            images_list: List[Tensor] = []
            data_task_list: List[TaskStructure] = []

            for unique_id in unique_ids_from_queue_head:
                data_task_redis = db_redis.hgetall(unique_id)

                data_task = TaskStructure()
                loaded_id = data_task.load_redis_hgetall(data_task_redis)

                # handle situation if task in queue, but not in database in some reason
                if not loaded_id.is_empty():

                    image_np = data_task.get_img_ndarray()
                    image_tensor = self.eval_transform(image_np).unsqueeze(0)
                    images_list.append(image_tensor)

                    data_task_list.append(data_task)

                    db_redis.hset(unique_id, TaskStructure.get_keyname_status(), status_on_classifying)

            if len(images_list) > 0:
                images_tensor_batch = torch.cat(images_list)
                predictions_list = self.classify_tensor_batch(images_tensor_batch)

                for data_task, prediction in zip(data_task_list, predictions_list):
                    data_task.update_prediction(prediction)
                    data_task.update_status(status_completed)
                    if redis_config.FLUSH_IMG_AFTER_CLASSIFICATION:
                        data_task.flush_img()
                    db_redis.hmset(data_task.get_uuid().to_str(), data_task.to_redis_hmset())

            time.sleep(redis_config.SERVER_SLEEP)


if __name__ == "__main__":
    
    classifier = Classifier(config.root_dir / "classification_service" / "imagenet_classes.txt")

    database_redis = redis.Redis(host=redis_config.REDIS_HOST,
                                 port=redis_config.REDIS_PORT,
                                 db=redis_config.REDIS_DB)

    classifier.loop_based_redis(database_redis)
