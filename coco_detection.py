import torchvision
from torchvision.transforms import v2
from constants import *


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, image_directory_path: str, image_processor, train: bool = True):
        # Augmentation, Change as per need
        # self.transforms = v2.Compose(
        #     [
        #         v2.ToImage(),
        #         v2.RandomPhotometricDistort(p=1),
        #         v2.RandomIoUCrop(),
        #         v2.RandomHorizontalFlip(p=1),
        #         v2.SanitizeBoundingBoxes(),
        #         v2.ToDtype(torch.float32, scale=True),
        #     ]
        # )

        annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)

        # if train:
        # super(CocoDetection, self).__init__(image_directory_path, annotation_file_path, transforms=self.transforms)

        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {"image_id": image_id, "annotations": annotations}
        encoding = self.image_processor(
            images=images, annotations=annotations, return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target
