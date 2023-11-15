import torch
import os
import logging
from sahi.slicing import slice_coco
from constants import DATASET_LOCATION


class LoggerWriter:
    def __init__(self, logfct):
        self.logfct = logfct
        self.buf = []

    def write(self, msg):
        if msg.endswith("\n"):
            self.buf.append(msg.rstrip("\n"))
            self.logfct("".join(self.buf))
            self.buf = []
        else:
            self.buf.append(msg)

    def flush(self):
        pass


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results


def slice_dataset(sahi_params, _type="train"):
    logging.info(f"Salicing {_type} dataset...")

    # renaming data directory
    temp_dir = os.path.join(DATASET_LOCATION, f"{_type}_orig_dataset")
    os.rename(os.path.join(DATASET_LOCATION, _type), temp_dir)

    # Sliced dataset directory name
    output_dir = os.path.join(DATASET_LOCATION, _type)

    # set verbose = True for logs
    coco_dict, coco_path = slice_coco(
        coco_annotation_file_path=temp_dir + "/_annotations.coco.json",
        image_dir=temp_dir,
        output_coco_annotation_file_name="sliced",
        ignore_negative_samples=False,
        output_dir=output_dir,
        slice_height=sahi_params["slice_height"],
        slice_width=sahi_params["slice_width"],
        overlap_height_ratio=sahi_params["overlap_height_ratio"],
        overlap_width_ratio=sahi_params["overlap_width_ratio"],
        min_area_ratio=0.1,
        verbose=False,
    )

    # rename the annotation file back again
    os.rename(
        os.path.join(output_dir, "sliced_coco.json"),
        os.path.join(output_dir, "_annotations.coco.json"),
    )
