import os
import json
import sys
import logging
import traceback
import numpy as np
import cv2
import supervision as sv
from imutils import paths
from constants import *
from detr import Detr
from coco_detection import CocoDetection
from utils import LoggerWriter, prepare_for_coco_detection
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor
from pytorch_lightning import Trainer
from coco_eval import CocoEvaluator
from tqdm.notebook import tqdm


class CustomDETR:
    def __init__(self) -> None:
        self.read_config()

        self.test_img_paths = list(paths.list_images(DATASET_LOCATION + "test"))

        self.box_annotator = sv.BoxAnnotator()

    def read_config(self):
        with open(CONFIG_FILE, "r") as fp:
            self.config = json.load(fp)

        self.confidence_score = self.config["confidence_threshold"]
        self.iou_score = self.config["iou_threshold"]

        self.batch_size = self.config["batch_size"]
        self.learning_rate = self.config["learning_rate"]
        self.lr_backbone = self.config["lr_backbone"]
        self.weight_decay = self.config["weight_decay"]
        self.epochs = self.config["epochs"]

        self.retrain = self.config["retrain"]

        # retrain means loading weights from previous training on same dataset(i.e. Re-Finetuning)
        # else it will take open source weights to train the model
        if self.retrain:
            CHECKPOINT = PREVIOUS_TRAIN_PATH

    def get_model(self):
        self.image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)

        self.get_loader()

        self.model = Detr(
            self.learning_rate,
            self.lr_backbone,
            self.weight_decay,
            self.id2label,
            self.train_dataloader,
            self.val_dataloader,
        )

    def collate_fn(self, batch):
        # DETR authors employ various image sizes during training, making it not possible
        # to directly batch together images. Hence they pad the images to the biggest
        # resolution in a given batch, and create a corresponding binary pixel_mask
        # which indicates which pixels are real/which are padding
        pixel_values = [item[0] for item in batch]
        encoding = self.image_processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        return {
            "pixel_values": encoding["pixel_values"],
            "pixel_mask": encoding["pixel_mask"],
            "labels": labels,
        }

    def get_loader(self):
        self.train_dataset = CocoDetection(
            image_directory_path=TRAIN_DIRECTORY,
            image_processor=self.image_processor,
            train=True,
        )
        self.val_dataset = CocoDetection(
            image_directory_path=VAL_DIRECTORY,
            image_processor=self.image_processor,
            train=False,
        )
        self.test_dataset = CocoDetection(
            image_directory_path=TEST_DIRECTORY,
            image_processor=self.image_processor,
            train=False,
        )

        logging.info(f"Number of training examples: {len(self.train_dataset)}")
        logging.info(f"Number of validation examples: {len(self.val_dataset)}")
        logging.info(f"Number of test examples: {len(self.test_dataset)}")

        self.categories = self.train_dataset.coco.cats
        self.id2label = {k: v["name"] for k, v in self.categories.items()}

        logging.info(f"Categories : {self.categories}")
        logging.info(self.id2label)

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
        )
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
        )

    def train(self):
        trainer = Trainer(
            devices=1,
            accelerator="gpu",
            max_epochs=self.epochs,
            gradient_clip_val=0.1,
            accumulate_grad_batches=8,
            log_every_n_steps=5,
        )

        trainer.fit(self.model)

    def evaluate(self):
        evaluator = CocoEvaluator(coco_gt=self.test_dataset.coco, iou_types=["bbox"])

        for idx, batch in enumerate(
            tqdm(self.test_dataloader, desc="Running evaluation...")
        ):
            pixel_values = batch["pixel_values"].to(DEVICE)
            pixel_mask = batch["pixel_mask"].to(DEVICE)
            labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]

            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            orig_target_sizes = torch.stack(
                [target["orig_size"] for target in labels], dim=0
            )
            results = self.image_processor.post_process_object_detection(
                outputs, target_sizes=orig_target_sizes
            )

            predictions = {
                target["image_id"].item(): output
                for target, output in zip(labels, results)
            }
            predictions = prepare_for_coco_detection(predictions)
            evaluator.update(predictions)

        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()

    def prediction(self):
        for img_path in tqdm(self.test_img_paths, desc="Running Predictions..."):
            image = cv2.imread(img_path)

            # inference
            with torch.no_grad():
                # load image and predict
                inputs = self.image_processor(images=image, return_tensors="pt").to(
                    DEVICE
                )
                outputs = self.model(**inputs)

                # post-process
                target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
                results = self.image_processor.post_process_object_detection(
                    outputs=outputs,
                    threshold=self.confidence_score,
                    target_sizes=target_sizes,
                )[0]

            if len(results["boxes"].cpu()) != 0:
                detections = sv.Detections.from_transformers(
                    transformers_results=results
                ).with_nms(threshold=self.iou_score)

                labels = [
                    f"{self.id2label[class_id]} {confidence:.2f}"
                    for _, confidence, class_id, _ in detections
                ]

                frame = self.box_annotator.annotate(
                    scene=image.copy(), detections=detections, labels=labels
                )

                cv2.imwrite(
                    os.path.join(PREDICTION_DIR, os.path.basename(img_path)), frame
                )
            else:
                cv2.imwrite(
                    os.path.join(PREDICTION_DIR, os.path.basename(img_path)), image
                )

    def save_model(self):
        logging.info("Saving Weights...")
        self.model.model.save_pretrained(MODEL_PATH)


if __name__ == "__main__":
    try:
        # Remove all handlers associated with the root logger object.
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.INFO,
            filename=os.path.join(LOG_FILE),
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode="w",
        )

        sys.stdout = LoggerWriter(logging.info)
        sys.stderr = LoggerWriter(logging.error)

        detr = CustomDETR()

        detr.get_model()

        detr.train()

        detr.evaluate()

        detr.save_model()

    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())
