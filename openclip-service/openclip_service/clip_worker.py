import logging
import sys
import time
from multiprocessing import Queue
from pathlib import Path
from queue import Empty
from threading import Event, Thread
from typing import Any, Callable, Dict
import random

import cv2
import numpy as np
import open_clip
import torch
import yaml
from PIL import Image
from torch import Tensor

from era_5g_interface.interface_helpers import LatencyMeasurements

logger = logging.getLogger(__name__)


class CLIPWorker(Thread):
    """CLIP worker. Reads data from passed queue, performs CLIP processing and returns results using callback.

    TODO: Make variant with images as input features
    """

    def __init__(
        self,
        image_queue: Queue,
        send_function: Callable[[Dict[str, Any]], None],
        config: Dict,
        send_error_function: Callable[[Dict[str, Any]], None] = None,
        frame_in_results: bool = False,
        **kw,
    ) -> None:
        """Constructor.

        Args:
            image_queue (Queue): The queue with all to-be-processed images.
            send_function (Callable[[Dict], None]): Callback used to send results.
            config (Dict): CLIP config.
            send_error_function (Callable[[Dict], None]): Callback used to send errors.
            frame_in_results (bool): Whether to add frame into results.
            **kw: Thread arguments.
        """

        super().__init__(**kw)

        self._stop_event = Event()
        self.image_queue = image_queue
        self._send_function = send_function
        self._send_error_function = send_error_function
        self._frame_id = 0
        self.latency_measurements: LatencyMeasurements = LatencyMeasurements()
        self._config = config
        self._frame_in_results = frame_in_results

        # Use CUDA if it is possible.
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Torch device: {self._device}")
        self._text_features = {}
        # Load models specified in config.
        models = config.get("models", {})
        for model_name in models:
            pretrained = None
            if "pretrained" in models[model_name]:
                pretrained = models[model_name]["pretrained"]
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self._device
            )
            self._text_features[model_name] = {"model": model, "preprocess": preprocess, "text_features": {}}

        # Compute initial texts features specified in config.
        init_texts = config.get("init_texts", {})
        logger.info(f"Compute initial texts features ...")
        for model_name in init_texts:
            logger.info(f"Model name {model_name}, initial texts: {init_texts[model_name]}")
            for text in init_texts[model_name]:
                self.compute_text_features(text=text, model_name=model_name)

    def compute_text_features(self, text: str, model_name: str = "ViT-B-32") -> Tensor:
        """Compute text features (tokens) (if they don't already exist) and return them.

        Args:
            text (str): Text to tokenize.
            model_name (str): OpenCLIP model name.

        Returns: Text features for given text.

        """

        # Load model on-the-fly if it is not loaded.
        if model_name not in self._text_features:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, device=self._device
            )
            self._text_features[model_name] = {"model": model, "preprocess": preprocess, "text_features": {}}
        text_features = self._text_features[model_name]
        if text not in text_features["text_features"]:
            tokenizer = open_clip.get_tokenizer(model_name)
            text_tokenized = tokenizer(text).to(self._device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                text_features["text_features"][text] = text_features["model"].encode_text(text_tokenized)
                text_features["text_features"][text] /= text_features["text_features"][text].norm(dim=-1, keepdim=True)
        return text_features["text_features"][text]

    def stop(self) -> None:
        """Set stop event to stop CLIP worker."""

        self._stop_event.set()

    def __del__(self) -> None:
        logger.info("Delete models")
        del self._text_features

    def _process_image(self, metadata, frame: np.array):
        """Process image with metadata and send results.

        Args:
            metadata (Dict): Model name and texts as image metadata - e.g. {"metadata": {"request": {"model_name":
                "ViT-B-32", "texts": ["car", "road"]}}}:
            frame (np.ndarray): Image to be processed.
        """

        # Store timestamp before processing.
        metadata["timestamp_before_process"] = time.perf_counter_ns()

        request_metadata = metadata.get("metadata", {})
        logger.info(f"Request: {metadata}")

        self._frame_id += 1
        # logger.info(f"Worker received frame id: {self.frame_id} {metadata['timestamp']}")

        try:
            # Compute or use cached text features.
            text_features_list = []
            for text in request_metadata["texts"]:
                text_features_list.append(
                    self.compute_text_features(text=text, model_name=request_metadata["model_name"])
                )

            # Concatenate requested text_features.
            text_features = torch.cat(text_features_list)
            preprocess = self._text_features[request_metadata["model_name"]]["preprocess"]
            model = self._text_features[request_metadata["model_name"]]["model"]

            # Evaluation of texts with the image.
            image = preprocess(Image.fromarray(frame)).unsqueeze(0).to(self._device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy().tolist()[0]
            logger.info(f"Results: {probs}")

            # Store timestamp after processing.
            metadata["timestamp_after_process"] = time.perf_counter_ns()

            # Generate results.
            results = {
                "timestamp": metadata.get("timestamp", 0),
                "recv_timestamp": metadata.get("recv_timestamp", 0),
                "timestamp_before_process": metadata["timestamp_before_process"],
                "timestamp_after_process": metadata["timestamp_after_process"],
                "send_timestamp": time.perf_counter_ns(),
                "texts": request_metadata["texts"],
                "probs": probs,
            }
            if self._frame_in_results:
                results["frame"] = frame

            # Send results via the provided callback.
            self._send_function(results)

            # Store latency (local).
            self.latency_measurements.store_latency(time.perf_counter_ns() - metadata.get("recv_timestamp", 0))

        except Exception as ex:
            logger.error(f"Exception with image processing ({type(ex)}): {repr(ex)}")
            if self._send_error_function:
                self._send_error_function({"message": f"Exception with image processing ({type(ex)}): {repr(ex)}"})
            raise ex

    def run(self) -> None:
        """CLIP worker loop. Periodically reads images from python internal queue process them."""

        logger.info(f"{self.name} thread is running.")

        while not self._stop_event.is_set():
            # Get image and metadata from input queue.
            metadata: Dict[str, Any]
            image: np.ndarray
            try:
                metadata, image = self.image_queue.get(block=True, timeout=1)
            except Empty:
                continue

            self._process_image(metadata, image)

        logger.info(f"{self.name} thread is stopping.")


def visualization(results: [Dict[str, Any]]) -> None:
    """Testing send function - visualization."""

    logger.info(f"{results['texts']}: {results['probs']}")
    logger.info(
        f"delay: "
        f"{(results.get('timestamp_after_process', 0) - results.get('timestamp_before_process', 0)) * 1.0e-9:.3f}s"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        results["frame"],
        f"{results['texts']}: {results['probs']}",
        (20, 20),
        font,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    try:
        cv2.imshow("Results", results["frame"])
        cv2.waitKey(1)
    except Exception as ex:
        logger.error(repr(ex))


def main() -> None:
    """CLIP worker testing."""

    logging.basicConfig(
        stream=sys.stdout, level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Testing configuration of the algorithm.
    config = Path("../../config/config.yaml")
    config_dict = yaml.safe_load(config.open())
    test_video_file = str("../../videos/video3.mp4")

    clip_worker = CLIPWorker(
        Queue(1), send_function=visualization, config=config_dict, frame_in_results=True, daemon=True
    )

    cap = cv2.VideoCapture(test_video_file)
    if not cap.isOpened():
        raise Exception("Cannot open video file")

    texts = [
        ["big car", "small car", "bicycle", "person"],
        ["car on the right", "car on the left", "car in front"],
        ["one person", "no person", "more than 5 people"],
        ["colored cars", "black and white cars"],
        ["the sun shines from the right side", "the sun shines from the left side"],
        ["stone road", "asphalt road", "concrete road", "dirt road"],
        ["I drive in the city", "I drive in the forest", "I drive outside the city"],
        ["there is a traffic light in front of me",
         "there is no traffic light in front of me",
         "there is a traffic sign in front of me",
         "no sign or traffic light"],
        ["the traffic light is red", "the traffic light is green", "the traffic light is orange"],
        ["it's cloudy, it's clear, it's raining"]
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        metadata = {
            "metadata": {"model_name": "ViT-B-32", "texts": random.choice(texts)}}
        #metadata = {"metadata": {"model_name": "ViT-B-32", "texts": ["car", "road"]}}
        clip_worker._process_image(metadata, frame)
        # time.sleep(1 / 30)


if __name__ == "__main__":
    main()
