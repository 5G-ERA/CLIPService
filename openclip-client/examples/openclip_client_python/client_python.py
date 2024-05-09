from __future__ import annotations

import logging
import signal
import sys
import time
import traceback
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional
import random

import cv2

# CLIP and 5G-ERA stuff.
from era_5g_client.exceptions import FailedToConnect
from era_5g_interface.exceptions import BackPressureException
from era_5g_interface.utils.rate_timer import RateTimer
from openclip_client.client_common import CLIPClient, StreamType

# Set logging.
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CLIP client python")

# Testing video file.
# TEST_VIDEO_FILE = str("../../../videos/video3.mp4")
TEST_VIDEO_FILE = str("../../../videos/bringauto_2023-03-20_short.mp4")

# Testing configuration of the algorithm.
CONFIG_FILE = Path("../../../config/config.yaml")

# stopped flag for SIGTERM handler (stopping video frames sending).
stopped = False
clip_client: Optional[CLIPClient] = None


def signal_handler(sig: int, *_) -> None:
    """Signal handler for SIGTERM and SIGINT."""

    logger.info(f"Terminating ({signal.Signals(sig).name}) ...")
    global stopped
    if stopped:
        if clip_client is not None:
            clip_client.stop()
        sys.exit(1)
    stopped = True


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def main() -> None:
    """Example of CLIP client."""

    # Parse arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "-s", "--stream_type", type=int, help="StreamType: 1 = JPEG, 2 = H.264, 3 = HEVC", default=StreamType.H264
    )
    parser.add_argument("-c", "--config", type=Path, help="Collision warning config", default=CONFIG_FILE)
    parser.add_argument(
        "-p", "--out_prefix", type=str, help="Prefix of output csv file with measurements", default=None
    )
    parser.add_argument("-t", "--play_time", type=int, help="Video play time in seconds", default=5000)
    parser.add_argument("source_video", type=str, help="Video stream (file or url)", nargs="?", default=TEST_VIDEO_FILE)
    parser.add_argument("-m", "--measuring", type=bool, help="Enable extended measuring logs", default=True)
    parser.add_argument("--stats", type=bool, help="Store output data sizes", default=True)
    args = parser.parse_args()

    global clip_client

    try:
        # Create a video capture to pass images to the 5G-ERA Network Application.
        logger.info(f"Opening video capture {args.source_video}")
        cap = cv2.VideoCapture(args.source_video)
        if not cap.isOpened():
            raise Exception("Cannot open video capture")

        # Check bad loaded FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 60:
            logger.warning(f"FPS {fps} is strangely high, newly set to 30")
            fps = 30

        # Get width, height and number of frames.
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Number of frames can be wrong.
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        logger.info(f"Video {width}x{height}, {fps} FPS, " f"{frames} frames, " f"{frames / fps} seconds")
        logger.info(f"Stream type: {StreamType(args.stream_type)}")

        # Create collision warning client with given parameters,
        # width and height for client registration is loaded from camera config.
        clip_client = CLIPClient(
            config=args.config,
            stream_type=StreamType(args.stream_type),
            stats=args.stats,
            extended_measuring=args.measuring,
        )

        # Testing texts.
        texts = [
            ["big car", "small car", "bicycle", "person"],
            ["car on the right", "car on the left", "car in front"],
            ["one person", "no person", "more than 5 people"],
            ["colored cars", "black and white cars"],
            ["the sun shines from the right side", "the sun shines from the left side"],
            ["stone road", "asphalt road", "concrete road", "dirt road"],
            ["I drive in the city", "I drive in the forest", "I drive outside the city"],
            [
                "there is a traffic light in front of me",
                "there is no traffic light in front of me",
                "there is a traffic sign in front of me",
                "no sign or traffic light",
            ],
            ["the traffic light is red", "the traffic light is green", "the traffic light is orange"],
            ["it's cloudy, it's clear, it's raining"],
        ]

        # Rate timer for control the speed of a loop (fps).
        rate_timer = RateTimer(rate=fps, time_function=time.perf_counter, iteration_miss_warning=True)
        # Start time
        start_time = time.perf_counter_ns()
        # Check elapsed time or stopped flag.
        # Play time in ns
        play_time_ns = args.play_time * 1.0e9
        # Main processing loop
        logger.info("Start sending images ...")
        while time.perf_counter_ns() - start_time < play_time_ns and not stopped:
            # Read single frame from a stream.
            ret, frame = cap.read()
            if not ret:
                break
            # Send to CLIP service for processing.
            try:
                metadata = {"model_name": "ViT-B-32", "texts": random.choice(texts)}
                clip_client.send_image(frame, metadata=metadata)
            except BackPressureException as e:
                logger.warning(f"BackPressureException raised while sending: {e}")
            # Sleep until next frame should be sent (with given fps).
            rate_timer.sleep()
        end_time = time.perf_counter_ns()
        logger.info(f"Total streaming time: {(end_time - start_time) * 1.0e-9:.3f}s")
        cap.release()

    except FailedToConnect as ex:
        logger.error(f"Failed to connect to server: {ex}")
    except KeyboardInterrupt:
        logger.info("Terminating ...")
    except Exception as ex:
        traceback.print_exc()
        logger.error(f"Exception: {repr(ex)}")
    finally:
        if clip_client is not None:
            clip_client.stop()


if __name__ == "__main__":
    main()
