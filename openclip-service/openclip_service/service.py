import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from queue import Queue
from typing import Any, Dict, Tuple

import numpy as np

from era_5g_interface.channels import CallbackInfoServer, ChannelType, DATA_ERROR_EVENT, DATA_NAMESPACE
from era_5g_interface.dataclasses.control_command import ControlCmdType, ControlCommand
from era_5g_interface.interface_helpers import HeartbeatSender
from era_5g_interface.task_handler_internal_q import TaskHandlerInternalQ
from era_5g_server.server import NETAPP_STATUS_ADDRESS, NetworkApplicationServer, generate_application_heartbeat_data
from openclip_service.clip_worker import CLIPWorker

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CLIP service")

# Port of the 5G-ERA Network Application's server.
NETAPP_PORT = int(os.getenv("NETAPP_PORT", 5896))
# Input queue size.
NETAPP_INPUT_QUEUE = int(os.getenv("NETAPP_INPUT_QUEUE", 1))
# Event name for image error.
IMAGE_ERROR_EVENT = str("image_error")

EXTENDED_MEASURING = bool(os.getenv("EXTENDED_MEASURING", False))


@dataclass
class TaskAndWorker:
    """Class for task and worker."""

    task: TaskHandlerInternalQ
    worker: CLIPWorker


class Server(NetworkApplicationServer):
    """CLIP server receives images from clients, manages tasks and workers, sends results to clients."""

    def __init__(
        self,
        **kwargs,
    ) -> None:
        """Constructor.

        Args:
            *args: NetworkApplicationServer arguments.
            **kwargs: NetworkApplicationServer arguments.
        """

        super().__init__(
            callbacks_info={
                "image_h264": CallbackInfoServer(ChannelType.H264, self.image_callback),
                "image_hevc": CallbackInfoServer(ChannelType.HEVC, self.image_callback),
                "image_jpeg": CallbackInfoServer(ChannelType.JPEG, self.image_callback),
            },
            **kwargs,
        )

        # List of registered tasks.
        self.tasks: Dict[str, TaskAndWorker] = dict()

        # Create Heartbeat sender
        self.heartbeat_sender = HeartbeatSender(NETAPP_STATUS_ADDRESS, self.generate_heartbeat_data)

    def generate_heartbeat_data(self):
        """Application heartbeat data generation using queue info and latencies."""

        latencies = []
        queue_occupancy = 0
        queue_size = 0
        for task_and_worker in self.tasks.values():
            queue_occupancy += task_and_worker.task.data_queue_occupancy()
            queue_size += task_and_worker.task.data_queue_size()
            latencies.extend(task_and_worker.worker.latency_measurements.get_latencies())
        avg_latency = 0
        if len(latencies) > 0:
            avg_latency = float(np.mean(np.array(latencies)))

        return generate_application_heartbeat_data(avg_latency, queue_size, queue_occupancy, len(self.tasks))

    def image_callback(self, sid: str, data: Dict[str, Any]) -> None:
        """Allows to receive decoded image using the websocket transport.

        Args:
            sid (str): Namespace sid.
            data (Dict[str, Any]): Data dict including decoded frame (data["frame"]) and send timestamp
                (data["timestamp"]).
        """

        eio_sid = self.get_eio_sid_of_data(sid)

        if eio_sid not in self.tasks:
            logger.error(f"Non-registered client {eio_sid} tried to send data")
            self.send_data({"message": "Non-registered client tried to send data"}, DATA_ERROR_EVENT, sid=sid)
            self.disconnect(sid)
            return

        if not self.tasks[eio_sid].worker.is_alive():
            logger.error(f"Worker is not alive, eio_sid {eio_sid}, sid {sid}")
            self.send_data({"message": f"Worker is not alive, eio_sid {eio_sid}, sid {sid}"}, DATA_ERROR_EVENT, sid=sid)
            self.disconnect(sid)
            return

        task = self.tasks[eio_sid].task
        task.store_data(
            {"timestamp": data["timestamp"], "recv_timestamp": time.perf_counter_ns(), "metadata": data["metadata"]},
            data["frame"],
        )

    def disconnect(self, sid: str) -> None:
        """Disconnects the client from DATA_NAMESPACE by sid.

        Args:
            sid (str): Namespace sid.
        """

        self._sio.disconnect(sid, DATA_NAMESPACE)

    def command_callback(self, command: ControlCommand, sid: str) -> Tuple[bool, str]:
        """Process initialization control command - create task, worker and start the worker.

        Args:
            command (ControlCommand): Control command to be processed.
            sid (str): Namespace sid.

        Returns:
            (initialized (bool), message (str)): If False, initialization failed.
        """

        eio_sid = self.get_eio_sid_of_control(sid)

        logger.info(f"Control command {command} processing: session id: {sid}")

        if command and command.cmd_type == ControlCmdType.INIT:
            # Check that initialization has not been called before.
            if eio_sid in self.tasks:
                logger.error(f"Client attempted to call initialization multiple times")
                self.send_command_error("Initialization has already been called before", sid)
                return False, "Initialization has already been called before"

            args = command.data
            config = {}
            if args:
                config = args.get("config", config)

            # Queue with received images.
            image_queue = Queue(NETAPP_INPUT_QUEUE)

            task = TaskHandlerInternalQ(image_queue)

            try:
                # Create worker.
                worker = CLIPWorker(
                    image_queue=image_queue,
                    send_function=lambda results: self.send_data(
                        data=results, event="results",
                        sid=self.get_sid_of_data(eio_sid)
                    ),
                    send_error_function=lambda message: self.send_data(
                        data=message, event=DATA_ERROR_EVENT, sid=self.get_sid_of_data(eio_sid)
                    ),
                    config=config,
                    name=f"CLIP Worker {eio_sid}",
                    daemon=True,
                )
            except Exception as ex:
                logger.error(f"Failed to create CLIPWorker: {repr(ex)}")
                logger.error(traceback.format_exc())
                self.send_command_error(f"Failed to create CLIPWorker: {repr(ex)}", sid)
                return False, f"Failed to create CLIPWorker: {repr(ex)}"

            self.tasks[eio_sid] = TaskAndWorker(task, worker)
            self.tasks[eio_sid].worker.start()
            t0 = time.perf_counter_ns()
            while True:
                if self.tasks[eio_sid].worker.is_alive():
                    break
                if time.perf_counter_ns() > t0 + 5 * 1.0e9:
                    logger.error(f"Timed out to start worker, eio_sid {eio_sid}, sid {sid}")
                    return False, f"Timed out to start worker"

            logger.info(f"Task handler and worker created and started: {eio_sid}")

        logger.info(
            f"Control command applied, eio_sid {eio_sid}, sid {sid}, "
            f"results sid {self.get_sid_of_data(eio_sid)}, command {command}"
        )
        return True, (
            f"Control command applied, eio_sid {eio_sid}, sid {sid}, results sid"
            f" {self.get_sid_of_data(eio_sid)}, command {command}"
        )

    def disconnect_callback(self, sid: str) -> None:
        """Called with client disconnection - deletes task and worker.

        Args:
            sid (str): Namespace sid.
        """

        eio_sid = self.get_eio_sid_of_data(sid)
        task_and_worker = self.tasks.get(eio_sid)
        if task_and_worker:
            task_and_worker.worker.stop()
            task_and_worker.worker.join()
            del self.tasks[eio_sid]
            del task_and_worker
            logger.info(f"Task handler and worker deleted: {eio_sid}")

        logger.info(f"Client disconnected from {DATA_NAMESPACE} namespace, eio_sid {eio_sid}, sid {sid}")


def main():
    """Main function."""

    # parser = argparse.ArgumentParser(description='Forward Collision Warning Service')
    # args = parser.parse_args()

    logger.info(f"The size of the queue set to: {NETAPP_INPUT_QUEUE}")

    # logger.info("Initializing default object detector for faster first startup")
    # detector = YOLODetector.from_dict({})
    # del detector

    server = Server(port=NETAPP_PORT, host="0.0.0.0", extended_measuring=EXTENDED_MEASURING)

    try:
        server.run_server()
    except KeyboardInterrupt:
        logger.info("Terminating ...")


if __name__ == "__main__":
    main()
