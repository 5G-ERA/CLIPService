import json
import logging
import os
from queue import Queue
from typing import Dict, List

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Publisher
from rclpy.time import Time
from std_msgs.msg import String
from sensor_msgs.msg import Image

from openclip_service.clip_worker import CLIPWorker
from openclip_service_ros2.ros2_numpy_image import *
from era_5g_interface.task_handler_internal_q import TaskHandlerInternalQ, QueueFullAction

INPUT_TOPIC = str(os.getenv("INPUT_TOPIC", "/image_raw"))
OUTPUT_TOPIC = str(os.getenv("OUTPUT_TOPIC", "/res"))

NETAPP_INPUT_QUEUE = int(os.getenv("NETAPP_INPUT_QUEUE", 1))

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CLIP service node")

# TODO: This is not functional yet.

class Worker(CLIPWorker):
    """Worker class."""

    def __init__(
        self,
        image_queue: Queue,
        publisher: Publisher,
        config: Dict,
        **kw,
    ) -> None:
        super().__init__(
            image_queue=image_queue,
            send_function=self.publish_results,
            config=config,
            send_error_function=self.worker_error_callback,
            name=f"CLIP Worker {id(self)}",
            daemon=True,
            **kw,
        )
        self.publisher = publisher

    def worker_error_callback(self, message):
        pass

    def publish_results(self, results) -> None:
        """Publish results to ROS 2 topic.

        Args:
            results ():
        """

        msg = String()
        msg.data = json.dumps(results)
        self.publisher.publish(msg)


def parameters_to_dict(parameters: Dict) -> Dict:
    """Convert ROS parameters dict into new dict.

    Args:
        parameters (Dict): ROS parameters dict

    Returns:
        New simple form of parameters dict used in CLIP worker.
    """
    parameters_dict = {}
    for name, value in parameters.items():
        keys = name.split(".")
        dict_inner = parameters_dict
        for key in keys:
            if key not in dict_inner:
                dict_inner[key] = {}
            if key == keys[-1]:
                dict_inner[key] = value.value
            dict_inner = dict_inner[key]
    return parameters_dict


class CLIPServiceNode(rclpy.node.Node):
    """CLIP Service ROS 2 Node."""

    def __init__(self) -> None:
        """Constructor."""

        super().__init__("clip_service_node", automatically_declare_parameters_from_overrides=True)

        # Set ROS 2 parameter callback.
        self.add_on_set_parameters_callback(self.parameter_callback)

        # This get parameters which are set through command line arguments.
        self.config_dict = parameters_to_dict(self.get_parameters_by_prefix("config"))
        self.get_logger().info(f"config dict: {self.config_dict}")

        self.publisher = self.create_publisher(String, OUTPUT_TOPIC, 10)
        self.subscriber = self.create_subscription(Image, INPUT_TOPIC, self.image_callback, 10)

        # Queue with received images.
        self.image_queue = Queue(NETAPP_INPUT_QUEUE)
        self.task_handler = None
        self.worker = None
        # Start worker only with config.
        if self.config_dict:
            self.start()

    def start(self) -> None:
        """Start worker, delete old worker."""

        self.get_logger().info(f"Starting new worker...")

        if self.task_handler:
            del self.task_handler
        if self.worker:
            del self.worker

        self.image_queue.queue.clear()
        self.task_handler = TaskHandlerInternalQ(self.image_queue, if_queue_full=QueueFullAction.DISCARD_OLDEST)

        # Create new worker.
        self.worker = Worker(
            self.image_queue,
            self.publisher,
            self.config_dict,
        )
        # Start worker.
        self.worker.start()

    def parameter_callback(self, parameters: List[rclpy.Parameter]) -> SetParametersResult:
        """Parameter callback - ROS 2 parameter service used for CLIP parameters. Starts new Worker with new parameters.

        Args:
            parameters (List[rclpy.Parameter]): CLIP parameters.

        Returns:
            SetParametersResult
        """
        try:
            self.get_logger().info(f"parameters: {parameters}")

            parameters_dict = {}
            for parameter in parameters:
                parameters_dict[parameter.name] = parameter
                self.get_logger().debug(f"{parameter.name} {parameter.value}")

            parameters_dict = parameters_to_dict(parameters_dict)

            self.config_dict = parameters_dict.get("config", self.config_dict)
            self.get_logger().info(f"config dict: {self.config_dict}")

            # Start worker only with config.
            if self.config_dict:
                self.start()
        except Exception as ex:
            self.get_logger().error(f"Parameter callback exception: {repr(ex)}")
            return SetParametersResult(successful=False)

        return SetParametersResult(successful=True)

    def image_callback(self, image: Image) -> None:
        """Image callback.

        Args:
            image (Image): Image.
        """

        try:
            # Convert the ROS image message to numpy format.
            np_image = image_to_numpy(image)
        except TypeError as e:
            self.get_logger().error(f"Can't convert image to numpy. {e}")
            return
        # TODO: create custom ROS2 message with images and texts.
        if np_image is not None:
            metadata = {
                "timestamp": Time.from_msg(image.header.stamp).nanoseconds,
                "recv_timestamp": self.get_clock().now().nanoseconds,
            }
            if self.task_handler is not None:
                # Check worker is alive
                if not self.worker.is_alive():
                    self.get_logger().error(f"Worker is not alive")
                    rclpy.shutdown()
                    self.executor.remove_node(self)
                    return
                self.task_handler.store_data(metadata, np_image)
            else:
                self.get_logger().warning("Uninitialized task handler and worker!")
        else:
            self.get_logger().warning("Empty image received!")


def main(args=None) -> None:
    """Main function."""

    rclpy.init(args=args)
    node = CLIPServiceNode()

    try:
        # Spin until interrupted.
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=1.0)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Exception in node: {type(e)}): {repr(e)}")
        raise
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    if None in [INPUT_TOPIC, OUTPUT_TOPIC]:
        logger.error("INPUT_TOPIC and OUTPUT_TOPIC environment variables needs to be specified!")
    else:
        main()
