from argparse import ArgumentParser
from typing import List, Sequence, Set
import sys

import rclpy
from rcl_interfaces.srv import SetParametersAtomically
from rcl_interfaces.msg import Parameter
from rclpy.node import Node
from rclpy.task import Future


class SetParametersAtomicallyNode(Node):
    def __init__(self, service_name: str):
        super().__init__('set_parameters_atomically_node', automatically_declare_parameters_from_overrides=True)

        self.future = None
        self.client = self.create_client(SetParametersAtomically, service_name)
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again ...')
        self.req = SetParametersAtomically.Request()
        print(self.get_parameters_by_prefix(""))

    def send_request(self, parameters: List[Parameter] = None):
        if parameters:
            self.req.parameters = parameters
        else:
            p: List[rclpy.Parameter] = list(self.get_parameters_by_prefix("").values())
            self.req.parameters = [v.to_parameter_msg() for v in p]
        print(self.req.parameters)
        self.future: Future = self.client.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def main(args=None) -> None:
    """Main function."""
    rclpy.init(args=args)
    parser = ArgumentParser(description='Sets parameters atomically from yaml file.')
    parser.add_argument("-s", "--service_name", type=str, help="The name of the service to set the parameters, e.g. /node_name/set_parameters_atomically", default="set_parameters_atomically")
    args = parser.parse_args()
    node = SetParametersAtomicallyNode(args.service_name)
    node.send_request()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
