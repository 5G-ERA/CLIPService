from setuptools import setup

package_name = "openclip_service_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Petr Kleparnik",
    maintainer_email="p.kleparnik@cognitechna.cz",
    description="CLIP Service Network Application - ROS 2 service",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "openclip_service_node = openclip_service_ros2.openclip_service_node:main",
        ],
    },
)
