from setuptools import setup, find_packages

package_name = "da3_stream"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer='jun',
    maintainer_email='snowpoet@naver.com',
    description="Depth Anything 3 streaming node (vendored depth_anything_3).",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "da3_depth_node = da3_stream.da3_depth_node:main",
            "da3_pcd_node = da3_stream.da3_pcd_node:main",
        ],
    },
    include_package_data=True,  # <- 중요: MANIFEST.in에 적은 데이터 파일도 설치에 포함
)
