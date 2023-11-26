from setuptools import setup, find_packages

setup(
    name="tts_server",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "flask",
        "styletts2 @ git+https://github.com/lxe/styletts2.git",
    ],
    entry_points={
        "console_scripts": [
            "tts-server = tts_server.server:main",
            "tts-server-cli = tts_server.cli:main",
        ],
    },
)