from setuptools import setup, find_packages
from pathlib import Path

# Hack the styletts2 module things
Path('tts_server/styletts2/__init__.py').touch()
Path('tts_server/styletts2/Utils/PLBERT/__init__.py').touch()

setup(
    name="tts_server",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "cached_path",
        "nltk",
        "scipy",
        "numpy",
        "munch",
        "librosa",
        "phonemizer",
        "sounddevice",
        "einops",
        "einops_exts",
        "transformers",
        "matplotlib",
        "flask",
        "monotonic_align @ git+https://github.com/resemble-ai/monotonic_align.git",
    ],
    entry_points={
        "console_scripts": [
            "tts-server = tts_server.server:main",
            "tts-server-cli = tts_server.cli:main",
        ],
    },
)