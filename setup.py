from setuptools import setup, find_packages

setup(
    name="xsynth",
    version="0.1.0",
    author="Trey Guest",
    author_email="trey.guest@xfel.eu",
    description=(
        "A package for controlling kicker devices and performing beam steering "
        "using N-dimensional scan tools."
    ),
    packages=find_packages(),
    install_requires=[
        "numpy",
        "xarray",
        "numexpr",
        "tqdm",
        "IPython",
    ],
    extras_require={
        "gui": [
            "dearpygui",
        ],
    },
    entry_points={
        "console_scripts": [
            "xsynth-gui=xsynth.gui.app:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)