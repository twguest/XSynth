# xsynth — Examples & Demo

This repository contains the `xsynth` package and example notebooks demonstrating how to create devices, signals, and scans.

Getting started

- Install the package (editable) in your environment:

```bash
pip install -e .
```

- Install the GUI extras if you want the desktop interface:

```bash
pip install -e .[gui]
```

- Start the GUI with:

```bash
xsynth-gui
```

- Open the example notebooks in `examples/` with Jupyter or JupyterLab.

Files

- `examples/fady_demo.ipynb`: A step-by-step notebook showing progressively more complex examples of devices, signals, and scans.

Quick demo outline

- Example 1: Basic device set and read operations
- Example 2: Single-device scan with a simple oscillator
- Example 3: Multi-device multi-signal scan combining different oscillators
