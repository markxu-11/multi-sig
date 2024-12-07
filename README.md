# Signal Preprocessing Library

This library provides a modular and extensible framework for preprocessing multichannel signal data. It includes tools for windowing, digital filtration, and feature extraction, along with a manager class to coordinate these preprocessing tasks.

## Features

- **Modular Design**: Each preprocessing task is encapsulated in a dedicated class.
- **Windowing**: Apply windowing schemes with overlap to multichannel signals.
- **Digital Filtration**: Add filters, including low-pass, high-pass, band-pass, and notch filters.
- **Feature Extraction**: Extract features from signal data using time-domain methods.
- **Task Manager**: Easily manage and cascade preprocessing tasks for streamlined processing.
- **Visualisation**: Visualise signal data using Plotly.

## Dependencies

This code uses the following dependencies:
- **Python**: Version 3.7 and above
- **NumPy**
- **SciPy**
- **Plotly**: Optional for visualisation
