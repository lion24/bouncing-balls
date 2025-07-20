# Inner Circle

A basic Python 3 project using pygame, managed with Poetry.

## Demo

[![Watch the video](https://github.com/lion24/bouncing-balls/blob/main/demo/thumbnail.png?raw=true)](https://github.com/lion24/bouncing-balls/raw/refs/heads/main/demo/ball_escape.mp4)

## Features

- Simple pygame application with a bouncing ball
- Managed with Poetry for dependency management
- Ready-to-run example

## Requirements

- Python 3.13+
- Poetry

## Installation

1. Make sure you have Poetry installed
2. Install dependencies:
   ```bash
   poetry install
   ```

## Running the Application

You can run the application in several ways:

### Using Poetry script
```bash
poetry run inner-circle
```

### Using Python module
```bash
poetry run python -m inner_circle.main
```

### Using Poetry shell
```bash
poetry shell
python src/inner_circle/main.py
```

## Dependencies

- pygame (>=2.6.1,<3.0.0)
