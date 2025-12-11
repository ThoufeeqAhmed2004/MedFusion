# MedFusion
MedFusion - Adaptive Foundation Models for Medical Image Segmentation, seeks to enhance segmentation accuracy and versatility with the Segment Anything Model (SAM). SAM has a high performance in prompt-based segmentation. This work presents a scalable and versatile solution to the changing demands of medical imaging and diagnostics.

## Getting Started

You can run this project using either `uv` (for local development) or `Docker` (for a containerized environment).

## Dataset Structure

The project expects the dataset to be organized in the following YOLO-compatible structure:

```
data/
└── KIDNEY_CT/
    ├── data.yaml       # Class names and number of classes
    ├── train/
    │   ├── images/     # Training images (.jpg/.png)
    │   └── labels/     # YOLO-format labels (.txt)
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

### Label Format
Labels should be in **YOLO format** (`.txt` files matching image names):
```
<class_id> <x_center> <y_center> <width> <height>
```
*   Coordinates are normalized (0-1).
*   `class_id`: Integer representing the class index (e.g., 0 for Kidney/Tas_Var).

### Example `data.yaml`
```yaml
names:
- Tas_Var
nc: 1
train: ../train/images
val: ../valid/images
test: ../test/images
```

### Option 1: Local Development with uv

[uv](https://github.com/astral-sh/uv) is an extremely fast Python package installer and resolver.

1.  **Install uv** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Create a virtual environment**:
    ```bash
    uv venv
    ```

3.  **Activate the virtual environment**:
    ```bash
    # On macOS/Linux
    source .venv/bin/activate
    # On Windows
    .venv\Scripts\activate
    ```

4.  **Install dependencies**:
    ```bash
    uv pip install -r requirements.txt
    ```

5.  **Run the application**:
    ```bash
    python src/main.py
    ```

### Option 2: Docker Development

This method ensures a consistent environment and is recommended for compatibility, especially if you need to manage CUDA versions.

1.  **Build the Docker image**:
    ```bash
    docker build -t medfusion .
    ```

2.  **Run the container**:
    We mount the current directory (`$(pwd)`) to `/app` in the container so you can edit files locally and run them immediately. GPU support is enabled with `--gpus all`.

    ```bash
    docker run --gpus all -it -v $(pwd):/app medfusion /bin/bash
    ```

### Option 3: Google Colab

MedFusion is compatible with Google Colab.

1.  **Clone the repository** (in a code cell):
    ```python
    !git clone https://github.com/sundaresanv2004/MedFusion.git
    %cd MedFusion
    ```

2.  **Install Dependencies**:
    ```python
    !pip install -r requirements_colab.txt
    ```

3.  **Run the Inference**:
    ```python
    !python src/main.py
    ```
    *Note: Objects and outputs will be saved to the `output/` directory in the Colab file explorer.*
