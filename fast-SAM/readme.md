# Fast-SAM Setup and Usage

## Setup Instructions

1. Clone the repository:
    ```sh
    git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
    ```

2. Navigate to the `fast-SAM` directory:
    ```sh
    cd fast-SAM
    ```

3. Download the weights:
    ```sh
    !wget https://huggingface.co/spaces/An-619/FastSAM/resolve/main/weights/FastSAM.pt
    ```

4. Install the required Python packages:
    ```sh
    !pip install -r FastSAM/requirements.txt
    !pip install git+https://github.com/openai/CLIP.git
    ```

5. Add the following code to `fast-SAM/FastSAM/fastsam/__init__.py`:
    ```python
    from pathlib import Path
    import sys
    ROOT_DIR = Path(__file__).parents[1].as_posix()
    sys.path.append(ROOT_DIR)
    ```

6. Edit the desired configuration in `config.json`.

## Usage

1. Run the pipeline:
    ```sh
    python pipeline.py
    ```

2. The result will be saved in the `img/output` directory.