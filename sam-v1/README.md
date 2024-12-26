# SAM

## Setup Instructions

1. Clone the repository:
    ```sh
    git clone https://github.com/m15kh/SAM.git
    ```

2. Navigate to the `sam-v1` directory:
    ```sh
    cd sam-v1
    ```

3. Clone the Segment Anything repository and install it:
    ```sh
    git clone git@github.com:facebookresearch/segment-anything.git
    cd segment-anything
    pip install -e .
    ```

4. Install the required Python packages:
    ```sh
    pip install opencv-python pycocotools matplotlib onnxruntime onnx
    ```

5. Navigate to the `sam-v1/checkpoint` directory:
    ```sh
    cd ../checkpoint
    ```

6. Download the checkpoint file:
    ```sh
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    ```

## Usage

1. Ensure your input image is placed in the `img` directory and update the `config.json` file with the correct `input_image_path`.

2. Run the pipeline:
    ```sh
    python pipeline.py
    ```

3. The output will be saved in the `output` directory.