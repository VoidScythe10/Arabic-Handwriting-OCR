# Arabic Handwriting OCR Pipeline

A simple pipeline for running Arabic handwriting OCR using a pretrained model from Hugging Face (not mine, linked below). This project wraps the model in an easy-to-use script/pipeline so you can go from an image of Arabic handwriting to extracted text with minimal setup.

## Features

- Highly accurate Arabic handwriting recognition.
- Offers batch processing.
- Uses natsort to process the images in a convenient order.
- Efficient CPU/GPU-wise.
- Creates a PDF file of the results.
- Fully open source!

## Model

This project uses the following model from Hugging Face:

- **Model**: ((https://huggingface.co/sherif1313/Arabic-English-handwritten-OCR-v3))
- All credit for the model itself goes to its original authors.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/VoidScythe10/Arabic-Handwriting-OCR
   ```

2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```

3. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Requirements


```
os
json
logging
pathlib
torch
PIL
tqdm
natsort
transformers
qwen_vl_utils
huggingface_hub
```



## Usage

Open you code editor, then edit the directory paths. 

## Project Structure

```
Arabic-Handwriting-OCR/
├── ocr_pipeline.py      # main script
├── requirements.txt     # dependencies
└── README.md
```

## License

See the original model's license

## Acknowledgments

- Thanks to the creators of the original Hugging Face model this pipeline is built on.
