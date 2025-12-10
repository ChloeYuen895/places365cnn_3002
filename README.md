# Places365 CNN Scene Classification

A comprehensive scene classification project using the Places365 dataset and VGG16 architecture for both inference and training.

## Overview

This project provides multiple interfaces for scene classification:
- **GUI Application**: Easy-to-use interface for classifying single images
- **Webcam Inference**: Real-time scene classification using your camera
- **Command-line Inference**: Batch processing and scripted usage
- **Custom Training**: Train your own models on subset datasets

The project uses the pre-trained Places365 VGG16 model for 365-class scene recognition, and includes tools for training custom models with fewer classes.

## Features

- **GUI Application** with drag-and-drop image classification
- **Real-time webcam inference** with camera switching support
- **Command-line tools** for batch processing
- **Custom training** for reduced class sets (e.g., 5-class arctic/bamboo/desert/forest/grassland)
- **GPU acceleration** support with CUDA
- **Training progress tracking** with validation metrics

## Dataset Classes (5-class subset)

The included `zoo5/` dataset contains 5 scene categories:
- **Arctic** - Snowy, icy environments
- **Bamboo** - Bamboo forests and groves  
- **Desert** - Arid, sandy landscapes
- **Forest** - Dense woodland areas
- **Grassland** - Open grassy plains

## Installation

### Prerequisites
- Python 3.11+
- NVIDIA GPU with CUDA 12.1+ (optional, for training acceleration)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd places365cnn_3002
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   # For inference only (OpenCV-based)
   pip install -r requirements.txt
   
   # For training (PyTorch with CUDA support)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

## Usage

### GUI Application

Launch the graphical interface for easy image classification:

```bash
python gui_deploy.py
```

Features:
- Click "Open Image" to select and classify images
- View predictions overlaid on the image (no popup dialogs)
- Save annotated results
- Supports common image formats (JPG, PNG, BMP, etc.)

### Webcam Inference

Real-time scene classification using your camera:

```bash
python webcam_inference.py
```

**Controls:**
- `q` or `ESC`: Quit
- `s`: Save current frame with prediction
- `p`: Pause/unpause  
- `c`: Switch between camera 0 and 1
- `h`: Toggle UI overlay

### Command-line Inference

Process single images or batches:

```bash
python deploy.py
```

### Training Custom Models

Train a VGG16 model on the 5-class dataset:

```bash
# Quick training (100 images, 5-15 minutes)
python train_cnn.py zoo5/ --arch vgg16 --num_classes 5 --batch-size 8 --epochs 15 --lr 0.001 --pretrained --workers 1

# Full training (1000+ images, 1-2 hours)  
python train_cnn.py zoo5/ --arch vgg16 --num_classes 5 --batch-size 16 --epochs 25 --lr 0.001 --pretrained --workers 2
```

**Training Parameters:**
- `--arch vgg16`: Use VGG16 architecture
- `--num_classes 5`: Number of output classes
- `--batch-size`: Batch size (adjust based on GPU memory)
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--pretrained`: Use ImageNet pre-trained weights (recommended)
- `--workers`: Number of data loading workers

**Output:**
- `vgg16_latest.pth.tar`: Latest checkpoint
- `vgg16_best.pth.tar`: Best performing model

## Project Structure

```
places365cnn_3002/
├── README.md                           # This file
├── requirements.txt                    # OpenCV dependencies
├── categories_places365.txt            # Class labels (365 classes)
├── deploy_vgg16_places365.prototxt    # Caffe model definition
├── vgg16_places365.caffemodel         # Pre-trained weights
├── gui_deploy.py                      # GUI application
├── webcam_inference.py                # Real-time webcam classifier
├── deploy.py                          # Command-line inference
├── train_cnn.py                       # PyTorch training script
├── zoo5/                              # 5-class dataset
│   ├── train/                         # Training images
│   │   ├── arctic/
│   │   ├── bamboo/
│   │   ├── desert/
│   │   ├── forest/
│   │   └── grassland/
│   └── val/                           # Validation images
│       ├── arctic/
│       ├── bamboo/
│       ├── desert/
│       ├── forest/
│       └── grassland/
└── test/                              # Test images
```

## Model Architecture

### Pre-trained Model (Inference)
- **Architecture**: VGG16
- **Framework**: Caffe
- **Classes**: 365 scene categories
- **Input**: 224×224 RGB images
- **Preprocessing**: BGR format, mean subtraction (104, 117, 123)

### Custom Training
- **Architecture**: VGG16 (PyTorch)
- **Transfer Learning**: ImageNet pre-trained weights
- **Classes**: Configurable (5 classes in included dataset)
- **Preprocessing**: Standard ImageNet normalization

## Performance

### Hardware Requirements
- **Minimum**: CPU-only inference
- **Recommended**: NVIDIA GPU with 4GB+ VRAM for training
- **Tested on**: RTX 4050 (training time: 1-2 hours for full dataset)

### Training Times (RTX 4050)
- 100 images: 5-15 minutes
- 1000+ images: 1-2 hours

## Troubleshooting

### Common Issues

1. **Module not found errors**: Ensure virtual environment is activated
2. **CUDA out of memory**: Reduce batch size or use CPU-only mode
3. **Camera not detected**: Check camera permissions and driver installation
4. **Import errors**: Install missing dependencies from requirements.txt

### GPU Setup
```bash
# Verify CUDA installation
nvidia-smi

# Check PyTorch GPU access
python -c "import torch; print(torch.cuda.is_available())"
```

## Citation

This project is based on the Places365 dataset. If you use this code or dataset, please cite:

```bibtex
@article{zhou2017places,
  title={Places: A 10 million Image Database for Scene Recognition},
  author={Zhou, Bolei and Lapedriza, Agata and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2017},
  publisher={IEEE}
}
```

## License

This project builds upon the Places365 dataset and models. Please refer to the original Places365 license and terms of use.

## Acknowledgments

- **Places365 Team**: Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, Antonio Torralba
- **Original PyTorch training code**: Based on PyTorch ImageNet examples
- **VGG Architecture**: Karen Simonyan and Andrew Zisserman

download weights and deploy file 
[https://github.com/CSAILVision/places365](https://github.com/CSAILVision/places365)