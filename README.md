
# CervHernia-YOLO: Enhanced YOLOv10 Model for Cervical Spine MRI Segmentation

**CervHernia-YOLO:** is an advanced instance segmentation model based on YOLOv10, specifically designed for precise segmentation of cervical MRI images. The model integrates state-of-the-art techniques to improve segmentation accuracy, making it an effective tool for diagnosing cervical spine pathologies, such as disc bulging and herniation.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
  - [Dependencies](#dependencies)
  - [Clone the Repository](#clone-the-repository)
- [Usage](#usage)
- [Author Contributions](#author-contributions)
- [License](#license)

## Introduction

The CervHernia-YOLO model builds upon YOLOv10 by integrating several advanced techniques to enhance segmentation accuracy for small, complex targets in MRI images. This model focuses on cervical spine pathology segmentation and includes the following components:

- **YOLOv10**: Serves as the foundational model, known for efficient object detection performance. [YOLOv10 Paper](https://arxiv.org/pdf/2405.14458)
- **EfficientNetV2**: Optimized network architecture, providing high computational efficiency. [EfficientNetV2 Paper](https://arxiv.org/pdf/2104.00298)
- **Inner-ShapeIoU**: Improved IoU metric for robust shape-based segmentation. [ShapeIoU Paper](https://arxiv.org/pdf/2312.17663),[InnerIoU Paper](https://arxiv.org/pdf/2311.02877)
- **SDI (Shape Deformation-Integrated)**: Allows flexible shape deformation, improving performance in complex segmentation tasks. [SDI Paper](https://arxiv.org/pdf/2311.17791v1)
- **Dynamic Snake Convolution**: Enhanced boundary detection for precise segmentation of complex shapes. [Dynamic Snake Convolution Paper](https://arxiv.org/pdf/2307.08388)

Each of these advancements contributes to CS-YOLO’s strong performance on benchmark datasets, making it a valuable tool for cervical spine MRI segmentation.

## Features

- **Precise segmentation** of small and complex targets in cervical MRI images.
- Enhanced YOLOv10-based model, incorporating EfficientNetV2, Inner-ShapeIoU, SDI, and Dynamic Snake Convolution for improved accuracy.
- Specialized for the **diagnosis of cervical spine pathologies**.

## Installation

### Dependencies

To set up CervHernia-YOLO:, first install the required dependencies:

```bash
pip install -r requirements.txt
```

### Clone the Repository

Clone this repository and navigate into it:

```bash
git clone https://github.com/username/CS-YOLO.git
cd CervHernia-YOLO:
```

Then, install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run CervHernia-YOLO: on our dataset, simply use the `main.py` file. Please note that the dataset is private and not included in this repository.

```bash
python main.py
```

## Author Contributions

**Project Leadership**: HG conceived the initial idea and designed the study framework.

**Development and Implementation**: DW, XL, and WL led the software development, model construction, data processing, and analysis required for the project.

**Data Collection and Management**: DW and XL were responsible for the collection, organization, and preparation of the datasets used in the study.

**Writing and Documentation**: DW and XL drafted the initial documentation and code annotations. HG conducted validation, provided additional insights, and completed the final manuscript revisions.

All authors reviewed and approved the final version of the code and manuscript prior to submission.

## License

CS-YOLO is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).
