# Automatic License Plate Detector 🚗💨

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📖 About

**Automatic License Plate Detector** is a computer vision project designed to detect and recognize license plates from vehicles in images or video streams. 

This system utilizes deep learning and image processing techniques to identify the location of a license plate and extract the alphanumeric characters using Optical Character Recognition (OCR). It is suitable for applications such as automated parking systems, traffic monitoring, and vehicle identification.

## ✨ Features

- **License Plate Detection:** Accurately locates vehicle license plates within an image.
- **Character Recognition (OCR):** Extracts text from the detected plates.
- **Real-time Processing:** Capable of processing video feeds for live detection.
- **Image Preprocessing:** Includes gray-scaling, blurring, and edge detection to improve accuracy.
- **Visual Output:** Draws bounding boxes around detected plates and overlays the recognized text.

## 🛠️ Technologies Used

* **Python**: Core programming language.
* **OpenCV**: For image processing and video manipulation.
* **EasyOCR / Tesseract**: For Optical Character Recognition.
* **Pandas/NumPy**: For data handling and matrix operations.
* **Matplotlib**: For visualization (optional).

## 📋 Hardware Requirements

The following hardware specifications are recommended for deploying the Automatic License Plate Detector in production environments:

| Component            | Minimum          | Recommended                    |
| -------------------- | ---------------- | ------------------------------ |
| **CPU**              | Intel i5 8th Gen | Intel i7 10th Gen or newer     |
| **RAM**              | 8 GB             | 16 GB                          |
| **Storage**          | 128 GB SSD       | 256 GB SSD or larger           |
| **Operating System** | Windows 10       | Windows 10/11 or Ubuntu        |
| **Camera**           | 720p USB Camera  | 1080p IP Camera                |
| **Power Backup**     | UPS Backup       | UPS with 4+ hours backup       |
| **Cooling**          | Active Cooling   | Proper ventilation and cooling |

### Notes

* Higher CPU and RAM configurations improve real-time detection performance.
* SSD storage is recommended for faster data access and model loading.
* A 1080p camera provides better image quality, resulting in more accurate license plate recognition.
* For continuous operation, ensure adequate cooling and power backup to prevent downtime.


## 🚀 Getting Started

Follow these instructions to set up the project on your local machine.

### Prerequisites

Ensure you have Python installed. You can check this by running:
```bash
python --version
```

### Installation
1. Clone the repository
```bash
git clone [https://github.com/varun-kul/automatic-license-plate-detector.git](https://github.com/varun-kul/automatic-license-plate-detector.git)
cd automatic-license-plate-detector
```
2. Create a Virtual Environment (Optional but Recommended)

 Windows
```bash
python -m venv venv
.\venv\Scripts\activate
```
 macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```
(Note: If a requirements.txt file is missing, you typically need to install the following common libraries):
```bash
pip install opencv-python easyocr numpy matplotlib imutils
```

## 💻 Usage

1. Detect from an Image
Place your test images in the images/ folder (or specify your path) and run the main script.
```bash
python main.py --image images/car1.jpg
```

2. Detect from Video / Webcam
To run the detector on a video file or live webcam feed:
 For webcam
 ```bash
  python main.py --source 0
```

 For video file
 ```bash
 python main.py --source videos/traffic.mp4
```
(Note: Replace main.py with the actual name of your script, e.g., detect.py or app.py)

## 📂 Project Structure
automatic-license-plate-detector/

├── images/                # Sample images for testing

├── videos/                # Sample videos (optional)

├── model/                 # Pre-trained models (if any)

├── main.py                # Main script for detection

├── utils.py               # Helper functions (preprocessing, OCR)

├── requirements.txt       # Python dependencies

├── README.md              # Project documentation

└── LICENSE                # License file

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact
Varun Kul GitHub: @varun-kul

Project Link: https://github.com/varun-kul/automatic-license-plate-detector
