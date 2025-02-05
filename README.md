# EEG Theta Band Fatigue Detection

![Project Banner](docs/image_logo.png)

## Table of Contents

- [About the Project](#about-the-project)
  - [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [API Deployment](#api-deployment)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## About the Project

This project focuses on detecting fatigue levels by analyzing the EEG Theta band. Utilizing machine learning models, the system processes EEG data to provide real-time fatigue predictions.

### Built With

- [Python](https://www.python.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)

## Getting Started

To set up the project locally, follow these steps.

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your_username/EEG-Theta-Fatigue-Detection.git
   cd EEG-Theta-Fatigue-Detection
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Prepare Your EEG Data

Ensure your EEG data is in the correct format as expected by the model. The data should be in .mat files and placed in the `data/` directory.

### 2. Run the FastAPI Application

```bash
uvicorn main:app --reload
```

The API will be accessible at [http://127.0.0.1:8000](http://127.0.0.1:8000).

### 3. Make Predictions

Use tools like Postman or curl to send POST requests to the API with your EEG data to receive fatigue predictions.

## Model Training

To train the model:

### 1. Prepare the Dataset

Place your EEG .mat files in the `data/` directory. Each file should contain the necessary EEG signals for training.

### 2. Run the Training Script

```bash
python train_model.py
```

This will process the data, extract features, train the model, and save it as `model.pkl`.

## API Deployment

The FastAPI application (`main.py`) loads the trained model (`model.pkl`) and provides an endpoint for real-time fatigue prediction. Ensure the model file is in the same directory as `main.py`.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## Contact

Syed Mohd Haider Rizvi - [smhrizvi281@gmail.com](mailto:smhrizvi281@gmail.com)

Project Link: [EEG Theta Band Fatigue Detection](https://github.com/haider1998/EEG-Theta-Fatigue-Detection)

## Acknowledgements

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [GitHub Markdown Guide](https://docs.github.com/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)

---

### Notes

- Replace placeholders like `your_username`, `your.email@example.com`, and `path_to_your_image.png` with your actual information.
- Ensure that the `requirements.txt` file lists all necessary dependencies for your project.
- The `train_model.py` script should contain the code for training your machine learning model and saving it as `model.pkl`.
- The `main.py` script should set up the FastAPI application to load the `model.pkl` and provide endpoints for predictions.
