# Chaotic Industrial Time Series Prediction

The official release of the `ChaoFactory` dataset and implementation of `Chaotic_Industrial_TimeSeriesPrediction` model.

## 1. Dataset Overview

### Chaotic Factory Dataset (ChaoFactory.csv)

This dataset is a robust collection of sensor data from a complex, multi-input multi-output (MIMO) production line in a real industrial setting, specifically designed for machine learning applications in time-series prediction, anomaly detection, and system optimization.

**Download**
The data can be downloaded from [Google Drive](https://drive.google.com/file/d/1FACTZXEUtZqULQIYYJmfNzcSw-Rih_Xe/view?usp=sharing), please unzip and save it to the project directory.

**Data Collection**

- **Original File**: `BigDataFactory.csv` located at `ChaoFactoryDataset\BigDataFactory.csv`.
- **Processed File**: After channel selection and mapping through gateway addresses, the refined dataset is available at `.\ChaoFactoryDataset\ChaoFactory.csv`.

**Characteristics**

- **Source**: "Chaotic Factory"
- **Duration**: Five months
- **Sensors**: 284 unique sensors
- **Data Points**: Approximately 540,000 timestamps
- **Frequency**: Every 30 seconds

**Attributes**
Includes 12 key attributes such as `MillExitTemp` (Grinding mill outlet temperature) and `FurnaceExitTemp` (Hot air furnace outlet temperature), among others, essential for monitoring various aspects of the production process like temperature, pressure, and current.

## 2. Environment Setup

### Installing Dependencies

Before running the model, ensure all required dependencies are installed. Use the following steps to set up your environment:

1. **Install Anaconda or Miniconda**
   If not already installed, download and install Anaconda or Miniconda from their official websites.
2. **Create a Conda Environment**
   Create a new Conda environment to isolate your package installations:

   ```bash
   conda create --name your_env python=3.8.10
   conda activate your_env
   ```
3. **Install Required Packages**
   Install all necessary packages using the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

### Activate Environment

Activate the Conda environment using the following command:

```bash
conda activate your_env
```


## 3. Model Execution

### Run Pre-Trained Model

Navigate to the code directory and run the model:

```bash
cd Factory
python test.py
```

- **Test Dataset Path**: `Factory/dataset`
- **Model File Path**: `Factory/trained_model`

### Training on Opendata Dataset

For a simplified training process:

```bash
cd Opendata
python train.py
```

- **Test Dataset Path**: `Opendata/dataset`


## Outreach

Thanks for the hardworking you put into the review! We would love to address about questions during rebuttal!
