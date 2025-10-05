# ExoSeeker

ExoSeeker is an interactive web user interface for creating custom machine learning models to analyze Kepler object of interests. It empowers anyone to easily discover potential exoplanets by having a streamlined process of taking in new data, build a unique machine learning model, and generate predicted classifications of objects of interests.

## Features

- Its has user-friendly web interface allows anyone to to create their own AI models for identifying new exoplanets. 
- Each component has a tooltip that provides useful information such as instructions in uploading new data, properties of each estimators' main hyperparameters, and the description of each metric in the model evaluation. 
- ExoSeeker allows for stacking multiple estimators enabling to harness the strength of one to increase the model's performance. 
- The user can mix and match multiple models, tweak each ones' main hyperparameters, empowering the user to create fully customized machine learning models.

## Prerequisites
Please use Python version >= 3.13.7

## Installation

Clone the repository:
```powershell
git clone https://github.com/gospacedev/exoplanet.git
```

Create a  virtual enviorment:
```powershell
python -m venv exoseeker-venv
```

Activate the virtual enviorment:
```powershell
exoseeker-venv/Scripts/Activate.ps1
```

Install the requirements:
```powershell
pip install -r requirements.txt
```

Run the web interface:
```powershell
streamlit run app.py
```

## Getting Started

1. Go to dataset from [Kepler Objects of Interest](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)

2. Download the cumulative dataset from "Download Table"

3. You can use the Jupyter notebook named "" to split the down loaded dataset into two files:
- training_dataset.csv: a copy of the cumulative dataset with the last one thousand rows dropped
- target_data.csv: the last one thousand rows of the downloaded data with the exoplanet disposition removed to be used for predictions

4. The traning dataset can then be uploaded as trinaing data to Exoseeker

5. You can create your own custom machine learning model in the model build section, select estimators and adjust their main hyperparameters

6. Once the model has been trained, the model would be saved locally as a pickle file and its performance would be visualized in the model eveluation

7. You can then go to target data forecast to run predictions on target_data.csv
