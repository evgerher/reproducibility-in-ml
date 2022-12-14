# ML project example (ydata-demo)

### Dependencies installation

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage

Run training `python ml_project/train.py --config configs/train_config.yaml`  
Upload model `python ml_project/upload_s3.py` :: _Modify manually what to upload and where!_

### Test


Tests framework == `pytest`

`pytest tests/`

---

### Project structure

```
- ml_project - folder with source code
  - data: package for loading from local/remote source data
  - features: package for data transformation
  - models: package for model training procedures
- configs - folder with configuration files to run training experiments
  - train_lr.yaml - example of linear regression config
  - train_rf.yaml - example of random forest regressor config
- tests - folder with tests files
- notebooks - folder with initial .ipynb files
- artifacts
  - models - trained model binary file and other files

requirements.txt   <- The requirements file for reproducing the analysis environment
```
