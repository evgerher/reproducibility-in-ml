# ML project example (ydata-demo)

### Dependencies installation

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage

`python ml_project/train.py --config configs/train_config.yaml`

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
- artifacts - 
  - reports - generated .html files from notebooks folder
  - models - trained model binary files and other files

requirements.txt   <- The requirements file for reproducing the analysis environment
```
