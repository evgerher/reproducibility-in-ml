# Ydata-demo ML project

During workshop we will:  
- Transform initial experiments from notebooks into reproducible python code.  
- Make out code and data shared among the rest of the team (cloud storage, git)  
- Get familiar with tools like boto3 (s3 python api), dvc, mlflow.  


Folder structure
```
1_notebooks - folder with experiments around dataset. That is how we started to work on this project
2_python - folder with pre-production code, here we converted all the knowledge from notebooks into reusable code.
3_dvc_mlflow - folder with final code, we added mlflow and DVC functionality
```

I made third folder a separate git-project to make clean representation for dvc+git integration.


### DVC

What is DVC?
> Open-source Version Control System for Machine Learning Projects

Documentation https://dvc.org  

### mlflow

What is mlflow?
> mlflow is a framework that supports the machine learning lifecycle. This means that it has components to monitor your model during training and running, ability to store models, load the model in production code and create a pipeline.


Documentation https://www.mlflow.org  

Alternatives: 
- Weights & biases: https://wandb.ai/site  
- Neptune: https://neptune.ai  
- Comet: https://www.comet.com/site/  
- Pulsar (our tool): 
