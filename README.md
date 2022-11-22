## Project Overview

- The goal of this project is to train machine learning classification models to predict default probabilities of [Lending Club dataset](https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1) loans issued in 2018 by training these models on pre-2018 loan data from this dataset
- Leveraging these predictions, an IRR-optimized portfolio of highest-yielding 2018 loans is constructed for a hypothetical investor looking to maximize his or her returns on this loan set
- To better present and visualize key ML results and recommendations an interactive dashboard application using a Python Dash frontend and Flask backend is created
    - See **Build and run app** for instructions on how to build and run app 
    - See **Sample app visualizations** for sample screenshots of app


## Build and run app

If docker-compose not already installed, see installation [instructions](https://docs.docker.com/compose/gettingstarted/) 

**Scripted e2e**:
```
# Run from root dir
bash build_e2e.sh
```

**Manually**:

- See **/app/backend/build_backend.md** for instructions on how to manually build backend
- See **/app/frontend/build_frontend.md** for instructions on how to manually build frontend

## Sample app visualizations

1. Distributions of loan grades by state:

<div align="center">
    <a href="./">
        <img src="./images/choropleth.gif" width="79%"/>
    </a>
</div>

  

2. Loan default rates & interest rates vs FICO score:

<div align="center">
    <a href="./">
        <img src="./images/lineplots_bargraphs.gif" width="79%"/>
    </a>
</div>

  

3. Retrieve live ML model default predictions on sample anonymized customer data:

<div align="center">
    <a href="./">
        <img src="./images/predict_models.gif" width="79%"/>
    </a>
</div>


## Dataset
- See [Kaggle dataset](https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1)


## Blog post + live presentation

- Link to [blog post](https://nycdatascience.com/blog/student-works/predicting-loan-defaults-using-machine-learning-classification-models/) 

- Link to [live presentation](https://www.youtube.com/watch?v=1U1pIe5-GZ0&ab_channel=NYCDataScienceAcademy) (youtube)


## Presentation slides

Please refer to **/presentation/NYCDSA_Capstone_Presentation_vF.pdf** for presentation slides of presentation given on January 5th, 2021 to NYCDSA prospective students and alums regarding this project
