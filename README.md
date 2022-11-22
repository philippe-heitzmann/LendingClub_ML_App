## Project Overview

- The goal of this project is train machine learning classification models to predict default probabilities of [Lending Club dataset](https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1) loans issued in 2018 by training these models on pre-2018 loan data from this dataset
- The end goals is to uncover the best investment opportunity set for an investor looking to maximize his or her returns on the 2018 loan set. 
- To better present and visualize key ML results and recommendations an interactive dashboard application using a Python Dash frontend and Flask backend is created
    - See **Build and run app** for instructions on how to build and run app 
    - See **Sample app visualizations** for sample screenshots of app


## Build and run app

If docker-compose not already installed, see installation [instructions](https://docs.docker.com/compose/gettingstarted/) 

Scripted e2e:
```
# Run from root dir
bash build_e2e.sh
```

Manually:

- See /app/backend/build_backend.md for instructions on how to manually build backend
- See /app/frontend/build_frontend.md for instructions on how to manually build frontend

## Sample app visualizations

1. Distributions of loan grades by state:

<div align="center">
    <a href="./">
        <img src="./images/choropleth.gif" width="79%"/>
    </a>
</div>

1. Default rates and interest rate by FICO score:

<div align="center">
    <a href="./">
        <img src="./images/lineplots_bargraphs.gif" width="79%"/>
    </a>
</div>

## Dataset
- See [Kaggle dataset](https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1)


## Research Goals 

1. Train machine learning and deep learning models on 2007-2017 Lending club data to accurately predict loan defaults in the 2018 loan pool 
1. Leverage predictions from (1) to optimize portfolio of 2018 loans maximizing investment returns (IRR) for a given investor
1. Construct real-time machine learning prediction tools to allow users to leverage classification models for portfolio construction

- To achieve (2), our model’s predicted loan default probabilities for a given loan are combined with that loan’s term (36 or 60 months), monthly installment notional (the amount the debtor pays every month) and funded amount (the initial amount of the loan) in order to produce an expected internal rate of return (IRR) for that loan 
- Highest IRR-yielding loans are then picked from this filtered output to maximize IRR for a given portfolio notional

## Blog post + live presentation

- Link to [blog post](https://nycdatascience.com/blog/student-works/predicting-loan-defaults-using-machine-learning-classification-models/) 

- Link to [live presentation](https://www.youtube.com/watch?v=1U1pIe5-GZ0&ab_channel=NYCDataScienceAcademy)


## Presentation slides

Please refer to **/presentation/NYCDSA_Capstone_Presentation_vF.pdf** for presentation slides of presentation given on January 5th, 2021 to NYCDSA prospective students and alums regarding this project
