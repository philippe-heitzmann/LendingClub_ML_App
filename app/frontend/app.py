import logging
import pickle
import sys

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import requests

from constants.input_data import (
    CF_DIFF,
    ML_RESULTS,
    MODEL_OPTIONS,
    MODEL_ABBREVIATIONS,
    LINEAR_MODELS,
    TREE_MODELS,
    MARKS,
)
from constants.stylings import COLOR_BLUE, COLOR_GREEN
from constants.html_classnames import (
    CLASSNAME_ROW,
    CLASSNAME_COL_CENTERED,
    CLASSNAME_EIGHT_COLUMNS,
    CLASSNAME_DCC_CONTROL,
    CLASSNAME_PRETTY_CONTAINER,
    CLASSNAME_PRETTY_CONTAINER_4C,
    CLASSNAME_CONTROL_LABEL,
    CLASSNAME_INFO_TEXT,
)
from constants.html_styles import(STYLE_INLINE_BLOCK_4850, STYLE_UNDERLINE_11, STYLE_PADDING_BOTTOM_11PX)

logging.basicConfig(
    format="%(asctime)s, %(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)

# Set to communicate with Flask service named backend
# on default network
URL = "http://backend:5000/api/v1/predict"

# Create Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])
app.title = "Philippe Heitzmann Capstone Project"
app.config.suppress_callback_exceptions = True

# Create server
server = app.server

# Load data
df = pd.read_csv("/app/app/data/accepted_2007_to_2018Q4_500.csv", sep=",")
df.set_index("id", inplace=True)
df["emp_length"].fillna("5 years", inplace=True)
df1 = pd.DataFrame(df.groupby(["emp_length"])["annual_inc"].mean()).reset_index()

# Loading groupby addr_state, emp_length, data
with open("/app/app/data/eda1.pickle", "rb") as edaFile:
    eda1 = pickle.load(edaFile)

# Loading groupby addr_state, Year data
with open("/app/app/data/eda2.pickle", "rb") as edaFile:
    eda2 = pickle.load(edaFile)

# Loading groupby addr_state, grade data
with open("/app/app/data/eda3.pickle", "rb") as edaFile:
    eda3 = pickle.load(edaFile)

# Loading groupby addr_state, grade, int data
with open("/app/app/data/eda4.pickle", "rb") as edaFile:
    eda4 = pickle.load(edaFile)

# Loading groupby addr_state, grade, int data
with open("/app/app/data/eda5.pickle", "rb") as edaFile:
    eda5 = pickle.load(edaFile)

# Loading other input data to be visualized
with open("/app/app/data/x_train_lg1.pickle", "rb") as inputDataFile:
    x_train_lg1 = pickle.load(inputDataFile)

with open("/app/app/data/test_data2.pickle", "rb") as inputDataFile:
    test_data2 = pickle.load(inputDataFile)

# Create other input data to be visualized
data_inc = x_train_lg1["annual_inc"]
test_data2["Predicted_IRR"] = np.round(test_data2["Predicted_IRR"], 4)
test_data2["True_IRR"] = np.round(test_data2["True_IRR"], 4)

# Create options for ID's
ids_options = [{"label": str(x), "value": x} for x in list(x_train_lg1.index)]


@app.callback(
    Output("predictionText", "children"),
    [Input("id_options", "value"), Input("model_options", "value")],
)
def update_prediction_text(id1, model):
    """Update prediction text (default or no default) for a given model prediction on a given input observation (denoted by ID)"""
    # Create data to be passed to model
    test1 = [list(np.array(x_train_lg1.loc[id1, :]))]
    params = {"query": test1, "model": model}
    response = requests.get(URL, json=params)
    return response.json()["prediction"]


@app.callback(
    Output("probabilityText", "children"),
    [Input("id_options", "value"), Input("model_options", "value")],
)
def update_probability_text(id1, model):
    """Update predicted probability text for a given model prediction on a given input observation (denoted by ID)"""
    logging.info(f"ID is {str(id1)} & Selected model is {str(model)}")
    test1 = [list(np.array(x_train_lg1.loc[id1, :]))]
    params = {"query": test1, "model": model}
    response = requests.get(URL, json=params)
    if response.json()["prediction"] == "No Default":
        return np.round(response.json()["confidence"][1], 2)
    return np.round(response.json()["confidence"][0], 2)


@app.callback(Output("ml_descriptions", "children"), [Input("model_options", "value")])
def update_ml_description_text(model: str) -> str:
    """Update model description based on model param passed"""
    if model == "QDA":
        return "QDA is a supervised dimensionality reduction algorithm that maximizes separability of the data into K different classes. The difference between QDA and LDA is that QDA does not follow the equal covariance amongst K classes assumption, which therefore does not result in the cancellation of the quadratic term in the discriminant function, hence the name Quadratic Discriminant Analysis. Similar to LDA, QDA model uses the discriminant rule to divide the data space into 2 disjoint regions that represent each class, Default or Non-Default. With these regions, classification by discriminant analysis means that we allocate x to class j if x is in region j. Following the Bayesian rule, we use the discriminant function to classify the data x to class j if it has the highest likelihood to belong to that class amongst all K classes for i = 1,…,K. From this, QDA can therefore create a decision boundary separating any two classes, k and l, derived from the set of x where two discriminant functions have the same value, and output classification predictions based on this boundary."
    elif model == "LDA":
        return "LDA is a supervised dimensionality reduction algorithm that maximizes separability of the data into K different classes, assuming equal covariance among the K classes. Applying LDA to our Lending Club data, the model uses the discriminant rule to divide the data space into 2 disjoint regions that represent each class, Default or Non-Default. With these regions, classification by discriminant analysis means that we allocate x to class j if x has the highest likelihood of being in region j. Regions are defined by leveraging the Bayesian rule to use the discriminant function to classify the data x to class j if it has the highest likelihood among all K classes for i = 1,…,K. As the equal covariance assumption cancels out the quadratic term in the discriminant function, the resulting equation is a linear function, hence the name Linear Discriminant Analysis. From this, LDA can therefore create a decision boundary separating any two classes, k and l, derived from the set of x where two discriminant functions have the same value, and output classification predictions based on this boundary."
    elif model == "LOGIT":
        return "Logistic Regression works by combining inputs linearly using coefficient values to predict a binary output value. As maximizing the log-likelihood equation has no closed form solution, coefficients are estimated through iteratively reweighted least squares. While Logistic Regression is a linear model, its predictions are calculated using the logistic function, which inputs values of the coefficients and of the independent variables into the logistic curve equation to output a prediction for the default class. Logistic Regression models this probability of the default class and outputs a prediction based on a provided cutoff value."
    elif model == "GBC":
        return "The Gradient Boosting Classifier model is a composite model that combines the efforts of multiple weak models to create a strong model, with each additional weak model being trained to reduce the mean squared error (MSE) of the overall model. GBM therefore works by sequenially optimizing the mean squared error (MSE), also called the L2 loss or cost, of each previous weak learner."
    elif model == "RF":
        return "Random Forests is an ensemble tree-based learning algorithm that works by creating a set of decision trees predicting outputs from randomly selected subsets of the training set. The Random Forests model then aggregates the votes from different decision trees to decide the final class of the test object. Random Forets then takes the average of these predictions, or the majority vote over all trees, to output a final singular prediction."
    return ""


# Create Dash layout
layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
    title="Satellite Overview",
)

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H1(
                                    [
                                        "Capstone Project: Predicting Loan Default Rates using ML",
                                    ],
                                ),
                            ],
                            className=CLASSNAME_COL_CENTERED,
                            style={"padding-left": "65px"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H5(
                                            "by Philippe Heitzmann",
                                            style={
                                                "color": "#5dbcd2",
                                                "font-style": "italic",
                                                "font-weight": "bold",
                                                "opacity": "0.8",
                                            },
                                        )
                                    ],
                                    style={"padding-left": "80px"},
                                ),
                                html.Div(
                                    [
                                        html.Img(
                                            src="https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTA3rgASWzdVLcpKDLzet7I-7a2FUGVtSRSqQ&usqp=CAU",
                                            style={"width": "35%"},
                                        )
                                    ],
                                    style={"padding-left": "565px"},
                                ),
                            ],
                            className=CLASSNAME_INFO_TEXT,
                        ),
                    ],
                    className=CLASSNAME_EIGHT_COLUMNS,
                    style={
                        "marginBottom": 20,
                        "marginTop": 0,
                        "width": "96%",
                        "padding-left": "5%",
                        "padding-right": "0%",
                        "border": "4px black solid",
                        "backgroundColor": "#FFF",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            [html.H4(["Abstract",],),],
                            style={"padding-top": "8px", "padding-left": "0.75%"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.P(
                                            [
                                                """In this Capstone Project, eight different linear and tree-ensembling machine learning models and a MultiLayer Perceptron (“MLP”) Neural Net are constructed to predict loan defaults on the Lending Club platform. The predictions from our best performing model are then used to allocate capital to loans predicted as ‘good’ and to deny investment for loans predicted as ‘bad’. This optimized portfolio produces a net realized IRR of 7.40% for 36-month loans and 10.63% for 60-month loans, assuming 0% loan recoveries in the case of default, versus Lending Club's""",
                                                html.A(
                                                    " 2018 rates of return ",
                                                    href="https://www.lendingclub.com/info/demand-and-credit-profile.action",
                                                ),
                                                """of 6.30% for 36-month loans and 8.11% for 60-month loans, which are further inclusive of actual loan recoveries post-default. These results further compare favorably to a baseline model predicting all loans as ‘good’ loans, which yields 5.89% IRR for 36-month loans and 9.67% IRR for 60-month loans. A two-sample t-test of our model portfolio against the baseline portfolio further shows these results are statistically significant to the 1% level, and that our model produces 1.51% and 0.99% of alpha, or active return on investment, for 36-month and 60-month loans respectively versus the baseline model portfolio. """,
                                            ]
                                        )
                                    ],
                                    style={"padding-left": "2.5%"},
                                ),
                            ],
                            className=CLASSNAME_INFO_TEXT,
                        ),
                    ],
                    className=CLASSNAME_EIGHT_COLUMNS,
                    style={
                        "marginBottom": 10,
                        "marginTop": 10,
                        "width": "96%",
                        "padding-left": "0%",
                        "padding-right": "2.5%",
                        "border": "2px black solid",
                        "backgroundColor": "#FFF",
                    },
                ),
                html.Div(
                    [
                        html.H4(
                            "Background",
                            style=STYLE_UNDERLINE_11,
                        ),
                        dbc.Row(
                            [
                                html.P(
                                    [
                                        "Nowadays consumers can invest in consumer loans through peer-to-peer financing platforms such as ",
                                        html.A(
                                            "Lending Club", href="www.lendingclub.com/"
                                        ),
                                        ". Lending Club enables investors to browse consumer loan applications containing the applicant’s credit history, loan details, employment status, and other self-reported personal information, in order to make determinations as to which loans to fund.",
                                        html.Br(),
                                        html.Br(),
                                        "The Lending Club dataset contains 2,260,701 individual loan application observations collected between 2007-2018. There are 151 quantitative and qualitative features in the dataset capturing information on the loan amount, loan interest rate, loan status (i.e. whether the loan is current or in default), loan tenor (either 36 or 60 months), applicant FICO credit score, applicant employment history, and other personal information. Notably, the dependent variable, which captures loan defaults as 0's and loan non-defaults as 1's, is heavily imbalanced, with non-defaults outnumbering defaults 6.7:1 in this dataset. This class imbalance will be important later on when training our machine learning models.",
                                        html.Br(),
                                        html.Br(),
                                        "Click",
                                        html.A(
                                            " here ",
                                            href="https://resources.lendingclub.com/LCDataDictionary.xlsx",
                                        ),
                                        "to access the full Lending Club data dictionary with detailed descriptions of each variable.",
                                    ],
                                    style={
                                        "padding-left": "1.8%",
                                        "display": "inline-block",
                                        "width": "45.5%",
                                    },
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(
                                            style={
                                                "height": "300px",
                                                "marginBottom": 10,
                                            },
                                            figure={
                                                "data": [
                                                    {
                                                        "type": "bar",
                                                        "x": [
                                                            "1Q13",
                                                            "2Q13",
                                                            "3Q13",
                                                            "4Q13",
                                                            "1Q14",
                                                            "2Q14",
                                                            "3Q14",
                                                            "4Q14",
                                                            "1Q15",
                                                            "2Q15",
                                                            "3Q15",
                                                            "4Q15",
                                                            "1Q16",
                                                            "2Q16",
                                                            "3Q16",
                                                            "4Q16",
                                                            "1Q17",
                                                            "2Q17",
                                                            "3Q17",
                                                            "4Q17",
                                                            "1Q18",
                                                            "2Q18",
                                                            "3Q18",
                                                            "4Q18",
                                                            "1Q19",
                                                            "2Q19",
                                                            "3Q19",
                                                            "4Q19",
                                                            "1Q20",
                                                            "2Q20",
                                                            "3Q20",
                                                        ],
                                                        "y": [
                                                            1.5,
                                                            2,
                                                            2.5,
                                                            3.2,
                                                            4,
                                                            5,
                                                            6.2,
                                                            7.6,
                                                            9.3,
                                                            11.1,
                                                            13.4,
                                                            16,
                                                            18.7,
                                                            20.7,
                                                            22.7,
                                                            24.6,
                                                            26.6,
                                                            28.9,
                                                            31.2,
                                                            33.6,
                                                            35.9,
                                                            38.8,
                                                            41.6,
                                                            44.5,
                                                            47.2,
                                                            50.4,
                                                            53.7,
                                                            56.8,
                                                            59.3,
                                                            59.6,
                                                            60.2,
                                                        ],
                                                    }
                                                ],
                                                "layout": {
                                                    "autosize": True,
                                                    "automargin": True,
                                                    "margin": dict(
                                                        l=40, r=30, b=40, t=40
                                                    ),
                                                    "hovermode": "closest",
                                                    "plot_bgcolor": "#F9F9F9",
                                                    "paper_bgcolor": "#F9F9F9",
                                                    "title": "Lending Club Platform Issuance Growth 2013-20",
                                                    "yaxis": {
                                                        "title": "Loan Issuance Volume ($B)"
                                                    },
                                                    "paper_bgcolor": "#F9F9F9",
                                                    "plot_bgcolor": "#F9F9F9",
                                                },
                                            },
                                        ),
                                    ],
                                    style=STYLE_INLINE_BLOCK_4850,
                                    className=CLASSNAME_PRETTY_CONTAINER,
                                ),
                            ]
                        ),
                        html.H4(
                            "Research Goals",
                            style=STYLE_UNDERLINE_11,
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Markdown(
                                            """The goal of this project is to predict default probabilities of 2018 loans in the Lending Club portfolio by training our models on pre-2018 loan data in order to uncover the best investment opportunity set for an investor looking to maximize his or her returns on the 2018 loan set. To achieve this, our model’s predicted loan default probabilities for a given loan are combined with that loan’s term (36 or 60 months), monthly installment notional (the amount the debtor pays every month) and funded amount (the initial amount of the loan) in order to produce an expected internal rate of return (IRR) for the loan. Using these predictions, our model then allocates capital to all loans it predicted as good in the 2018 test dataset and does not fund any loans it predicted as bad in order to arrive at an IRR-optimized portfolio.""",
                                            style={
                                                "padding-left": "0.8%",
                                                "padding-bottom": "15px",
                                            },
                                        ),
                                        dcc.Markdown(
                                            """ For this project I will be using the Python libraries for data manipulation (**pandas, numpy**), regular expressions (**re**), data visualization (**plotly, matplotlib, seaborn**), machine learning (**scikit-learn, catboost**), deep learning (**keras, tensorflow**), API implementation (**flask**), web frameworks (**dash**) and statistics (**scipy stats, itertools and statsmodels**). **Amazon Web Services’ (AWS) Elastic Computing (EC2)** platform was also used to train our neural network model in the cloud.""",
                                            style={
                                                "padding-left": "0.8%",
                                                "padding-bottom": "15px",
                                            },
                                        ),
                                    ]
                                )
                            ]
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.H4(
                            "Exploratory Data Analysis",
                            style=STYLE_UNDERLINE_11,
                        )
                    ],
                ),
            ],
            id="header",
            className=CLASSNAME_ROW,
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            "Filter by loan application date",
                            className=CLASSNAME_CONTROL_LABEL,
                        ),
                        dcc.RangeSlider(
                            id="year_slider",
                            min=2014,
                            max=2018,
                            value=[2014, 2016],
                            marks=MARKS,
                            className=CLASSNAME_DCC_CONTROL,
                        ),
                        dcc.Checklist(value=[], className=CLASSNAME_DCC_CONTROL),
                    ],
                    className=CLASSNAME_PRETTY_CONTAINER_4C,
                ),
            ],
            className=CLASSNAME_ROW,
        ),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="main_graph")],
                    style=STYLE_INLINE_BLOCK_4850,
                    className=CLASSNAME_PRETTY_CONTAINER,
                ),
                html.Div(
                    [dcc.Graph(id="individual_graph")],
                    style=STYLE_INLINE_BLOCK_4850,
                    className=CLASSNAME_PRETTY_CONTAINER,
                ),
            ],
            className=CLASSNAME_ROW,
        ),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="reg_graph1")],
                    style=STYLE_INLINE_BLOCK_4850,
                    className=CLASSNAME_PRETTY_CONTAINER,
                ),
                html.Div(
                    [dcc.Graph(id="reg_graph2")],
                    style=STYLE_INLINE_BLOCK_4850,
                    className=CLASSNAME_PRETTY_CONTAINER,
                ),
            ],
            className=CLASSNAME_ROW,
        ),
        html.Div(
            [
                dcc.Markdown(
                    """_**Top Left Graph - **_ At first glance, this choropleth plot shows that certain states exhibit significantly greater average default rates than others and that a loan applicant’s state may be an important predictor of loan default. Interestingly, there appears to be regional clustering in default patterns, as southern states such as Oklahoma, Arkansas, Louisiana, Mississippi, and Alabama exhibit higher average default rates (>21%) while northwestern states such as Oregon, Washington, Idaho, and Wyoming exhibit lower average default rates (<17%) versus the average 14.97% default rate across all states. Interestingly however, despite these apparent differences in loan default rates by state, interest rates charged by loan grade by state are quasi-identical, likely due to federal and state anti-discriminatory and usury regulations."""
                ),
                dcc.Markdown(
                    """_**Top Right Graph - **_ While one might reason these default rate differences by state may be due to different loan quality distributions by state, the adjoining scatterplot shows this not to be the case, as all states display a similar left-skewed distribution in loan grade quality. Indeed, all states exhibit 15-20 percent of loans graded as ‘A’, 25-30 percent graded as ‘B’, 25-33 percent graded as ‘C’, 15-20 percent graded as ‘D’, 5-10 percent graded as ‘E’, 0-5 percent graded as ‘F’, and 0-1 percent graded as ‘G’. Based on these similar loan quality distributions between states, we would expect state variables to not be a strong predictor of loan defaults, though other unobserved macro variables operating at the state-level may also impact loan defaults rates. Further investigation is needed to ascertain this."""
                ),
                dcc.Markdown(
                    """_**Bottom Left Graph - **_ I was interested in visualizing how repayment rate trended with FICO scores in our dataset. As expected, FICO scores exhibit a strong positive correlation with repayment rate, with an added logarithmic flattening at the tail end of the curve as repayment rate essentially plateaus past >700 FICO scores. On the other hand, interest rates display an expected strong negative correlation with FICO scores, with added negative convexity. These two observations taken together offer a potentially interesting insight that applicants with FICO scores in the 700-750 range, having a similar repayment risk profile than applicants in the 800-850 range while being charged 3% higher interest rates on average, may offer an investor the opportunity to unlock added alpha by offering more loans to applicants in this 700-750 FICO score subgroup.""",
                ),
                dcc.Markdown(
                    """_**Bottom Right Graph - **_ I was also interested in looking at default rates by employment length, as my original hypothesis was that applicants with lesser amounts of work experience (bars colored in green) would exhibit higher default rates on average. As shown in these bar graphs, this is not the case across almost all states, as applicants with <5 years of work experience actually display higher loan repayment rates than applicants with >5 years of experience on average. Work experience may therefore not be as predictive of a variable as originally thought, though additional investigation would be needed to confirm this.""",
                ),
            ]
        ),
        html.Div(
            [
                html.H4(
                    "Data Preprocessing",
                    style={"text-decoration": "underline", "marginLeft": 0},
                ),
                dcc.Markdown(
                    """The first step in this data preprocessing process was to drop variables with >50% missing data, which included 42 variables. Variables that would introduce data leakage into our models, such as *collection_recovery_fee*, which captures the collection recovery fee earned by Lending Club once a loan has been charged off, were dropped as well."""
                ),
                dcc.Markdown(
                    """Next, other variables that included only one feature, such as *policy_code*, were dropped, while features like *zip_code*, which could potentially cause our models to overfit noise, were dropped as well. Based on a careful read of Lending Club’s provided data dictionary, certain other numeric features were imputed using the mean, such as *annual_inc* (variable capturing self-reported annual income provided by the borrower during registration), *bc_util* (variable capturing ratio of total current balance to high credit/credit limit for all bankcard accounts), or *delinq_2yrs* (variable capturing number of 30+ days past-due incidences of delinquency in the borrower\'s credit file for the past 2 years), while others were imputed to zero depending on what NaN values were interpreted as most likely representing."""
                ),
                html.P(
                    "Similarly, missing values in certain other numeric and categorical variables were imputed with the highest-frequency number / class for those variables, as appropriate. Post this data preprocessing, our dataset contained 94 features versus the 150 we started with."
                ),
                html.P(
                    "Independent variables were scaled only in cases where the models being trained on this data required feature normalization in order to ensure better convergence. Feature normalization was therefore used to scale the training data used in our multilayer perceptron while our linear and tree ensembling models were trained using unscaled data. As the y target variable was already in the set [0,1] once our qualitative loan default description were mapped to 0 and 1 values, feature normalization was not required for our target variable.",
                    style=STYLE_PADDING_BOTTOM_11PX,
                ),
            ],
        ),
        html.Div(
            [
                html.H4(
                    "Feature Engineering",
                    style={"text-decoration": "underline", "marginLeft": 0},
                ),
                dcc.Markdown(
                    """In order to calculate Internal Rate of Return values for each of our loans, a variable *cashflows* was created to model cash flow schedules for our loans based on loan funded amount, loan term, loan funded date, loan interest rate, and loan default probability outputted by our CatBoost Classifier model for that loan. Based on the cashflow variable, another xirr variable was created based on calculating the net present value of this series of cashflows and optimizing for the IRR’s of these loans using the scipy **optimize.brentq()** function. These IRR values will later be used to calculate the final IRR of our selected portfolio subset of loans based on our machine learning predictions. """
                ),
                dcc.Markdown(
                    """Similarly, in order to later optimize our loan prediction cutoff values, a variable _False_Negative_Cost_ was created to quantify the opportunity cost of predicting loan x as bad when it was in fact a good loan. This variable would later be used to further compare our different models as well. """
                ),
                html.P(
                    "Lastly, as date variables in the Lending Club dataset were passed in string format, different date variables were created by transforming these variables to datetime format and extracting specific date attributes such as loan issuance month and year, which were passed as standalone features in the final dataset used to train our models. ",
                    style=STYLE_PADDING_BOTTOM_11PX,
                ),
            ],
        ),
        html.Div(
            [
                html.H4(
                    "Feature Selection",
                    style={"text-decoration": "underline", "marginLeft": 0},
                ),
                dcc.Markdown(
                    """To avoid multicollinearity issues in our data, both numeric and categorial variables exhibiting high degrees of multicollinearity >0.85 were dropped from the dataset. For categorical variables, this resulted in *sub_grade*, which had a 1.00 correation with *grade*, being dropped."""
                ),
                dcc.Markdown(
                    """Similarly, for numeric variables, this method flagged (*funded_amnt, installment*), (*fico_range_low, fico_range_high*), (*open_acc, num_sats*), among others, as highly correlated pairs, of which the first variable of each pair was dropped."""
                ),
                dcc.Markdown(
                    """For linear models, **Recursive Feature Elimination** (“RFE”) was employed to select the most important half of features in the dataset. The precision, recall and accuracy scores of these RFE-selected models were then compared to those of the linear models with no feature selection. Additional features were added / dropped from that point as appropriate. Conversely, RFE was not employed for tree-ensembling models that implicitly perform feature selection when outputting predictions. """,
                    style=STYLE_PADDING_BOTTOM_11PX,
                ),
            ],
        ),
        html.Div(
            [
                html.H4(
                    "Machine Learning Results",
                    style={"text-decoration": "underline", "marginLeft": 0},
                ),
                dcc.Markdown(
                    """The eight models I was interested in testing for this problem were **Logistic Regression**, **Linear Discriminant Analysis**, **Quadratic Discriminant Analysis**, **Multinomial Naïve Bayes**, **Gaussian Naïve Bayes**, **Gradient Boosting Classifier**, **Random Forest Classifier**, and **CatBoost Classifier**."""
                ),
                dcc.Markdown(
                    """I also built a **Multilayer Perceptron** (MLP) neural network with eight Dense layers and five dropout layers using AWS Elastic Compute in order to investigate how a neural net would perform with this tabular data with a large amount of features."""
                ),
                html.P(
                    "ROC_AUC score was used to measure the performance of each model as ROC_AUC offered a holistic view of the impact of both false negatives and false positives in the data in order to get a sense for how imperfect specificity and sensitivity would later translate into lost interest income and loan principal loss respectively.",
                ),
                html.P(
                    "ROC_AUC scores were computed on predictions for both the non-2018 test data and the 2018 test data in order to see if any of our models were overfitting the non-2018 training data. The majority of models reported higher ROC_AUC scores on the non-2018 target data than on the 2018 target data in the range of 1-6 percentage points, which is likely the result of slight overfitting that should nonetheless not be a cause for concern given the relatively small AUC score deltas. Notably, however, our Quadratic Discriminant Analysis model exhibits some slight underfitting behavior with ROC_AUC being 2.5 percentage points higher when predicting 2018 data, and our Linear Discriminant Analysis and Random Forest Classifier models exhibit overfitting behavior of 7pts+ on the non-2018 data vs the 2018 data.",  # To address these overfitting issues in our LDA model, future iterations
                    style=STYLE_PADDING_BOTTOM_11PX,
                ),
            ],
        ),
        html.Div(
            [
                html.H5(
                    "ROC_AUC Model Scores",
                    style={"text-decoration": "underline", "marginLeft": 0},
                ),
                html.Div(
                    [
                        dash_table.DataTable(
                            id="datatable-interactivity",
                            columns=[
                                {
                                    "name": i,
                                    "id": i,
                                    "deletable": True,
                                    "selectable": True,
                                }
                                for i in ML_RESULTS.columns
                            ],
                            data=ML_RESULTS.to_dict("records"),
                            editable=True,
                            filter_action="native",
                            sort_action="native",
                            sort_mode="multi",
                            column_selectable="single",
                            row_selectable="multi",
                            row_deletable=True,
                            selected_columns=[],
                            selected_rows=[],
                            page_action="native",
                            page_current=0,
                            page_size=10,
                            style_cell={"fontSize": 15, "font-family": "sans-serif"},
                        ),
                    ],
                    style={"padding-left": "15px", "padding-right": "10px"},
                ),
                html.Div(
                    [
                        html.Div(
                            id="datatable-interactivity-container",
                            style={
                                "display": "inline-block",
                                "width": "50%",
                                "padding-top": "30px",
                                "padding-left": "5px",
                            },
                            className=CLASSNAME_COL_CENTERED,
                        ),
                        html.Div(
                            id="datatable-interactivity-container2",
                            style={
                                "display": "inline-block",
                                "width": "50%",
                                "padding-top": "30px",
                            },
                            className=CLASSNAME_COL_CENTERED,
                        ),
                    ],
                ),
            ]
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            "As can be seen from the above interactive table, the models with the highest ROC_AUC scores are Catboost Classifier and MLP Neural Net. One surprising result was our Random Forest Classifier not reaching the 0.80 ROC_AUC benchmark achieved by the other tree-based models, namely Gradient Boosting Classifier and CatBoost Classifier. These ROC_AUC scores by model are visualized in the adjoining bar graphs for both the non-2018 test data and 2018 test data. Of note, the Dash app also allows the user to interactively sort the models by ROC_AUC score by clicking the arrows at the top of each column in the datatable, which automatically triggers a Dash callback that reactively updates the bar graphs below. The user can also select different models in the datatable, which will also reactively highlight in a different color the corresponding columns in the bar graphs.",
                            style={
                                "padding-bottom": "15px",
                                "padding-top": "15px",
                                "padding-left": "15px",
                            },
                        )
                    ],
                ),
                html.H5(
                    "Model Confusion Matrices",
                    style={"text-decoration": "underline", "marginLeft": 15},
                ),
            ],
            className=CLASSNAME_ROW,
        ),
        html.Div(
            [
                dcc.Markdown(
                    """  The confusion matrices below further provide additional detail into the prediction strengths and weaknesses of each model on the non-2018 data. """
                )
            ]
        ),
        html.Div(
            [
                dcc.Tabs(
                    [
                        dcc.Tab(
                            label="Linear Models",
                            children=[
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.P(
                                                    "Select Linear Model #1",
                                                    className=CLASSNAME_CONTROL_LABEL,
                                                ),
                                                dcc.Dropdown(
                                                    id="linear_models",
                                                    options=LINEAR_MODELS,
                                                    multi=False,
                                                    value="LDA",
                                                    className=CLASSNAME_DCC_CONTROL,
                                                ),
                                            ],
                                            style={
                                                "display": "inline-block",
                                                "width": "48.5%",
                                            },
                                            className=CLASSNAME_PRETTY_CONTAINER,
                                        ),
                                        html.Div(
                                            [
                                                html.P(
                                                    "Select Linear Model #2",
                                                    # style = {'marginLeft':50},
                                                    className=CLASSNAME_CONTROL_LABEL,
                                                ),
                                                dcc.Dropdown(
                                                    id="linear_models2",
                                                    options=LINEAR_MODELS,
                                                    multi=False,
                                                    value="GNB",
                                                    className=CLASSNAME_DCC_CONTROL,
                                                ),
                                            ],
                                            style={
                                                "display": "inline-block",
                                                "width": "48.5%",
                                            },
                                            className=CLASSNAME_PRETTY_CONTAINER,
                                        ),
                                    ],
                                    className=CLASSNAME_INFO_TEXT,
                                ),
                                html.Div(
                                    [dcc.Graph(id="confusion_matrix_linear"),],
                                    style=STYLE_INLINE_BLOCK_4850,
                                    className=CLASSNAME_PRETTY_CONTAINER,
                                ),
                                html.Div(
                                    [dcc.Graph(id="confusion_matrix_linear2"),],
                                    style=STYLE_INLINE_BLOCK_4850,
                                    className=CLASSNAME_PRETTY_CONTAINER,
                                ),
                            ],
                        ),
                        # Tab 2
                        dcc.Tab(
                            label="Tree-based Models",
                            children=[
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.P(
                                                    "Select Tree Model #1",
                                                    className=CLASSNAME_CONTROL_LABEL,
                                                ),
                                                dcc.Dropdown(
                                                    id="tree_models",
                                                    options=TREE_MODELS,
                                                    multi=False,
                                                    value="GBC",
                                                    className=CLASSNAME_DCC_CONTROL,
                                                ),
                                            ],
                                            style={
                                                "display": "inline-block",
                                                "width": "48.5%",
                                            },
                                            className=CLASSNAME_PRETTY_CONTAINER,
                                        ),
                                        html.Div(
                                            [
                                                html.P(
                                                    "Select Tree Model #2",
                                                    className=CLASSNAME_CONTROL_LABEL,
                                                ),
                                                dcc.Dropdown(
                                                    id="tree_models2",
                                                    options=TREE_MODELS,
                                                    multi=False,
                                                    value="RF",
                                                    className=CLASSNAME_DCC_CONTROL,
                                                ),
                                            ],
                                            style={
                                                "display": "inline-block",
                                                "width": "48.5%",
                                            },
                                            className=CLASSNAME_PRETTY_CONTAINER,
                                        ),
                                    ],
                                    className=CLASSNAME_INFO_TEXT,
                                ),
                                html.Div(
                                    [dcc.Graph(id="confusion_matrix_tree"),],
                                    style=STYLE_INLINE_BLOCK_4850,
                                    className=CLASSNAME_PRETTY_CONTAINER,
                                ),
                                html.Div(
                                    [dcc.Graph(id="confusion_matrix_tree2"),],
                                    style=STYLE_INLINE_BLOCK_4850,
                                    className=CLASSNAME_PRETTY_CONTAINER,
                                ),
                            ],
                        ),
                        # Tab 3
                        dcc.Tab(
                            label="Multilayer Perceptron Neural Net",
                            children=[
                                html.Div(
                                    [dcc.Graph(id="confusion_matrix_mlp"),],
                                    style=STYLE_INLINE_BLOCK_4850,
                                    className=CLASSNAME_PRETTY_CONTAINER,
                                ),
                            ],
                        ),
                    ]
                ),
            ],
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H5(
                            "Tuning Cutoff Threshold in Catboost Classifier",
                            style={"text-decoration": "underline", "marginLeft": 0},
                        ),
                    ]
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            """In order to optimize the prediction cutoff values in our CatBoost Classifier model, I created a scatter plot of a new variable *cf_diff* capturing the cost of both types of mis-classifications for different prediction cutoff values. If a loan was classified by our model as good when it was in fact bad, the principal loss of that loan would be added to *cf_diff*. Conversely, if a loan was classified as bad when it was in fact good, the sum of the interest payments we missed out on times the default probability of that loan would be added to *cf_diff*. Summing the *cf_diff* variable for different prediction cutoff values produced the following graph, showing that a cutoff value of 0.73737 minimized the *cf_diff* loss. This high cutoff value makes intuitive sense as the principal loss of defaulted loans (=0) that were predicted as good (=1) should be expected to outweigh the lost interest income of non-defaulted loans (=1) that were predicted as bad (=0) in the aggregate. The user can hover over each point in this graph to view the total dollar cost for different prediction cutoff points.""",
                            style={"padding-top": "15px"},
                        ),
                        html.Div(
                            [
                                dcc.Graph(
                                    style={"height": "300px", "marginBottom": 10,},
                                    id="cutoff_graph",
                                ),
                            ],
                            style={
                                "display": "inline-block",
                                "width": "48.5%",
                                "padding-top": "15px",
                            },
                            className=CLASSNAME_PRETTY_CONTAINER,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.H5(
                            "Portfolio Results",
                            style={"text-decoration": "underline", "marginLeft": 0},
                        ),
                        dcc.Markdown(
                            """The predictions from our Catboost were used to allocate capital to loans predicted as ‘good’ and to deny investment for loans predicted as ‘bad’. This optimized portfolio produces a net realized IRR of **7.40%** for 36-month loans and **10.63%** for 60-month loans, assuming 0% loan recoveries in the case of default, versus Lending Club\'s self-reported 2018 rates of return of **6.30%** for 36-month loans and **8.11%** for 60-month loans, which are further inclusive of actual loan recoveries post-default. These results further compare favorably to a baseline model predicting all loans as ‘good’ loans, which yields **5.89%** IRR for 36-month loans and **9.67%** IRR for 60-month loans. A two-sample t-test of our model portfolio against the baseline portfolio further shows these results are statistically significant to the 1% level, and that our model produces **1.51%** and **0.99%** of alpha for 36-month and 60-month loans respectively versus the baseline model portfolio.""",
                            style={"marginLeft": 0},
                        ),
                    ]
                ),
            ],
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H4(
                            "Real-time Machine Learning Predictions using Flask",
                            style={"text-decoration": "underline", "marginLeft": 16},
                        ),
                        html.P(
                            "In order to output real-time loan default predictions for each of the models, I created a Flask app that takes pickle files of each of our models to output predictions for different subsets of our data. These different subsets are passed through as queries through the Flask API, which returns final predictions and predicted probabilities for each of our models.",
                            style={"marginLeft": 7},
                            className=CLASSNAME_CONTROL_LABEL,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.P(
                            "Select a Loan Applicant ID",
                            style={"marginLeft": 15},
                            className=CLASSNAME_CONTROL_LABEL,
                        ),
                        dcc.Dropdown(
                            id="id_options",
                            options=ids_options,
                            multi=False,
                            value=1041258,
                            className=CLASSNAME_DCC_CONTROL,
                            style={"marginLeft": 5},
                        ),
                        dcc.Checklist(
                            id="lock_selector",
                            value=[],
                            className=CLASSNAME_DCC_CONTROL,
                        ),
                        html.P(
                            "Select a Model to view",
                            style={"marginLeft": 15},
                            className=CLASSNAME_CONTROL_LABEL,
                        ),
                        dcc.Dropdown(
                            id="model_options",
                            options=MODEL_OPTIONS,
                            multi=False,
                            value="GBC",
                            className=CLASSNAME_DCC_CONTROL,
                            style={"marginLeft": 5},
                        ),
                    ],
                ),
                html.Div(
                    [
                        html.P("Loan Default Prediction"),
                        html.H6(id="predictionText", className=CLASSNAME_INFO_TEXT),
                    ],
                    className=CLASSNAME_PRETTY_CONTAINER_4C,
                ),
                html.Div(
                    [
                        html.P("Prediction Confidence (/1.00)"),
                        html.H6(id="probabilityText", className=CLASSNAME_INFO_TEXT),
                    ],
                    className=CLASSNAME_PRETTY_CONTAINER_4C,
                ),
                html.Div(
                    [
                        html.H5(
                            "Model Description",
                            style={"text-decoration": "underline", "marginLeft": 13.5},
                        ),
                        html.P(id="ml_descriptions", style={"marginLeft": 13.5}),
                    ],
                ),
            ],
            className=CLASSNAME_ROW,
        ),
        html.Div(
            [
                html.H5(
                    "Applicant Overview",
                    style={"text-decoration": "underline", "marginLeft": 0},
                )
            ],
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.P("Annual Income of Loan Applicant"),
                        html.H6(id="annual_inc", className=CLASSNAME_INFO_TEXT),
                    ],
                    className="pretty_container two columns",
                ),
                html.Div(
                    [
                        html.P("FICO Score of the Loan Applicant"),
                        html.H6(id="fico_score_text", className=CLASSNAME_INFO_TEXT),
                    ],
                    className="pretty_container two columns",
                ),
                html.Div(
                    [
                        html.P("Total Accounts of Loan Applicant"),
                        html.H6(id="acc_text", className=CLASSNAME_INFO_TEXT),
                    ],
                    className="pretty_container two columns",
                ),
                html.Div(
                    [
                        html.P(
                            "Number of Personal Finance Inquiries of Loan Applicant"
                        ),
                        html.H6(id="inq_text", className=CLASSNAME_INFO_TEXT),
                    ],
                    className="pretty_container two columns",
                ),
                html.Div(
                    [
                        html.P("Employment Length of Loan Applicant"),
                        html.H6(id="emp_length_text", className=CLASSNAME_INFO_TEXT),
                    ],
                    className="pretty_container two columns",
                ),
            ],
            className=CLASSNAME_ROW,
        ),
        html.Div(
            [
                html.H4(
                    "Portfolio Optimization & Customization",
                    style={"text-decoration": "underline", "marginLeft": 0},
                ),
                html.P(
                    "The below interactive datatable allows the user to optimize a portfolio as measured by IRR by specifying certain portfolio characteristics, such as exposure to a certain state, to a certain loan purpose class, or a certain loan applicant homeownership class. The datatable automatically sorts these loans by our model's predicted IRR to return the optimal loan subset based on the specified Home_Ownership_Status, Loan_Purpose and State parameters. This allows the user to quickly return a loan portfolio tailored to their investment criteria.",
                    style={"marginLeft": 0},
                ),
            ],
        ),
        html.Div(
            [
                html.Div(
                    [
                        dash_table.DataTable(
                            columns=[
                                {
                                    "name": i,
                                    "id": i,
                                    "deletable": True,
                                    "selectable": True,
                                }
                                for i in test_data2.columns
                            ],
                            data=test_data2.to_dict("records"),
                            editable=True,
                            filter_action="native",
                            sort_action="native",
                            sort_mode="multi",
                            column_selectable="single",
                            row_selectable="multi",
                            row_deletable=True,
                            selected_columns=[],
                            selected_rows=[],
                            page_action="native",
                            page_current=0,
                            page_size=10,
                            style_cell={"fontSize": 15, "font-family": "sans-serif"},
                        ),
                    ],
                    style={"padding-left": "35px", "padding-right": "0px"},
                ),
            ],
            className=CLASSNAME_INFO_TEXT,
        ),
        html.Div(
            [
                html.H4(
                    "Thanks for Reading!",
                    style={"text-decoration": "underline", "marginLeft": 0},
                ),
                dcc.Markdown(
                    """If you enjoyed this Dash app or would have any questions on some of this methodology please feel free to contact me at **philheitz6[at]gmail[dot]com.** Thanks for reading!""",
                    style={"padding-left": "0%", "padding-bottom": "15px"},
                ),
            ],
        ),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)


def filter_df_portfolio(test_data2, state, homeownership, purpose):

    data = test_data2.loc[
        (test_data2["addr_state"] == state)
        & (test_data2["home_ownership"] == homeownership)
        & (test_data2["purpose"] == purpose)
    ]
    data = data.sort_values("xirr", ascending=False)
    return data[:10]


# Datatable ROC_AUC
@app.callback(
    Output("datatable-interactivity", "style_data_conditional"),
    Input("datatable-interactivity", "selected_columns"),
)
def update_styles(selected_columns):
    return [
        {"if": {"column_id": i}, "background_color": "#D2F3FF"}
        for i in selected_columns
    ]


@app.callback(
    Output("datatable-interactivity-container", "children"),
    Input("datatable-interactivity", "derived_virtual_data"),
    Input("datatable-interactivity", "derived_virtual_selected_rows"),
)
def update_graphs(rows, derived_virtual_selected_rows):
    """Returns first Non-2018 data graph"""
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []

    dff = ML_RESULTS if rows is None else pd.DataFrame(rows)

    colors = [
        "#a9bb95" if i in derived_virtual_selected_rows else "#5dbcd2"
        for i in range(len(dff))
    ]

    return [
        dcc.Graph(
            id=column,
            figure={
                "data": [
                    {
                        "x": dff["Model"],
                        "y": dff[column],
                        "type": "bar",
                        "marker": {"color": colors},
                    }
                ],
                "layout": {
                    "xaxis": {"automargin": True},
                    "yaxis": {"automargin": True, "title": {"text": column},},
                    "title": {"text": column},
                    "height": 400,
                    "margin": {"t": 40, "l": 40, "r": 10},
                    "paper_bgcolor": "#F9F9F9",
                    "plot_bgcolor": "#F9F9F9",
                },
            },
        )
        for column in ["Non-2018 test data AUC score"]
        if column in dff
    ]


@app.callback(
    Output("datatable-interactivity-container2", "children"),
    Input("datatable-interactivity", "derived_virtual_data"),
    Input("datatable-interactivity", "derived_virtual_selected_rows"),
)
def update_graphs(rows, derived_virtual_selected_rows):
    """Returns second 2018 graph"""
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []

    dff = ML_RESULTS if rows is None else pd.DataFrame(rows)

    colors = [
        "#a9bb95" if i in derived_virtual_selected_rows else "#5dbcd2"
        for i in range(len(dff))
    ]

    return [
        dcc.Graph(
            id=column,
            figure={
                "data": [
                    {
                        "x": dff["Model"],
                        "y": dff[column],
                        "type": "bar",
                        "marker": {"color": colors},
                    }
                ],
                "layout": {
                    "xaxis": {"automargin": True},
                    "yaxis": {"automargin": True, "title": {"text": column},},
                    "title": {"text": column},
                    "height": 400,
                    "margin": {"t": 40, "l": 40, "r": 10},
                    "paper_bgcolor": "#F9F9F9",
                    "plot_bgcolor": "#F9F9F9",
                },
            },
        )
        for column in ["2018 test data AUC score"]
        if column in dff
    ]


@app.callback(Output("cutoff_graph", "figure"), [Input("model_options", "value")])
def make_cutoff_figure(model):
    """Render cutoff threshold lineplot graph"""
    trace1 = go.Scatter(
        x=[x[0] for x in CF_DIFF],
        y=[x[1] for x in CF_DIFF],
        name="Dollar Loss",
        yaxis="y1",
        mode="lines+markers",
        line=dict(shape="spline", color="#5dbcd2", smoothing=1, width=1),
        marker=dict(symbol="diamond-open", color="#5dbcd2", size=4),
    )

    data = [trace1]

    layout = go.Layout(
        paper_bgcolor="#F9F9F9",
        plot_bgcolor="#F9F9F9",
        margin=dict(l=10, r=10, b=10, t=30,),
        title=dict(text="Total Dollar Misclassification Loss by Cutoff Threshold"),
        yaxis_title="Dollar Loss ($B)",
        xaxis_title="Cutoff Threshold",
    )

    figure = go.Figure(data=data, layout=layout)

    figure.update_layout(
        title_font_color="black", title_font_size=18, title_x=0.5, font_color="black"
    )
    figure.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor="Lightgrey", zeroline=False
    )
    figure.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor="Lightgrey", zeroline=False
    )
    figure.add_shape(
        type="line",
        x0=0.73737,
        y0=-9000000000,
        x1=0.73737,
        y1=0,
        line=dict(color="#a9bb95", width=3),
    )
    return figure


@app.callback(
    Output("confusion_matrix_mlp", "figure"), [Input("model_options", "value")]
)
def make_cm_figure(model):
    """Render confusion matrix graph"""
    colorscale = [[0, "blue"], [1, "orange"]]
    font_colors = ["white", "white", "white", "black"]
    fig = ff.create_annotated_heatmap(
        x=["Predicted - Default", "Predicted - No Default"],
        y=["True - No Default", "True - Default"],
        z=[[24418, 454275], [63935, 25927]],
        colorscale=colorscale,
        font_colors=font_colors,
    )

    fig.update_layout(
        title_text="<i><b>Multilayer Perceptron Neural Net Confusion Matrix</b></i>",
        paper_bgcolor="#F9F9F9",
        plot_bgcolor="#F9F9F9",
        autosize=True,
    )
    fig["layout"]["xaxis"]["side"] = "bottom"
    return fig


@app.callback(
    Output("confusion_matrix_linear", "figure"), [Input("linear_models", "value")]
)
def make_cm_figure(model):
    """Render confusion matrix figure"""
    colorscale = [[0, "blue"], [1, "green"]]
    font_colors = ["white"]
    if model == "QDA":
        z = [[329939, 148754], [82782, 7080]]
    elif model == "LDA":
        z = [[24418, 454275], [63935, 25927]]
    elif model == "LOGIT":
        z = [[215394, 263299], [53999, 35863]]
    elif model == "GNB":
        z = [[135632, 343061], [77864, 12178]]
    elif model == "MNB":
        z = [[259339, 219354], [59003, 30859]]

    fig = ff.create_annotated_heatmap(
        x=["Predicted - Default", "Predicted - No Default"],
        y=["True - No Default", "True - Default"],
        z=z,
        colorscale=colorscale,
        font_colors=font_colors,
    )

    fig.update_layout(
        title_text="<i><b>{} Confusion Matrix </b></i>".format(
            list(MODEL_ABBREVIATIONS.keys())[
                list(MODEL_ABBREVIATIONS.values()).index(model)
            ]
        ),
        paper_bgcolor="#F9F9F9",
        plot_bgcolor="#F9F9F9",
        autosize=True,
    )
    fig["layout"]["xaxis"]["side"] = "bottom"

    return fig


@app.callback(
    Output("confusion_matrix_linear2", "figure"), [Input("linear_models2", "value")]
)
def make_cm_figure(model):

    colorscale = [[0, "blue"], [1, "green"]]
    font_colors = ["white"]

    if model == "QDA":
        z = [[329939, 148754], [82782, 7080]]
    elif model == "LDA":
        z = [[24418, 454275], [63935, 25927]]
    elif model == "LOGIT":
        z = [[215394, 263299], [53999, 35863]]
    elif model == "GNB":
        z = [[135632, 343061], [77864, 12178]]
    elif model == "MNB":
        z = [[259339, 219354], [59003, 30859]]

    fig = ff.create_annotated_heatmap(
        x=["Predicted - Default", "Predicted - No Default"],
        y=["True - No Default", "True - Default"],
        z=z,
        colorscale=colorscale,
        font_colors=font_colors,
    )

    fig.update_layout(
        title_text="<i><b>{} Confusion matrix</b></i>".format(
            list(MODEL_ABBREVIATIONS.keys())[
                list(MODEL_ABBREVIATIONS.values()).index(model)
            ]
        ),
        paper_bgcolor="#F9F9F9",
        plot_bgcolor="#F9F9F9",
        autosize=True,
    )
    fig["layout"]["xaxis"]["side"] = "bottom"
    return fig


@app.callback(
    Output("confusion_matrix_tree", "figure"), [Input("tree_models", "value")]
)
def make_cm_figure(model):

    colorscale = [[0, "blue"], [1, "yellow"]]
    font_colors = ["white", "white", "white", "black"]

    if model == "CAT":
        z = [[13093, 464451], [12654, 5134]]
    elif model == "GBC":
        z = [[22026, 456667], [63556, 26306]]
    elif model == "RF":
        z = [[16416, 462277], [51262, 38600]]

    fig = ff.create_annotated_heatmap(
        x=["Predicted - Default", "Predicted - No Default"],
        y=["True - No Default", "True - Default"],
        z=z,
        colorscale=colorscale,
        font_colors=font_colors,
    )

    fig.update_layout(
        title_text="<i><b>{} Confusion matrix</b></i>".format(
            list(MODEL_ABBREVIATIONS.keys())[
                list(MODEL_ABBREVIATIONS.values()).index(model)
            ]
        ),
        paper_bgcolor="#F9F9F9",
        plot_bgcolor="#F9F9F9",
        autosize=True,
    )
    fig["layout"]["xaxis"]["side"] = "bottom"
    return fig


@app.callback(
    Output("confusion_matrix_tree2", "figure"), [Input("tree_models2", "value")]
)
def make_cm_figure(model):

    colorscale = [[0, "blue"], [1, "yellow"]]
    font_colors = ["white", "white", "white", "black"]

    if model == "CAT":
        z = [[13093, 464451], [12654, 5134]]
    elif model == "GBC":
        z = [[22026, 456667], [63556, 26306]]
    elif model == "RF":
        z = [[16416, 462277], [51262, 38600]]

    fig = ff.create_annotated_heatmap(
        x=["Predicted - Default", "Predicted - No Default"],
        y=["True - No Default", "True - Default"],
        z=z,
        colorscale=colorscale,
        font_colors=font_colors,
    )

    fig.update_layout(
        title_text="<i><b>{} Confusion matrix</b></i>".format(
            list(MODEL_ABBREVIATIONS.keys())[
                list(MODEL_ABBREVIATIONS.values()).index(model)
            ]
        ),
        paper_bgcolor="#F9F9F9",
        plot_bgcolor="#F9F9F9",
        autosize=True,
    )

    fig["layout"]["xaxis"]["side"] = "bottom"
    return fig


@app.callback(Output("annual_inc", "children"), [Input("id_options", "value")])
def output_stats(id1):
    """Return annual income for given user"""
    num = x_train_lg1.loc[id1, "annual_inc"]
    num = "{0:,.2f}".format(num)
    return "$" + str(num)


@app.callback(Output("fico_score_text", "children"), [Input("id_options", "value")])
def output_fico(id1):
    """Return FICO score for given user"""
    num = x_train_lg1.loc[id1, "last_fico_range_high"]
    return str(num)


@app.callback(Output("acc_text", "children"), [Input("id_options", "value")])
def output_fico(id1):
    """Return total accounts for given user"""
    num = x_train_lg1.loc[id1, "total_acc"]
    return str(num)


@app.callback(Output("inq_text", "children"), [Input("id_options", "value")])
def output_fico(id1):
    """Return number of inquiries for given user"""
    num = x_train_lg1.loc[id1, "inq_fi"]
    return str(np.round(num, 2))


@app.callback(Output("emp_length_text", "children"), [Input("id_options", "value")])
def output_fico(id1):
    """Return employment length for given user"""
    num = np.round(x_train_lg1.loc[id1, "emp_length"], 2)
    return str(num)


def filter_dataframe(df, year_slider):
    dff = df[(df["Year"] >= year_slider[0]) & (df["Year"] <= year_slider[1])]
    return dff


@app.callback(
    Output("main_graph", "figure"),
    [Input("year_slider", "value")],
    [State("lock_selector", "value"), State("main_graph", "relayoutData")],
)
def make_main_figure(year_slider, selector, main_graph_layout):
    """Create interactive choropleth plot"""
    dff = filter_dataframe(eda2, year_slider)
    fig = go.Figure(
        data=go.Choropleth(
            # Spatial coordinates
            locations=dff["addr_state"],
            # Data to be color-coded
            z=np.round(dff["loan_status"].astype(float), 3),
            # Set of locations match entries in `locations`
            locationmode="USA-states",
            colorscale="Blues",
            colorbar_title="Default Rates (%)",
        )
    )

    fig.update_layout(
        title_text="Lending Club Default Rates by State",
        title_x=0.5,
        geo_scope="usa",
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        font_color="black",
    )

    if main_graph_layout is not None and "locked" in selector:
        zoom = float(main_graph_layout["mapbox"]["zoom"])
        layout["mapbox"]["zoom"] = zoom

    return fig


def filter_dataframe_ind(df, state):
    dff = df.loc[df["addr_state"] == state, :]
    return dff


# Main graph -> individual graph
@app.callback(Output("individual_graph", "figure"), [Input("main_graph", "hoverData")])
def make_individual_figure(main_graph_hover):

    if main_graph_hover is None:
        main_graph_hover = {"points": [{"location": "MA"}]}

    chosen = [point["location"] for point in main_graph_hover["points"]]
    data1 = filter_dataframe_ind(eda3, chosen[0])
    data2 = filter_dataframe_ind(eda4, chosen[0])

    data = [
        dict(
            type="scatter",
            mode="lines+markers",
            name="Loan Grades as Percent of Total Issuance (%)",
            x=data1["grade"],
            y=np.round(data1["loan_amnt"], 2),
            line=dict(shape="spline", smoothing=2, width=1, color="#5dbcd2"),
            marker=dict(
                symbol="diamond-open",  # color=['rgba(204,204,204,1)', 'rgba(222,45,38,0.8)']
            ),
        ),
        dict(
            type="scatter",
            mode="lines+markers",
            name="Interest Rates by Grade (%)",
            x=data2["grade"],
            y=data2["int_rate"],
            line=dict(shape="spline", smoothing=2, width=1, color="#a9bb95"),
            marker=dict(symbol="diamond-open"),
        ),
    ]

    figure = dict(
        data=data,
        layout={
            "autosize": True,
            "automargin": True,
            "margin": dict(l=50, r=30, b=40, t=40),
            "hovermode": "closest",
            "plot_bgcolor": "#F9F9F9",
            "paper_bgcolor": "#F9F9F9",
            "title": "{} Loan Grade & Interest Rates Distribution".format(chosen[0]),
            "yaxis": {"title": "Percent (%)"},
            "xaxis": {"title": "Loan Grade"},
            "legend": dict(yanchor="bottom", y=0.05, xanchor="left", x=0.01),
        },
    )
    return figure


# Main graph -> reg_graph1 graph
@app.callback(Output("reg_graph1", "figure"), [Input("main_graph", "hoverData")])
def make_reg1_figure(main_graph_hover):

    if main_graph_hover is None:
        main_graph_hover = {"points": [{"location": "MA"}]}

    chosen = [point["location"] for point in main_graph_hover["points"]]
    data1 = filter_dataframe_ind(eda5, chosen[0])
    data1 = data1.loc[(data1["last_fico_range_high"] > 495)]

    # Add traces
    trace1 = go.Scatter(
        x=data1["last_fico_range_high"],
        y=np.round(data1["loan_status"], 3),
        name="Repayment Rate",
        yaxis="y1",
        mode="lines+markers",
        line=dict(shape="spline", color="#5dbcd2", smoothing=1, width=1),
        marker=dict(symbol="diamond-open", color="#5dbcd2", size=6),
    )

    trace2 = go.Scatter(
        x=data1["last_fico_range_high"],
        y=np.round(data1["int_rate"], 3),
        name="Interest Rate",
        yaxis="y2",
        mode="lines+markers",
        line=dict(shape="spline", color="#a9bb95", smoothing=1, width=1),
        marker=dict(symbol="diamond-open", color="#a9bb95", size=6),
    )

    data3 = [trace1, trace2]

    figure = dict(
        data=data3,
        layout={
            "autosize": True,
            "automargin": True,
            "margin": dict(l=50, r=50, b=40, t=40),
            "hovermode": "closest",
            "plot_bgcolor": "#F9F9F9",
            "paper_bgcolor": "#F9F9F9",
            "title": "{} Loan Default & Interest Rate by FICO Score".format(chosen[0]),
            "yaxis": {"title": "Repayment Rate (%)", "showgrid": False},
            "yaxis2": dict(
                title="Interest Rate (%)", overlaying="y", side="right", showgrid=False
            ),
            "xaxis": {"title": "FICO Score"},
            "legend": dict(yanchor="bottom", y=0.05, xanchor="left", x=0.01),
        },
    )
    return figure


# Main graph -> reg_graph2 graph
@app.callback(Output("reg_graph2", "figure"), [Input("main_graph", "hoverData")])
def make_reg2_figure(main_graph_hover):

    if main_graph_hover is None:
        main_graph_hover = {"points": [{"location": "MA"}]}

    chosen = [point["location"] for point in main_graph_hover["points"]]
    data1 = filter_dataframe_ind(eda1, chosen[0])
    clrs = [COLOR_BLUE if x == 1 else COLOR_GREEN for x in data1["above_average"]]

    data = [
        dict(
            type="bar",
            name=data1["above_average"],
            x=data1["emp_length"],
            y=np.round(data1["loan_status"], 3),
            line=dict(shape="spline", smoothing=2, width=1, color="#5dbcd2"),
            marker=dict(symbol="triangle", color=clrs),
            # showlegend = True,
        ),
    ]

    figure = dict(
        data=data,
        layout={
            "autosize": True,
            "automargin": True,
            "margin": dict(l=50, r=30, b=40, t=40),
            "hovermode": "closest",
            "plot_bgcolor": "#F9F9F9",
            "paper_bgcolor": "#F9F9F9",
            "title": "{} Default Rate by Employment Length".format(chosen[0]),
            "yaxis": {"title": "Default Rate (%)", "showgrid": False},
            "xaxis": {"title": "Employment Length (years)"},
            "legend": dict(yanchor="bottom", y=1.02, xanchor="right", x=1),
        },
    )
    return figure


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port="8050", debug=True)
