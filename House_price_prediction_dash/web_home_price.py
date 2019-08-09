import dash
import dash_core_components as dcc
import dash_html_components as html
from textwrap import dedent
from numpy import mean
import pandas as pd
import numpy as np

from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import xgboost as xgb

train = pd.read_csv("./home-data-for-ml-course/train.csv")
train = train[['LotArea', 'GrLivArea', 'BsmtFinSF1', 'OverallQual', 'OverallCond',
       'TotalBsmtSF', 'YearBuilt', 'GarageYrBlt', '1stFlrSF', 'GarageArea',
       '2ndFlrSF', 'Functional', 'OpenPorchSF', 'WoodDeckSF', 'BsmtUnfSF',
       'YearRemodAdd', 'BsmtExposure', 'LotFrontage', 'BsmtQual', 'EnclosedPorch', 'SalePrice']]

train["LotFrontage"] = train["LotFrontage"].fillna(train["LotFrontage"].median())

train["GarageYrBlt"] = train["GarageYrBlt"].fillna(0)

train["BsmtExposure"] = train["BsmtExposure"].fillna("None")
train["BsmtQual"] = train["BsmtQual"].fillna("None")


#Changing OverallCond into a categorical variable
train['OverallCond'] = train['OverallCond'].astype(str)

cols = ('BsmtQual', 'Functional', 'BsmtExposure','OverallCond')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train[c].values)) 
    train[c] = lbl.transform(list(train[c].values))



train["SalePrice"] = np.log1p(train["SalePrice"])

y_train = train["SalePrice"]
x_train = train.drop(axis=1,columns=["SalePrice"])

model_xgb = xgb.XGBRegressor(n_jobs=8)
model_xgb.fit(x_train, y_train)
bcnts = 0

app = dash.Dash()

app.layout = html.Div([
							html.H1("House price prediction",
									style={'text-align': 'center'}),

							html.Div([
								dcc.Markdown(dedent("Please input the **total area** of the house.(range 1500 - 20000).")),
								dcc.Input(
									placeholder='Full area',
									type='number',
									value=10000,
									id='full_area'
								),
								dcc.Markdown(dedent("Please input the **built year** of the house (range 1900 - 2010).")),
								dcc.Input(
									placeholder='Year Built',
									type='number',
									value=1975,
									id='year_built'
								)
							], style={'display': 'inline-block', 'width': '100%', 'margin': '1.5vh'}),
							
							html.Div([
								dcc.Markdown(dedent("Please input the **total Garage area** (range 0 - 1500).")),
								dcc.Input(
									placeholder='Garage area',
									type='number',
									value=500,
									id='grg_area'
								),
								dcc.Markdown(dedent("Please input the **garage built year** (range 1900 - 2010).")),
								dcc.Input(
									placeholder='Garage Year Built',
									type='number',
									value=1980,
									id='grg_year_built'
								)
							], style={'display': 'inline-block', 'width': '100%', 'margin': '1.5vh'}),
								
							html.Div([
								dcc.Markdown(dedent("Please select the **overall condition** of the house.")),
								dcc.Dropdown(
									options=[{'label': name, 'value': name} for name in range(9)],
									id='overall_cond'
								),
								dcc.Markdown(dedent("Please select the **overall quality** of the house.")),
								dcc.Dropdown(
									options=[{'label': name, 'value': name} for name in range(1,11)],
									id='overall_qlty'
								)
							], style={'display': 'inline-block', 'width': '100%', 'margin': '1.5vh'}),	
							
							html.Div([
								dcc.Markdown(dedent("Please select the **basement quality** of the house.")),
								dcc.Dropdown(
									options=[{'label': name, 'value': name} for name in range(5)],
									id='bsmt_qual'
								),
								dcc.Markdown(dedent("Please select the **basement exposure** of the house.")),
								dcc.Dropdown(
									options=[{'label': name, 'value': name} for name in range(5)],
									id='bsmt_exposure'
								)], style={'display': 'inline-block', 'width': '100%', 'margin': '1.5vh'}),
								
							html.Button('Submit', id='button', style={'display': 'inline-block', 'margin': '1.5vh'}),

							html.Div(id="output")
						], style={'display': 'inline-block', 'width': '120vh', 'margin-left': '35vh', 'margin-right': '35vh'})



@app.callback(dash.dependencies.Output('output', 'children'),
			  [dash.dependencies.Input('full_area', 'value'),
			   dash.dependencies.Input('grg_area', 'value'),
			   dash.dependencies.Input('year_built', 'value'),
			   dash.dependencies.Input('grg_year_built', 'value'),
			   dash.dependencies.Input('overall_cond', 'value'),
			   dash.dependencies.Input('overall_qlty', 'value'),
			   dash.dependencies.Input('bsmt_qual', 'value'),
			   dash.dependencies.Input('bsmt_exposure', 'value'),
			   dash.dependencies.Input('button', 'n_clicks')])
def update_output(full_area, grg_area, year_built, grg_year_built, overall_cond, overall_qlty, bsmt_qual, bsmt_exposure, n_clicks):
	global bcnts
	if n_clicks <=bcnts:
		return ""
	bcnts+=1
	pred = pd.DataFrame(columns=x_train.columns)
	pred.loc[0] = [full_area, 1450, 383.5, overall_qlty, overall_cond, 991.5, year_built, grg_year_built, 1087, grg_area, 0, 6, 25, 0, 477, 1994, bsmt_exposure, 69, bsmt_qual, 0]

	out = np.expm1(model_xgb.predict(pred)[0])

	return dcc.Markdown(dedent("The predicted value for the house is "+str(out)+str(n_clicks)+" dollars."))



if __name__ == '__main__':
	
    app.run_server()