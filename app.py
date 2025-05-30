import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import io
import dash.dcc as dcc
import dash.html as html
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from flask import send_file
import flask

# Load dataset
file_path = "House Price Prediction Dataset.csv"
df = pd.read_csv(file_path)

# Convert YearBuilt to datetime (pseudo-time series)
df['YearBuilt'] = pd.to_datetime(df['YearBuilt'], format='%Y')
df = df.groupby(df['YearBuilt'].dt.year)['Price'].mean().reset_index()
df['YearBuilt'] = pd.to_datetime(df['YearBuilt'], format='%Y')

def forecast_prices(data, periods=2):
    try:
        model = ARIMA(data['Price'], order=(5, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=periods)
        future_dates = pd.date_range(start=data['YearBuilt'].iloc[-1], periods=periods+1, freq='Y')[1:]
        return pd.DataFrame({'YearBuilt': future_dates, 'Price': forecast})
    except Exception as e:
        return pd.DataFrame({'YearBuilt': [], 'Price': []})

# Generate initial predictions
forecast_df = forecast_prices(df, periods=2)

theme_options = {'light': 'plotly_white', 'dark': 'plotly_dark'}
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)

app.layout = html.Div([
    html.H1("House Price Prediction Dashboard - Done by Mann"),
    dcc.RadioItems(
        id='theme-toggle',
        options=[{'label': 'Light Mode', 'value': 'light'}, {'label': 'Dark Mode', 'value': 'dark'}],
        value='light',
        inline=True
    ),
    dcc.DatePickerRange(
        id='date-picker',
        start_date=df['YearBuilt'].min(),
        end_date=df['YearBuilt'].max()
    ),
    dcc.Dropdown(
        id='period-dropdown',
        options=[{'label': f'{i} Years', 'value': i} for i in range(1, 7)],
        value=2,
        clearable=False
    ),
    dcc.RangeSlider(
        id='price-range',
        min=df['Price'].min(),
        max=df['Price'].max(),
        step=10000,
        marks={int(price): str(int(price)) for price in np.linspace(df['Price'].min(), df['Price'].max(), num=5)},
        value=[df['Price'].min(), df['Price'].max()]
    ),
    html.Button("Update Forecast", id='update-button', n_clicks=0),
    html.A("Download Report", id='download-link', href="/download_report", download="House_Price_Report.csv"),
    dcc.Graph(id='price-chart'),
    dcc.Graph(id='bar-chart'),
    dcc.Graph(id='trend-chart')
])

@app.callback(
    [Output('price-chart', 'figure'), Output('bar-chart', 'figure'), Output('trend-chart', 'figure')],
    [Input('update-button', 'n_clicks'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('period-dropdown', 'value'),
     Input('price-range', 'value'),
     Input('theme-toggle', 'value')]
)
def update_graph(n_clicks, start_date, end_date, periods, price_range, theme):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    filtered_df = df[(df['YearBuilt'] >= start_date) & (df['YearBuilt'] <= end_date) &
                     (df['Price'] >= price_range[0]) & (df['Price'] <= price_range[1])]
    forecast_df = forecast_prices(filtered_df, periods)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df['YearBuilt'], y=filtered_df['Price'], mode='lines+markers', name='Actual Prices'))
    fig.add_trace(go.Scatter(x=forecast_df['YearBuilt'], y=forecast_df['Price'], mode='lines+markers', name='Predicted Prices'))
    fig.update_layout(title='House Price Trends & Prediction', xaxis_title='Year', yaxis_title='Price', template=theme_options[theme])
    
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(x=filtered_df['YearBuilt'], y=filtered_df['Price'], marker=dict(color='blue')))
    bar_fig.update_layout(title='Average House Price per Year', xaxis_title='Year', yaxis_title='Price', template=theme_options[theme])
    
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(x=filtered_df['YearBuilt'], y=filtered_df['Price'].rolling(3).mean(), mode='lines', name='Rolling Avg'))
    trend_fig.update_layout(title='Price Trend with Rolling Average', xaxis_title='Year', yaxis_title='Price', template=theme_options[theme])
    
    return fig, bar_fig, trend_fig

@server.route('/download_report')
def download_report():
    report_df = df.copy()
    output = io.StringIO()
    report_df.to_csv(output, index=False)
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, attachment_filename='House_Price_Report.csv')
