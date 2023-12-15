import streamlit as st 
import numpy as np 
import pandas as pd
import seaborn as sns
import statsmodels
import datetime
import matplotlib.pyplot as plt 
# Module to build AR, MA, ARMA, and ARIMA models
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(layout = 'wide', initial_sidebar_state = 'expanded')

st.header("Past and Future Trends: Bloomington, Indiana")
st.subheader("Context:")
link = "https://www.zillow.com/research/data/"
st.write("""Bloomington is a college town located in central Indiana. The city population is about 80,000 (Cencus 2020) and it hosts main campus of 
         Indiana University. 
         """)
st.subheader("Purpose:")
st.markdown("""Real state property and rental prices have been skyrocketing all across the United States for the past several years. 
            Based on the available datas, we focus our study to analyze characteristics of different housing related informationto to 
            estimate past trends. We then leverage power of data science and machine learning to make models, which make predictions
            into future. We present all of these findings displaying interactive graphs.""")

st.subheader("Data:")
st.markdown("""There are three data sets we want to look for patterns. One of the datasets consists of median sale price of all houses sold in a month, 
            the second data set consists of average monthly rent, and third one is related to number of house units available for sale each month. All of these data
            sets have been downloaded from zillow. To learn more about methodologies, experiments, etc about the data click here: """ f"{link}.")
list = ['Median Sale Price (msp) dataset', 'Average Rent dataset', 'Units for Sale dataset']
st.markdown("\n".join([f" - {item}" for item in list]))


header = st.container()
dataset = st.container()
model = st.container()
forecasting = st.container()
  


with header:
    st.subheader('Visualization of Dataset')
    dataset_name = st.selectbox("**Select Dataset**",("Median Sale Price","Average Rent", "Units for Sale" )) 
    #get the dataset:
    def get_data(data):
        if data == "Median Sale Price":
            housingprice = pd.read_csv('house_price.csv')
            housingprice1 = housingprice.copy()
            housingprice['month'] = pd.to_datetime(housingprice['month']).dt.date
            housingprice.set_index('month', inplace = True)
        elif data == "Average Rent":
            housingprice = pd.read_csv('rental_processed.csv')
            housingprice1 = housingprice.copy()
            housingprice['month'] = pd.to_datetime(housingprice['month']).dt.date
            housingprice.set_index('month', inplace = True)
        else:
            housingprice = pd.read_csv('unit_data.csv')
            housingprice1 = housingprice.copy()
            housingprice['month'] = pd.to_datetime(housingprice['month']).dt.date
            housingprice.set_index('month', inplace = True)
        return housingprice, housingprice1
    house_data, house_data1 = get_data(dataset_name)
        
         
    
    # Create two columns
    user_input1 = st.number_input('Enter an Integer', value=1, step=1)
    col1, col2 = st.columns(2)
    with col1:
        st.write(f'First {user_input1} rows: ', house_data1.head(user_input1))
    with col2:
        st.write(f'Last {user_input1} rows: ', house_data1.tail(user_input1))
    

    st.markdown("**Plotting Dataset:**")   
    fig, ax = plt.subplots()
    house_data.plot(ax = ax, color= 'red', marker = '*')
    plt.show()
    st.pyplot(fig)
    
#fit the ARIMA model
    def fit_arima(data, order = (3,1,3)):
        model = ARIMA(data, order = order)
        result = model.fit()
        return result
    #function which computes forecasts 'step' steps away and outputs a dataframe.
    def make_forecast(step):
        result = fit_arima(house_data) # fitting the model
        prediction = result.forecast(step) # forecasting
        price = prediction.reset_index() #to change two dimensional pandas series to a dataframe.
        price.rename(columns = {'index':'month', 'predicted_mean':'Predicted_value'}, inplace = True) #renaming column names
        price['month'] = price['month'].dt.date # extracting date only from datetime object. 
        price1 = price.copy()
        price1.set_index('month')
        return price, price1


with forecasting:
    st.subheader('Forecasting:')
    st.write("""To make a forecasting of future trends based on past values, we have built a time series model
             called ARIMA. Based on various metrics and parameter tuning, the model paramaters have been chosen to be (3,2,2). Below type in an integer as a step
             into future you want to make prediction: """)
     # Get an integer input from the user
     
    user_input = st.number_input('Enter an Integer', value=2, step=1)
    st.write(f"You are making prediction {user_input} months away from {house_data.index[-1].year}-{house_data.index[-1].month}-{house_data.index[-1].day}")
          
    col3,col4 = st.columns(2)
    with col3:
        # Display the user input
        prediction, prediction1 = make_forecast(user_input)
        
        # function to calculate growth rate:
        def growth_rate(A):
            price_initial = prediction.iloc[0].values[1]
            price_last = prediction.iloc[-1].values[1]
            rate = (price_last-price_initial)/price_initial
            formatted_rate = '{:.2%}'.format(rate)
            return formatted_rate
        rate = growth_rate(user_input)
        st.write('Prediction:', prediction)
        M = prediction.iloc[0].values[0]
        N = prediction.iloc[-1].values[0]
        final_growthrate = f'Growth Rate from: {M} to: {N} is =  {rate}'
        st.write(final_growthrate)
    with col4:
        figure1, ax1 = plt.subplots(1,1,figsize = (6,4))
        prediction1.plot(ax = ax1, color = 'blue', marker = '*')
        #house_data.plot(ax = ax1, color = 'red', marker = '*')
        plt.xlabel('Number of months')
        plt.ylabel('price/unit')
        plt.title('Forecasting Over Months')
        plt.show()
        st.pyplot(figure1)
        
    

    
    
