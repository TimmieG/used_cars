import streamlit as st
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('final.csv')


st.write("""
# Used Vehicle Price Prediction App

This app predicts the **Price of Used Vehicles**!
""")
st.write('---')

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    year = st.sidebar.slider('year', df['year'].min(), df['year'].max(), df['year'].mean())
    manufacturer = st.sidebar.selectbox("Select a car model", df['manufacturer'].unique())
    condition = st.sidebar.selectbox("Select a car condition", df['condition'].unique())  
    cylinders =  st.sidebar.selectbox("Select a car cylinders type", df['cylinders'].unique())
    fuel = st.sidebar.selectbox("Select a car fuel type", df['fuel'].unique())
    odometer = st.sidebar.slider('Odometer', df['odometer'].min(), df['odometer'].max(), df['odometer'].mean())
    title_status = st.sidebar.slider('Title Status', df['title_status'].min(), df['title_status'].max(), df['title_status'].mean())
    transmission = st.sidebar.selectbox("Select a car transmission type", df['transmission'].unique())
    drive = st.sidebar.selectbox("Select a car drive train", df['drive'].unique())
    size = st.sidebar.slider('Size', df['size'].min(), df['size'].max(), df['size'].mean())
    vehicle_type = st.sidebar.slider('Type', df['type'].min(), df['type'].max(), df['type'].mean())
    state = st.sidebar.selectbox("Select State", df['state'].unique())
    
    data = {
        'year': year,
        'manufacturer': manufacturer,
        'condition': condition,
        'cylinders': cylinders,
        'fuel': fuel,
        'odometer': odometer,
        'title_status': title_status,
        'transmission': transmission,
        'drive': drive,
        'size': size,
        'type': vehicle_type,
        'state': state
    }
    
    features = pd.DataFrame(data, index=[0])  # Create DataFrame with a single row
    return features

df1 = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df1)
st.write('---')

model_loaded = pickle.load(open('modelLgb.pkl', 'rb'))

# Regression Model
#model = RandomForestRegressor()
#model.fit(X, Y)
# Apply Model to Moke Prediction
#prediction = model.predict(df1)

st.header('Prediction of Price')
st.write("Predicted price to be filled here")
st.write('---')




# Explaining the model's predictions using SHAP values
#explainer = shap.TreeExplainer(model)
#shap_values = explainer.shap_values(X)

#st.header('Feature Importance')
#plt.title('Feature importance based on SHAP values')
#shap.summary_plot(shap_values, X)
#st.pyplot(bbox_inches='tight')
#st.write('---')

#plt.title('Feature importance based on SHAP values (Bar)')
#shap.summary_plot(shap_values, X, plot_type="bar")
#st.pyplot(bbox_inches='tight')
