import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the model
model = joblib.load('model.pkl')

preprocessor = joblib.load('preprocessor.pkl')

# Streamlit app
st.title("Road Accident Severity Prediction App")

# Create input widgets for user to input features

Day_of_Week=st.selectbox('Select a Day',('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'))
Junction_Control=st.selectbox('Select a Junction_Control',('Authorised person', 'Auto traffic signal', 'Give way or uncontrolled','Not at junction or within 20 metres','Stop sign'))
Light_Conditions=st.selectbox('Select a Light_Conditions',('Daylight', 'Darkness'))
Road_Surface_Conditions=st.selectbox('Select a Road_Surface_Conditions',('Dry', 'Wet or damp', 'Frost or ice'))
Road_Type=st.selectbox('Select a Road_Type',('Single carriageway', 'Dual carriageway', 'Roundabout','One way street'))
Speed_limit=st.slider('Select a Speed_limit', 10, 70,10)
Time = st.time_input("Select Time")
Urban_or_Rural_Area = st.radio("Select Urban or Rural Area", ['Urban', 'Rural'])
Weather_Conditions=st.selectbox('Select a ',('Fine no high winds', 'Fog or mist', 'Raining + high winds','Raining no high winds'))
Vehicle_Type=st.selectbox('Select a ',('Car', 'Van', 'Motorcycle'))

# Combine Date and Time into a single datetime column
Date = '2021-01-01'
DateTime = pd.to_datetime(f"{Date} {Time}")
# Get user input and make predictions
if st.button("Predict Accident Severity"):
    try:
        sample_data = pd.DataFrame({
            'Day_of_Week': [Day_of_Week],
            'Junction_Control': [Junction_Control],
            'Light_Conditions': [Light_Conditions],
            'Road_Surface_Conditions': [Road_Surface_Conditions],
            'Road_Type': [Road_Type],
            'Speed_limit': [Speed_limit],
            'Time':  [DateTime],
            'Urban_or_Rural_Area': [Urban_or_Rural_Area],
            'Weather_Conditions': [Weather_Conditions],
            'Vehicle_Type': [Vehicle_Type],
        })

        sample_data['Hour'] = sample_data['Time'].dt.hour

        # Drop the original 'Time' column
        sample_data = sample_data.drop('Time', axis=1)

        # Make predictions
        predictions = model.predict(sample_data)
        # Display predictions
        if predictions == 0:
            st.subheader('The predicted accident severity is :green[Slight] :sunglasses: :white_check_mark:')
        else:
            st.subheader('The predicted accident severity is :red[NOT Slight] :unamused: :-1: :x:')



    except Exception as e:
        st.error(f"An error occurred: {str(e)}")