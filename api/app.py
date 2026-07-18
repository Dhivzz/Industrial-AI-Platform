from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


app = FastAPI(
    title="Industrial AI Predictive Maintenance API"
)


# Load model

model = joblib.load(
    "models/random_forest_model.pkl"
)



# Input schema

class MachineInput(BaseModel):

    air_temperature: float
    process_temperature: float
    rotational_speed: int
    torque: float
    tool_wear: int
    temperature_difference: float
    power: float
    type_L: bool
    type_M: bool
    tool_wear_medium: bool
    tool_wear_high: bool



@app.get("/")
def home():

    return {
        "message": "Industrial AI Predictive Maintenance API Running"
    }



@app.post("/predict")
def predict(machine: MachineInput):


    data = {

        "Air temperature [K]": machine.air_temperature,

        "Process temperature [K]": machine.process_temperature,

        "Rotational speed [rpm]": machine.rotational_speed,

        "Torque [Nm]": machine.torque,

        "Tool wear [min]": machine.tool_wear,

        "Temperature Difference": machine.temperature_difference,

        "Power": machine.power,

        "Type_L": machine.type_L,

        "Type_M": machine.type_M,

        "Tool Wear Category_Medium": machine.tool_wear_medium,

        "Tool Wear Category_High": machine.tool_wear_high

    }



    input_df = pd.DataFrame(
        [data]
    )


    prediction = model.predict(
        input_df
    )[0]


    probability = model.predict_proba(
        input_df
    )[0][1]



    probability = round(
        probability * 100,
        2
    )



    if probability < 30:

        risk = "Low"

    elif probability < 70:

        risk = "Medium"

    else:

        risk = "High"



    feature_importance = {

        "Rotational Speed": 20.9,

        "Torque": 20.9,

        "Power": 19.3,

        "Tool Wear": 13.7,

        "Temperature Difference": 9.1

    }



    return {

        "failure_prediction": int(prediction),

        "risk": risk,

        "failure_probability": probability,

        "feature_importance": feature_importance

    }