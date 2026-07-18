from fastapi import FastAPI
import joblib
import pandas as pd

from api.schemas import MachineInput


app = FastAPI(
    title="Industrial AI Predictive Maintenance API"
)


model = joblib.load(
    "models/random_forest_model.pkl"
)


@app.get("/")
def home():

    return {
        "message":
        "Industrial AI Predictive Maintenance API running"
    }


@app.post("/predict")
def predict(machine: MachineInput):

    data = pd.DataFrame(
        [
            {
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
        ]
    )


    prediction = model.predict(
        data
    )[0]


    probability = model.predict_proba(
        data
    )[0][1]


    if prediction == 1:
        risk = "High"
    else:
        risk = "Low"


    return {
        "failure_prediction": int(prediction),
        "risk": risk,
        "failure_probability": round(probability * 100, 2)
    }