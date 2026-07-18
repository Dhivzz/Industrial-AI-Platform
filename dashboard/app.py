import streamlit as st
import requests


st.set_page_config(
    page_title="Industrial AI Predictive Maintenance",
    page_icon="⚙️",
    layout="centered"
)


st.title("⚙️ Industrial AI Predictive Maintenance System")

st.write(
    "Predict machine failure risk using Machine Learning"
)


st.sidebar.header("Machine Parameters")


air_temperature = st.sidebar.number_input(
    "Air Temperature (K)",
    value=300.5
)

process_temperature = st.sidebar.number_input(
    "Process Temperature (K)",
    value=309.8
)

rotational_speed = st.sidebar.number_input(
    "Rotational Speed (rpm)",
    value=1345
)

torque = st.sidebar.number_input(
    "Torque (Nm)",
    value=62.7
)

tool_wear = st.sidebar.number_input(
    "Tool Wear (min)",
    value=153
)


machine_type = st.sidebar.selectbox(
    "Machine Type",
    ["H", "L", "M"]
)


temperature_difference = (
    process_temperature - air_temperature
)


power = (
    rotational_speed * torque
)


tool_wear_medium = tool_wear >= 100 and tool_wear < 200
tool_wear_high = tool_wear >= 200


type_L = machine_type == "L"
type_M = machine_type == "M"



if st.button("Predict Machine Condition"):

    input_data = {

        "air_temperature": air_temperature,

        "process_temperature": process_temperature,

        "rotational_speed": rotational_speed,

        "torque": torque,

        "tool_wear": tool_wear,

        "temperature_difference": temperature_difference,

        "power": power,

        "type_L": type_L,

        "type_M": type_M,

        "tool_wear_medium": tool_wear_medium,

        "tool_wear_high": tool_wear_high
    }


    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=input_data
    )


    if response.status_code == 200:

        result = response.json()


        st.subheader("Prediction Result")


        if result["failure_prediction"] == 1:

            st.error(
                "⚠️ Machine Failure Risk Detected"
            )

        else:

            st.success(
                "✅ Machine Healthy"
            )


        st.metric(
            "Failure Probability",
            f'{result["failure_probability"]}%'
        )


        st.info(
            f'Risk Level: {result["risk"]}'
        )


    else:

        st.error(
            "API Error. Check FastAPI server."
        )