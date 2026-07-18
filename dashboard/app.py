import streamlit as st
import requests


st.set_page_config(
    page_title="Industrial AI Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)


# Title
st.title(
    "⚙️ Industrial AI Predictive Maintenance Dashboard"
)

st.write(
    "AI-powered machine failure prediction and condition monitoring"
)



# Sidebar Inputs

st.sidebar.header(
    "Machine Parameters"
)


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



# Feature Engineering

temperature_difference = (
    process_temperature - air_temperature
)


power = (
    rotational_speed * torque
)


tool_wear_medium = (
    tool_wear >= 100 and tool_wear < 200
)


tool_wear_high = (
    tool_wear >= 200
)


type_L = machine_type == "L"


type_M = machine_type == "M"



# Prediction

if st.button(
    "🔍 Predict Machine Condition"
):


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


        prediction = result["failure_prediction"]

        probability = result["failure_probability"]

        risk = result["risk"]


        # Get explanation data
        feature_importance = result.get(
            "feature_importance",
            {}
        )


        st.divider()


        col1, col2, col3 = st.columns(3)



        # Machine Status

        with col1:

            st.subheader(
                "Machine Status"
            )

            if prediction == 0:

                st.success(
                    "🟢 HEALTHY"
                )

            else:

                st.error(
                    "🔴 FAILURE RISK"
                )



        # Probability

        with col2:

            st.subheader(
                "Failure Probability"
            )

            st.metric(
                "Risk Probability",
                f"{probability}%"
            )



        # Risk

        with col3:

            st.subheader(
                "Risk Level"
            )

            st.info(
                risk
            )



        # Parameters

        st.divider()


        st.subheader(
            "Machine Parameters"
        )


        parameter_data = {


            "Air Temperature": air_temperature,

            "Process Temperature": process_temperature,

            "Rotational Speed": rotational_speed,

            "Torque": torque,

            "Tool Wear": tool_wear,

            "Power": power,

            "Temperature Difference": temperature_difference

        }


        st.json(
            parameter_data
        )



        # Recommendation

        st.divider()


        st.subheader(
            "Maintenance Recommendation"
        )


        if risk == "Low":

            st.success(
                "Continue normal operation. Monitor machine condition regularly."
            )


        elif risk == "Medium":

            st.warning(
                "Schedule preventive inspection."
            )


        else:

            st.error(
                "Immediate maintenance recommended."
            )



        # Explainable AI

        st.divider()


        st.subheader(
            "🔍 AI Explanation"
        )


        st.write(
            "Top factors influencing machine prediction:"
        )


        if feature_importance:


            st.bar_chart(
                feature_importance
            )


        else:

            st.info(
                "Feature importance not available from API."
            )



    else:

        st.error(
            "API connection failed. Check FastAPI server."
        )