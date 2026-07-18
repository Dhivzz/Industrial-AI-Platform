from pydantic import BaseModel


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