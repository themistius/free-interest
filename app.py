import mesa
from mesa_economic_model import EconomicModel, calculate_gini, HouseholdAgent, FirmAgent, BankAgent
import pandas as pd
import plotly.express as px
import numpy as np
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule, TextElement
from mesa.visualization.UserParam import Slider, Checkbox, NumberInput  # Updated import

class GiniElement(TextElement):
    def render(self, model):
        return f"Gini Coefficient: {calculate_gini(model.schedule.agents):.3f}"

class UnemploymentElement(TextElement):
    def render(self, model):
        return f"Unemployment Rate: {model.unemployment_rate:.1%}"

class GDPElement(TextElement):
    def render(self, model):
        return f"GDP: ${model.gdp:,.0f}"

class ActiveFirmsElement(TextElement):
    def render(self, model):
        firms = len([a for a in model.schedule.agents if isinstance(a, FirmAgent)])
        return f"Active Firms: {firms}"

def agent_portrayal(agent):
    portrayal = {}
    if isinstance(agent, HouseholdAgent):
        portrayal = {
            "Shape": "circle",
            "Color": "blue" if agent.is_employed else "red",
            "Filled": "true",
            "Layer": 0,
            "r": 0.5,
            "text": f"{agent.unique_id}<br>Wealth: {agent.wealth:.0f}",
            "text_color": "white"
        }
    elif isinstance(agent, FirmAgent):
        portrayal = {
            "Shape": "rect",
            "Color": "green",
            "Filled": "true",
            "Layer": 0,
            "w": 0.8,
            "h": 0.8,
            "text": f"{agent.unique_id}<br>Capital: {agent.capital:.0f}",
            "text_color": "white"
        }
    elif isinstance(agent, BankAgent):
        portrayal = {
            "Shape": "rect",
            "Color": "yellow",
            "Filled": "true",
            "Layer": 1,
            "w": 1,
            "h": 1,
            "text": "Bank",
            "text_color": "black"
        }
    return portrayal

# Create grid visualization
grid = mesa.visualization.CanvasGrid(agent_portrayal, 20, 20, 500, 500)

# Define charts
gini_chart = ChartModule(
    [{"Label": "Gini", "Color": "purple"}],
    data_collector_name="datacollector"
)

unemployment_chart = ChartModule(
    [{"Label": "Unemployment", "Color": "orange"}],
    data_collector_name="datacollector"
)

gdp_chart = ChartModule(
    [{"Label": "GDP", "Color": "green"}],
    data_collector_name="datacollector"
)

firms_chart = ChartModule(
    [{"Label": "Active_Firms", "Color": "blue"}],
    data_collector_name="datacollector"
)

# Define model parameters using the new parameter system
model_params = {
    "num_households": Slider(
        "Number of Households",
        500,
        100,
        2000,
        100,
        description="Initial number of household agents"
    ),
    "num_firms": Slider(
        "Number of Firms",
        50,
        10,
        200,
        5,
        description="Initial number of firm agents"
    ),
    "initial_interest_rate": Slider(
        "Initial Interest Rate",
        0.03,
        0.0,
        0.1,
        0.005,
        description="Initial central bank interest rate"
    ),
    "productivity_shock": Slider(
        "Productivity Shock",
        1.0,
        0.5,
        1.5,
        0.1,
        description="Multiplier for firm productivity"
    ),
    "confidence_shock": Slider(
        "Consumer Confidence",
        1.0,
        0.5,
        1.5,
        0.1,
        description="Multiplier for consumer confidence"
    )
}

# Text elements
text_elements = [
    GiniElement(),
    UnemploymentElement(),
    GDPElement(),
    ActiveFirmsElement()
]

# Create server
server = ModularServer(
    EconomicModel,
    [grid, *text_elements, gini_chart, unemployment_chart, gdp_chart, firms_chart],
    "Economic Model",
    model_params
)

if __name__ == "__main__":
    server.launch()