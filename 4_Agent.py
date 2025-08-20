from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent,AgentExecutor
from langchain import hub
from dotenv import load_dotenv
import os
os.environ["LANGCHAIN_PROJECT"]="ReAct Agent"
load_dotenv()

# Tool 1 : Web Search
search_tool = DuckDuckGoSearchRun()

# Tool 2 : Weather API
# @tool
# def get_weather_data(city : str)->str:
#     """
#     This function fetches the current weather data for a given city.
#         """
#     url = f'https://api.weatherstack.com/current?access_key=a57ebcc76af93af6285ba1d003101bb2&query={city}'
#     response = requests.get(url)
#     return response.json()

@tool
def get_weather_data(city: str) -> str:
    """
    Fetches current weather for a given city using Open-Meteo (free, no API key required).
    """
    try:
        def fetch_city(c):
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={c}&count=1"
            return requests.get(geo_url).json()

        geo_res = fetch_city(city)

        # if not found, try appending country "India"
        if "results" not in geo_res or len(geo_res["results"]) == 0:
            geo_res = fetch_city(f"{city}, India")
            if "results" not in geo_res or len(geo_res["results"]) == 0:
                return "City not found."

        lat = geo_res["results"][0]["latitude"]
        lon = geo_res["results"][0]["longitude"]

        # Get weather
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        )
        weather_res = requests.get(weather_url).json()

        current = weather_res.get("current_weather", {})
        if not current:
            return "Weather data not available."

        return f"Temperature: {current['temperature']}°C, Wind: {current['windspeed']} km/h"

    except Exception as e:
        return f"Error fetching weather: {str(e)}"


# use Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0)

# Step 2: Pull the ReAct Prompt from langchain Hub
prompt = hub.pull("hwchase17/react")

# Step 3: Create the ReAct Agent

agent = create_react_agent(
    llm=llm,
    tools = [search_tool,get_weather_data],
    prompt=prompt,
)

# Step 4: Wrap in executor
agent_executor = AgentExecutor(
    agent = agent,
    tools=[search_tool,get_weather_data],
    verbose=True,
    max_iterations=5
)
# Step 5: Run the agent
result = agent_executor.invoke({"input":"What is the temperature of Delhi?"})
print(result)
print(result['output'])