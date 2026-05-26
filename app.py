import os
import certifi
import requests
import streamlit as st

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain import hub
from langchain.tools import tool

from langchain.agents import (
    create_react_agent,
    AgentExecutor
)

# =========================
# LOAD ENV VARIABLES
# =========================

os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

# Streamlit page config

st.set_page_config(
    page_title="AI Agent App",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Agent with Weather + Search")
st.markdown("search + weather ai agent using langchain")

# =========================
# SEARCH TOOL
# =========================

search_tool = TavilySearchResults(
    tavily_api_key=TAVILY_API_KEY
)

# =========================
# WEATHER TOOL
# =========================

@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city
    """

    url = (
        f"http://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response = requests.get(url)

    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )

# =========================
# LLM
# =========================

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=OPENAI_API_KEY
)

# =========================
# PROMPT
# =========================

prompt = hub.pull("hwchase17/react")

# =========================
# TOOLS
# =========================

tools = [search_tool, get_weather_data]

# =========================
# CREATE AGENT
# =========================

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# =========================
# AGENT EXECUTOR
# =========================

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# =========================
# STREAMLIT UI
# =========================

user_input = st.text_area(
    "Enter your query:",
    height=120,
    placeholder="Example: Find the capital of India and then find its current weather."
)

if st.button("Run Agent"):

    if user_input.strip() == "":
        st.warning("Please enter a query.")
    else:

        with st.spinner("Agent is thinking..."):

            try:

                response = agent_executor.invoke({
                    "input": user_input
                })

                st.success("Done!")

                st.subheader("Response")
                st.write(response["output"])

            except Exception as e:
                st.error(f"Error: {str(e)}")