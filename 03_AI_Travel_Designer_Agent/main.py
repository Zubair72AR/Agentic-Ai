import os
import chainlit as cl
from dotenv import load_dotenv
from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel

# 🔐 Load environment variables
load_dotenv()


def setup_agents():
    external_client = AsyncOpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-1.5-flash",
        openai_client=external_client,
    )

    config = RunConfig(model=model, model_provider=external_client)

    # ✈️ Destination Suggestion Agent
    destination_agent = Agent(
        name="destination_agent",
        instructions="Suggest travel destinations based on user's mood or interests.",
        handoff_description="Suggests places to go.",
        model=model,
    )

    # 🏨 Booking Agent with mock tools
    booking_agent = Agent(
        name="booking_agent",
        instructions="Simulate travel bookings using get_flights() and suggest_hotels().",
        handoff_description="Handles travel & hotel bookings.",
        model=model,
        tools=[
            {
                "tool_name": "get_flights",
                "tool_description": "Finds available flights based on user query.",
                "tool": lambda input: f"✈️ Mock flight booked for: {input}"
            },
            {
                "tool_name": "suggest_hotels",
                "tool_description": "Suggests hotels for a given location.",
                "tool": lambda input: f"🏨 Suggested mock hotels in {input}: Hotel A, Hotel B"
            },
        ]
    )

    # 🧭 Explore Agent for food & activities
    explore_agent = Agent(
        name="explore_agent",
        instructions="Suggest local attractions, food, and experiences for a given destination.",
        handoff_description="Suggests what to do and eat.",
        model=model,
    )

    # 🧠 Triage/Main Travel Agent
    travel_agent = Agent(
        name="travel_agent",
        instructions=(
            "You are a travel planner AI. Use tools to suggest destinations, book travel, and explore experiences."
            "You must call the proper agents; never respond directly yourself."
        ),
        tools=[
            destination_agent.as_tool(
                "suggest_destination", "Suggest a destination."),
            booking_agent.as_tool(
                "handle_booking", "Book flights and hotels."),
            explore_agent.as_tool("suggest_experiences",
                                  "Suggest attractions and food."),
        ],
        model=model,
    )

    return travel_agent, config

# 🚀 Start chat


@cl.on_chat_start
async def start():
    agent, config = setup_agents()
    cl.user_session.set("agent", agent)
    cl.user_session.set("config", config)
    await cl.Message("🌍 Welcome to the AI Travel Designer! Tell me where or how you'd like to travel.").send()

# 📩 Handle user message


@cl.on_message
async def handle(message: cl.Message):
    msg = cl.Message("✈️ Planning your trip...")
    await msg.send()

    agent = cl.user_session.get("agent")
    config = cl.user_session.get("config")

    result = await Runner.run(agent, [{"role": "user", "content": message.content}], run_config=config)

    msg.content = result.final_output
    await msg.update()
