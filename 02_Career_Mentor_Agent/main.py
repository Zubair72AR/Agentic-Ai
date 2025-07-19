import os
import chainlit as cl
from dotenv import load_dotenv
from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel

# Load env
load_dotenv()

# Setup model and agents


def setup_agents():
    external_client = AsyncOpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client,
    )

    config = RunConfig(model=model, model_provider=external_client)

    # 🧭 Career Suggestion Agent
    career_agent = Agent(
        name="career_agent",
        instructions="Suggest career fields based on user's interests.",
        handoff_description="Recommends career fields.",
        model=model,
    )

    # 🛠 Skill Roadmap Agent
    skill_agent = Agent(
        name="skill_agent",
        instructions="Generate a skill roadmap for the given career field.",
        handoff_description="Provides skill-building plan.",
        model=model,
    )

    # 💼 Job Roles Agent
    job_agent = Agent(
        name="job_agent",
        instructions="Share job roles and opportunities related to a specific field.",
        handoff_description="Explains real-world job options.",
        model=model,
    )

    # 🎯 Triage/Main Agent
    mentor_agent = Agent(
        name="mentor_agent",
        instructions=(
            "You are a career mentor AI. Based on user input, delegate tasks to the proper agents."
            "Use only the provided tools; don't answer directly."
        ),
        tools=[
            career_agent.as_tool("suggest_careers", "Suggest career paths."),
            skill_agent.as_tool("get_career_roadmap", "Show required skills."),
            job_agent.as_tool("show_job_roles", "List related job roles."),
        ],
        model=model,
    )

    return mentor_agent, config

# 🌟 Chainlit startup


@cl.on_chat_start
async def start():
    agent, config = setup_agents()
    cl.user_session.set("agent", agent)
    cl.user_session.set("config", config)
    await cl.Message("🎓 Welcome to Career Mentor AI! Ask me about careers.").send()

# 💬 Handle messages


@cl.on_message
async def handle(message: cl.Message):
    msg = cl.Message("🔍 Let me think...")
    await msg.send()

    agent = cl.user_session.get("agent")
    config = cl.user_session.get("config")

    result = await Runner.run(agent, [{"role": "user", "content": message.content}], run_config=config)

    msg.content = result.final_output
    await msg.update()
