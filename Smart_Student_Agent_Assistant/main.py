from agents import Agent, RunConfig, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from dotenv import load_dotenv
import os


# 🔐 Load API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 🤖 Gemini API Setup
external_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# 🧠 Create Agent
agent = Agent(
    name="Smart Student Assistant",
    instructions="You are a helpful assistant who answers academic questions, gives study tips, and summarizes passages.",
    model=model
)

# 📚 Menu


def run_assistant():
    while True:
        print("\n📌 What would you like to do?")
        print("1. Answer academic questions")
        print("2. Get study tips")
        print("3. Summarize a passage")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ")

        if choice == "1":
            question = input("\n🎓 Enter your academic question: ")
            result = Runner.run_sync(agent, input=question, run_config=config)
            print("\n🤖 Answer:\n", result.final_output)

        elif choice == "2":
            result = Runner.run_sync(
                agent, input="Give me 5 helpful study tips.", run_config=config)
            print("\n📘 Study Tips:\n", result.final_output)

        elif choice == "3":
            passage = input("\n📄 Paste the passage you want summarized:\n")
            result = Runner.run_sync(
                agent, input=f"Summarize this passage: {passage}", run_config=config)
            print("\n📌 Summary:\n", result.final_output)

        elif choice == "4":
            print("\n👋 Bye! Study well.")
            break
        else:
            print("❌ Invalid choice. Please try again.")


# ✅ Run App
if __name__ == "__main__":
    run_assistant()
