# 🌍 AI Travel Designer Agent

An AI agent that plans full travel experiences by combining destination ideas, mock bookings, and exploration tips.

### 🧠 What It Does

- **DestinationAgent**: Recommends travel places
- **BookingAgent**: Mocks flights and hotel bookings
- **ExploreAgent**: Suggests food and attractions

### ⚙️ How It Works

- User provides a travel mood or preference.
- `travel_agent` uses tools like `get_flights()` and `suggest_hotels()` with handoff logic to the right agent.
- Final suggestions are returned to the user.

Built using **OpenAI Agent SDK + Runner** with **Chainlit UI**.

---

👤 **Zubair Ahmed**  
🔗 [linkedin.com/in/zubairdev](https://linkedin.com/in/zubairdev)
