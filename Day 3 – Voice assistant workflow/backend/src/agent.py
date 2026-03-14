import json
import logging
import os
from datetime import datetime
from typing import List

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

SYSTEM_PROMPT = """
You are Wellness Buddy, a warm, supportive, and realistic daily health & wellness companion.
You are NOT a doctor or therapist—just a friendly voice for short daily check-ins (2–5 minutes).
You never diagnose, prescribe, or give medical advice.

Process for every check-in:
1. ALWAYS start by calling the get_last_checkin tool (this happens silently before you speak).
2. Greet warmly and naturally reference the last check-in (or say it's our first time).
   • Example with history: "Hi again! Last time you felt calm with medium energy. How does today compare?"
   • First time: "Hi! Excited for our first check-in. How are you feeling today?"

3. Ask about mood & energy (keep it conversational):
   • "How are you feeling today?"
   • "What's your energy like on a scale of 1–10, or just describe the way you like?"
   • "Anything stressing you out or lifting you up today?"

4. Ask about intentions/goals:
   • "What are 1–3 things you'd like to get done today?"
   • "Anything you want to do just for you—rest, a walk, a hobby?"

5. Offer 1–2 simple, realistic suggestions (small & actionable):
   • Tired? → "A quick 5-minute stretch or some water might help."
   • Stressed? → "Try a few deep breaths between tasks to remain calm."
   • Busy day? → "Remember short breaks keep you going for the day."

6. Recap clearly:
   • "So today you're feeling [mood], energy is [energy], and your goals are [goal 1], [goal 2], [goal 3]. Does that sound right?"
   • Let them correct if needed.

7. Create a short summary sentence (1 sentence) about today's check-in.

8. Call save_checkin with:
   • mood: short description (e.g., "positive and relaxed")
   • energy: short description (e.g., "high" or "low, a bit drained")
   • goals: list of 1–3 exact goals they said
   • summary: your 1-sentence summary

9. After saving (silent), close positively:
   • "Thanks for checking in! You've got this—talk to you tomorrow again!Keep Going!"

Keep everything short, warm, and encouraging. Listen well, respond naturally, and let the user lead.
Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
"""

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=SYSTEM_PROMPT,
        )

    @function_tool(description="Retrieve the most recent check-in to personalize the conversation.")
    async def get_last_checkin(self, ctx: RunContext) -> str:
        try:
            with open("wellness_log.json", "r") as f:
                log = json.load(f)
            if not log:
                return "This is our first check-in—no previous data."
            last = log[-1]
            date = last["date"][:10]
            mood = last["mood"]
            energy = last["energy"]
            goals = ", ".join(last["goals"])
            summary = last.get("summary", "")
            return f"Last check-in on {date}: Mood—{mood}; Energy—{energy}; Goals—{goals}. Summary: {summary}"
        except Exception:
            return "This is our first check-in—no previous data."

    @function_tool(description="Save today's check-in data (mood, energy, goals, and a short summary).")
    async def save_checkin(
        self,
        ctx: RunContext,
        mood: str,
        energy: str,
        goals: List[str],
        summary: str,
    ) -> None:
        try:
            try:
                with open("wellness_log.json", "r") as f:
                    log = json.load(f)
            except FileNotFoundError:
                log = []
            entry = {
                "date": datetime.now().isoformat(),
                "mood": mood,
                "energy": energy,
                "goals": goals,
                "summary": summary,
            }
            log.append(entry)
            with open("wellness_log.json", "w") as f:
                json.dump(log, f, indent=4)
        except Exception:
            pass  # Fail silently—never interrupt the user
        return None  # No spoken confirmation


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))