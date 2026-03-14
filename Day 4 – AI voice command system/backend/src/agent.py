import logging
import json
import os
import asyncio
from typing import Annotated, Literal, Optional
from dataclasses import dataclass

print("\n" + "🧬" * 50)
print("Computer Science Specialist - Tutor")
print("💡 agent.py LOADED SUCCESSFULLY!")
print("CS" * 50 + "\n")

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

# 🔌 PLUGINS
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# 🆕 Renamed file so it generates fresh data for you
CONTENT_FILE = "cs_content.json" 

DEFAULT_CONTENT = [
  {
    "id": "Network",
    "title": "Network",
    "summary": "a group of interconnected devices that communicate and share resources, such as data, printers, and applications",
    "sample_question": "Give example of a Network?"
  },
  {
    "id": "LSTM",
    "title": "LSTM",
    "summary": "A Long short-term memory (LSTM) is a type of Recurrent Neural Network specially designed to prevent the neural network output for a given input from either decaying or exploding as it cycles through the feedback loops",
    "sample_question": "What is the full form of LSTM?"
  },
  {
    "id": "LLM",
    "title": "LLM",
    "summary": "LLM stands for Large Language Model, a type of artificial intelligence that understands and generates human language by learning from massive datasets of text",
    "sample_question": "What is the full form of LLM?"
  }
]

def load_content():
    """
    📖 Checks if CS JSON exists. 
    If NO: Generates it from DEFAULT_CONTENT.
    If YES: Loads it.
    """
    try:
        path = os.path.join(os.path.dirname(__file__), CONTENT_FILE)
        
        # Check if file exists
        if not os.path.exists(path):
            print(f"⚠️ {CONTENT_FILE} not found. Generating CS data...")
            with open(path, "w", encoding='utf-8') as f:
                json.dump(DEFAULT_CONTENT, f, indent=4)
            print("✅ CS content file created successfully.")
            
        # Read the file
        with open(path, "r", encoding='utf-8') as f:
            data = json.load(f)
            # Validate: Ensure required keys
            validated = [item for item in data if all(k in item for k in ['id', 'title', 'summary', 'sample_question'])]
            if len(validated) != len(data):
                print("⚠️ Some content entries invalid; using validated subset.")
            return validated
            
    except Exception as e:
        print(f"⚠️ Error managing content file: {e}")
        return DEFAULT_CONTENT  # Fallback to default

# Load data immediately on startup
COURSE_CONTENT = load_content()

# ======================================================
# 🧠 STATE MANAGEMENT
# ======================================================

@dataclass
class TutorState:
    """🧠 Tracks the current learning context"""
    current_topic_id: str | None = None
    current_topic_data: dict = Field(default_factory=dict)  # 🆕 Safe default: {} not None
    mode: Literal["learn", "quiz", "teach_back"] = "learn"
    
    def set_topic(self, topic_id: str) -> bool:
        # 🆕 Case-insensitive match
        topic = next((item for item in COURSE_CONTENT if item["id"].lower() == topic_id.lower()), None)
        if topic:
            self.current_topic_id = topic["id"]  # Preserve original casing
            self.current_topic_data = topic
            logger.info(f"Topic set: {topic['title']}")
            return True
        logger.warning(f"Topic not found: {topic_id}")
        return False

@dataclass
class Userdata:
    tutor_state: TutorState
    agent_session: Optional[AgentSession] = None 

# ======================================================
# 🛠️ TUTOR TOOLS
# ======================================================

@function_tool
async def select_topic(
    ctx: RunContext[Userdata], 
    topic_id: Annotated[str, Field(description="The ID of the topic to study (e.g., 'Network', 'LSTM', 'LLM')")]
) -> str:
    """📚 Selects a topic to study from the available list."""
    state = ctx.userdata.tutor_state
    success = state.set_topic(topic_id)
    
    if success:
        return f"Topic set to {state.current_topic_data['title']}. Now suggest modes: 'learn' to explain, 'quiz' to test, or 'teach_back' to practice explaining."
    else:
        available = ", ".join([t["id"] for t in COURSE_CONTENT])
        return f"Topic '{topic_id}' not found. Available: {available}. Try again."

@function_tool
async def set_learning_mode(
    ctx: RunContext[Userdata], 
    mode: Annotated[str, Field(description="The mode to switch to: 'learn', 'quiz', or 'teach_back'")]
) -> str:
    """🔄 Switches the interaction mode and updates the agent's voice/persona."""
    
    # 1. Validate & Update State
    state = ctx.userdata.tutor_state
    mode = mode.lower()
    if mode not in ["learn", "quiz", "teach_back"]:
        return "Invalid mode. Use 'learn', 'quiz', or 'teach_back'."
    
    if not state.current_topic_data:
        return "No topic selected. Use `select_topic` first (e.g., 'LSTM')."  # 🆕 Prevents NoneType crash
    
    state.mode = mode
    
    # 2. Switch Voice & Instruction
    agent_session = ctx.userdata.agent_session 
    instruction = ""  # Default
    
    if agent_session:
        try:
            if state.mode == "learn":
                # 👨‍🏫 MATTHEW: The Lecturer
                await agent_session.tts.update_options(voice="en-US-matthew", style="Promo")
                instruction = f"Mode: LEARN. Explain the topic '{state.current_topic_data['title']}' using this summary: {state.current_topic_data['summary']}. Be clear and engaging."
                
            elif state.mode == "quiz":
                # 👩‍🏫 ALICIA: The Examiner
                await agent_session.tts.update_options(voice="en-US-alicia", style="Conversational")
                instruction = f"Mode: QUIZ on '{state.current_topic_data['title']}'. Ask: {state.current_topic_data['sample_question']}. Wait for answer, then evaluate briefly."
                
            elif state.mode == "teach_back":
                # 👨‍🎓 KEN: The Student/Coach
                await agent_session.tts.update_options(voice="en-US-ken", style="Promo")
                instruction = f"Mode: TEACH_BACK on '{state.current_topic_data['title']}'. Pretend you're a beginner student. Ask the user to explain the concept to you step-by-step."
            
            # 🆕 Apply to LLM (dynamic system prompt update)
            # Note: Adjust if your LiveKit version uses different API
            await ctx.update_agent_instructions(instruction)  # Or: await agent_session.llm.update_system_prompt(instruction)
            
        except Exception as e:
            logger.error(f"TTS update failed: {e}")
            instruction = "Voice switch failed—continuing in text mode."
    else:
        instruction = "Session not ready—retry after startup."

    print(f"🔄 SWITCHING MODE -> {state.mode.upper()} | Topic: {state.current_topic_data.get('title', 'None')}")
    return f"✅ Switched to {state.mode} mode for '{state.current_topic_data['title']}'. {instruction}"

@function_tool
async def evaluate_teaching(
    ctx: RunContext[Userdata],
    user_explanation: Annotated[str, Field(description="The explanation given by the user during teach-back")]
) -> str:
    """📝 Call this when the user has finished explaining a concept in 'teach_back' mode."""
    if not ctx.userdata.tutor_state.current_topic_data:
        return "No topic context—select one first."
    
    print(f"📝 EVALUATING EXPLANATION: {user_explanation[:100]}...")  # Truncate for logs
    summary = ctx.userdata.tutor_state.current_topic_data['summary']
    return f"Analyze '{user_explanation}' against summary: '{summary}'. Score 1-10 on accuracy/clarity. Correct errors gently. Suggest improvements."

# ======================================================
# 🧠 AGENT DEFINITION
# ======================================================

class TutorAgent(Agent):
    def __init__(self):
        # Generate list of topics for the prompt
        topic_list = ", ".join([f"{t['id']} ({t['title']})" for t in COURSE_CONTENT])
        
        super().__init__(
            instructions=f"""
            You are a CS Tutor helping master concepts like Networks, LSTMs, and LLMs.
            
            📚 **AVAILABLE TOPICS:** {topic_list}
            
            🔄 **MODES (Use tools to switch):**
            1. **LEARN (Matthew voice):** Explain clearly from summary.
            2. **QUIZ (Alicia voice):** Test with sample question.
            3. **TEACH_BACK (Ken voice):** Act as student; get user to explain.
            
            ⚙️ **FLOW:**
            - Greet & ask for topic (use `select_topic` if mentioned).
            - After topic: Suggest modes.
            - Call `set_learning_mode` on user request (e.g., "quiz me").
            - In teach_back: Listen, then `evaluate_teaching` with their words.
            - Always confirm actions & keep engaging!
            """,
            tools=[select_topic, set_learning_mode, evaluate_teaching],
        )

# ======================================================
# 🎬 ENTRYPOINT
# ======================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    print("\n" + "🧬" * 25)
    print("🚀 STARTING CS TUTOR SESSION")
    print(f"📚 Loaded {len(COURSE_CONTENT)} topics from Knowledge Base")
    
    # 1. Initialize State
    userdata = Userdata(tutor_state=TutorState())

    # 2. Setup Agent
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew", 
            style="Promo",        
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )
    
    # 3. Store session in userdata for tools to access
    userdata.agent_session = session
    
    # 4. Start
    await session.start(
        agent=TutorAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))