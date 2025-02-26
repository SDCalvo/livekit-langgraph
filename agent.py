# agent.py
import logging
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia, deepgram, silero, turn_detector
from livekit.agents import llm
from _langgraph.graph_wrapper import LivekitGraphRunner  # our wrapper that adapts a compiled graph to LiveKit
from _langgraph.simple_graph import get_compiled_graph  # Import the compiled graph from simple_graph.py

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    # Instead of using the original langchain LLM (lc), we wrap our compiled graph.
    compiled_graph = await get_compiled_graph()
    graph_runner = LivekitGraphRunner(compiled_graph)
    
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=graph_runner,  # using our wrapped LangGraph for inference
        tts=cartesia.TTS(),
        turn_detector=turn_detector.EOUModel(),
        min_endpointing_delay=0.5,
        max_endpointing_delay=5.0,
    )

    usage_collector = metrics.UsageCollector()
    @agent.on("metrics_collected")
    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    agent.start(ctx.room, participant)
    await agent.say("Hey, how can I help you today?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
