# agent.py
import logging
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env.local")

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
from _langgraph.graph_wrapper import LivekitGraphRunner  # our wrapper that adapts a compiled graph to LiveKit
from _langgraph.graphs.tools_graph import get_compiled_graph

logger = logging.getLogger("voice-agent")

def prewarm(proc: JobProcess):
    """
    Prewarm

    Args:
        proc (JobProcess): The job process.

    Returns:
        None
    
    This method prewarms the VAD model so that it doesn't have a delay when it's first used.
    """
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """
    Entry point

    Args:
        ctx (JobContext): The job context.

    Returns:
        None

    This method connects to a room and starts the voice assistant.
    It's the main entry point for the agent.
    """
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    # Get the compiled graph and create a LiveKitGraphRunner instance, this instance is the one responsible for running the graph.
    # The LiveKitGraphRunner is a wrapper that adapts a compiled graph from LangGraph to be compliant with LiveKit's LLM interface.
    compiled_graph, initial_state = await get_compiled_graph()
    graph_runner = LivekitGraphRunner(compiled_graph, initial_state)
    
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
        """
        On metrics collected

        Args:
            agent_metrics (metrics.AgentMetrics): The agent metrics.

        Returns:
            None

        This method logs the agent metrics and collects usage metrics.
        """
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    # Start the agent and say the welcome message.
    agent.start(ctx.room, participant)
    await agent.say("Hey, how can I help you today?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
