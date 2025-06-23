#!/usr/bin/env python3
"""
Simple simulation runner for CheckmateModelManager.
Runs conversations with a Checkmate agent via WebSocket.
"""

import asyncio
import json
import logging
import os
import datetime
import uuid
from typing import Dict, List
from dotenv import load_dotenv

# Import the CheckmateModelManager
from CheckmateModelManager import CheckmateModelManager

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleCheckmateSimulation:
    def __init__(self):
        self.transcript = []
        self.simulation_complete = False
        self.run_id = str(uuid.uuid4())

    def run_simulation(
        self,
        user_objective: str = "I want to schedule a meeting",
        websocket_endpoint: str = None,
        simulation_prompt: str = None,
        duration_minutes: int = 5,
    ):
        """
        Run a simple simulation using CheckmateModelManager
        """
        logger.info("Starting Checkmate simulation...")

        # Validate required environment variables
        required_vars = [
            "OPENAI_API_KEY",
            "ELEVENLABS_API_KEY",
            "DEEPGRAM_API_KEY",
        ]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

        # Validate websocket endpoint
        if not websocket_endpoint:
            raise ValueError("WEBSOCKET_ENDPOINT is required")

        # Create the CheckmateModelManager
        manager = CheckmateModelManager(self.run_id)

        # Configure the manager
        config = {
            "websocket_endpoint": websocket_endpoint,
            "simulation_prompt": simulation_prompt or "You are a helpful assistant in a voice conversation.",
        }

        try:
            # Start the manager
            logger.info("Initializing CheckmateModelManager...")
            manager._start(config)

            # Run the simulation
            logger.info(f"Running simulation with objective: {user_objective}")
            message_history = manager.run_simulation(user_objective)

            if message_history:
                self.transcript = message_history
                logger.info(f"Simulation completed successfully with {len(message_history)} messages")
            else:
                logger.error("Simulation failed - no message history returned")

        except Exception as e:
            logger.error(f"Simulation error: {e}")
            raise
        finally:
            # Cleanup
            try:
                manager._cleanup()
            except Exception as e:
                logger.warning(f"Cleanup error: {e}")

        return self.transcript

    def save_results(self):
        """Save simulation results to files"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save transcript
        transcript_file = f"checkmate_simulation_transcript_{timestamp}.json"
        with open(transcript_file, "w") as f:
            json.dump(self.transcript, f, indent=2)
        logger.info(f"Transcript saved to: {transcript_file}")

        # Create a summary file
        summary_file = f"checkmate_simulation_summary_{timestamp}.txt"
        with open(summary_file, "w") as f:
            f.write("Checkmate Simulation Summary\n")
            f.write("=" * 30 + "\n")
            f.write(f"Run ID: {self.run_id}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total messages: {len(self.transcript)}\n")
            f.write(f"Simulation completed: {self.simulation_complete}\n\n")
            
            f.write("Message History:\n")
            f.write("-" * 20 + "\n")
            for i, message in enumerate(self.transcript):
                role = message.get("role", "unknown")
                content = message.get("content", "")
                f.write(f"{i+1}. [{role.upper()}] {content}\n")
        
        logger.info(f"Summary saved to: {summary_file}")


def main():
    """Main entry point for the simulation"""
    simulation = SimpleCheckmateSimulation()

    # Get websocket endpoint from environment or use default
    websocket_endpoint = os.getenv("WEBSOCKET_ENDPOINT")
    if not websocket_endpoint:
        logger.error("WEBSOCKET_ENDPOINT environment variable is required")
        logger.info("Example: WEBSOCKET_ENDPOINT=wss://your-checkmate-server.com/ws")
        return

    # Run simulation with custom parameters
    transcript = simulation.run_simulation(
        user_objective="I want a five dollar meal",
        websocket_endpoint=websocket_endpoint,
        simulation_prompt="You are a customer at a fast food restaurant. Be natural and conversational.",
        duration_minutes=5,
    )

    # Save results
    simulation.save_results()

    print("\n" + "=" * 50)
    print("CHECKMATE SIMULATION COMPLETE")
    print("=" * 50)
    print(f"Transcript entries: {len(transcript)}")
    print("Check the generated files for full results.")


if __name__ == "__main__":
    # Simple CLI for running simulations
    import argparse

    parser = argparse.ArgumentParser(description="Run Checkmate Agent simulation")
    parser.add_argument(
        "--objective",
        default="I want a five dollar meal",
        help="User objective for the simulation",
    )
    parser.add_argument(
        "--duration", type=int, default=5, help="Simulation duration in minutes"
    )
    parser.add_argument(
        "--websocket-endpoint", 
        help="WebSocket endpoint for Checkmate server (overrides WEBSOCKET_ENDPOINT env var)"
    )
    parser.add_argument(
        "--simulation-prompt", 
        help="Custom simulation prompt for the assistant"
    )

    args = parser.parse_args()

    # Override websocket endpoint if provided
    if args.websocket_endpoint:
        os.environ["WEBSOCKET_ENDPOINT"] = args.websocket_endpoint

    # Run the simulation
    main()
