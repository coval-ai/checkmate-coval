from pathlib import Path
import sys
import json
from dotenv import load_dotenv
from typing import Dict, List
from llm_clients.openai_client import CovalOpenAIClient
from elevenlabs import VoiceSettings, ElevenLabs
from CovalError import SimulationInitialSetupError, SimulationAPIError
import websockets
import asyncio
import uuid
import os
from deepgram import DeepgramClient, PrerecordedOptions
import io
import wave
from StatusCodes import StatusCode, Status
from dataclasses import dataclass
from typing import Optional

from models.ModelManager import ModelManager

CHECKMATE = "checkmate"

load_dotenv()


@dataclass
class SimulationResponse:
    message_history: List[dict]
    status: Status


@dataclass
class AudioState:
    user_is_speaking: bool = False
    agent_is_speaking: bool = False


class CheckmateModelManager(ModelManager):
    def __init__(self, run_id):
        super().__init__(run_id)
        self.type = CHECKMATE
        self.ready = False
        self.openai_client = None
        self.websocket = None
        self.websocket_endpoint = None
        self.simulation_prompt = None
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.elevenlabs_client = ElevenLabs(api_key=self.elevenlabs_api_key)
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        self.deepgram_client = DeepgramClient(self.deepgram_api_key)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def _start(self, config):
        """Initialize the Checkmate manager and OpenAI client"""
        try:
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] Starting CheckmateModelManager initialization")
            
            if "websocket_endpoint" not in config:
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] ERROR: Missing websocket_endpoint in config")
                raise SimulationInitialSetupError("Missing required websocket_endpoint in config")
            
            self.websocket_endpoint = config["websocket_endpoint"]
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] Original websocket endpoint: {self.websocket_endpoint}")
            
            connection_uuid = str(uuid.uuid4())
            # checkmate session handling
            self.websocket_endpoint = f"{self.websocket_endpoint}?from=coval&to=12029334007&uuid={connection_uuid}"
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] Final websocket endpoint: {self.websocket_endpoint}")

            async def connect_websocket():
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] Attempting websocket connection...")
                self.websocket = await websockets.connect(
                    self.websocket_endpoint,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10,
                )
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] SUCCESS: Connected to websocket")

                # Send initial handshake message
                initial_message = {
                    "event": "websocket:connected",
                    "content-type": "audio/l8;rate=8000",
                }
                await self.websocket.send(json.dumps(initial_message))
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] SUCCESS: Sent initial handshake message")
                await asyncio.sleep(1)  # Allow server to process handshake

            self.loop.run_until_complete(connect_websocket())

            if not self.websocket:
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] ERROR: WebSocket connection failed")
                raise SimulationInitialSetupError("WebSocket connection failed")

            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] Initializing OpenAI client...")
            self.openai_client = CovalOpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), manager=self)
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] OpenAI client initialized successfully")

            if "simulation_prompt" not in config:
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] ERROR: Missing simulation_prompt in config")
                raise SimulationInitialSetupError("Missing required simulation_prompt in config")
            
            self.simulation_prompt = config["simulation_prompt"]
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] Simulation prompt loaded: {self.simulation_prompt[:100]}...")

            self.ready = True
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] SUCCESS: CheckmateModelManager fully initialized and ready")

        except Exception as e:
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] ERROR: Failed to start: {str(e)}")
            raise SimulationInitialSetupError(f"Failed to start: {str(e)}")

    def _get_system_prompt(self, input_str: str) -> str:
        """Return the system prompt for the Checkmate conversation."""
        # Check if input_str is already a conversation history
        try:
            parsed_input = json.loads(input_str)
            if isinstance(parsed_input, list) and all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in parsed_input):
                # Input is conversation history, extract the objective from the context
                objective = "Follow this transcript as closely as possible. You are the USER."
            else:
                objective = input_str
        except (json.JSONDecodeError, TypeError):
            objective = input_str
            
        return f"""
        You are a USER talking to an assistant in a voice conversation.
        You are explicitly ROLE-PLAYING as a USER. Under no circumstances do you act as an agent or assistant.
        
        Additional context:
        {self.simulation_prompt if self.simulation_prompt else ""}
        
        Your objective is to: {objective}
        
        You must follow these rules:
        1. Respond naturally as a user would in a voice conversation
        2. Keep responses concise and conversational (1-2 sentences typically)
        3. Ask questions when appropriate to continue the conversation
        4. Provide relevant information when asked
        5. If the conversation has naturally ended, trigger the function call end_conversation with a suitable closing response
        6. Do not offer help or assistance unless specifically asked
        7. If given options or choices, select one and explain your preference
        8. Be polite and cooperative but maintain your role as a user
        
        This is the conversation thus far (you are the user):
        """

    def _stt(self, audio_data: bytes) -> str:
        """Convert audio data to text using Deepgram's file-based transcription"""
        try:
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] STT: Starting audio transcription, input size: {len(audio_data)} bytes")
            
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_data)

            wav_buffer.seek(0)
            wav_data = wav_buffer.read()
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] STT: Created WAV buffer, size: {len(wav_data)} bytes")

            payload = {
                "buffer": wav_data,
                "mimetype": "audio/wav",
            }

            options = PrerecordedOptions(
                model="nova-2",
                smart_format=True,
            )

            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] STT: Sending to Deepgram API...")
            response = self.deepgram_client.listen.rest.v("1").transcribe_file(payload, options)
            
            if response and response.results and response.results.channels:
                transcript = response.results.channels[0].alternatives[0].transcript
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] STT: SUCCESS - Transcript: '{transcript.strip()}'")
                return transcript.strip()
            else:
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] STT: WARNING - No transcript in response")
                return ""

        except Exception as e:
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] STT: ERROR - Failed to convert audio to text: {e}")
            raise SimulationAPIError(f"Failed to convert audio to text: {str(e)}")

    def _tts(self, text: str) -> Optional[List[bytes]]:
        """Convert text to speech using ElevenLabs and return audio data in 20ms chunks"""
        try:
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] TTS: Starting text-to-speech conversion for: '{text}'")
            
            response = self.elevenlabs_client.text_to_speech.convert(
                voice_id="iP95p4xoKVk53GoZ742B",
                output_format="pcm_16000",
                text=text,
                model_id="eleven_multilingual_v2",
                voice_settings=VoiceSettings(
                    stability=0.0,
                    similarity_boost=1.0,
                    style=0.0,
                    use_speaker_boost=True,
                ),
            )

            audio_data = b""
            chunk_count = 0
            for chunk in response:
                if chunk:
                    audio_data += chunk
                    chunk_count += 1

            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] TTS: Received {chunk_count} chunks, total audio size: {len(audio_data)} bytes")

            if not audio_data:
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] TTS: ERROR - No audio data received from text-to-speech conversion")
                return None

            # Split into 20ms chunks (640 bytes for 16kHz PCM-16)
            CHUNK_SIZE = 640  # 16000 Hz * 2 bytes * 0.02 seconds
            chunks = [audio_data[i : i + CHUNK_SIZE] for i in range(0, len(audio_data), CHUNK_SIZE)]

            # Pad the last chunk with silence if needed
            if len(chunks[-1]) < CHUNK_SIZE:
                original_last_chunk_size = len(chunks[-1])
                chunks[-1] = chunks[-1] + b"\x00" * (CHUNK_SIZE - len(chunks[-1]))
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] TTS: Padded last chunk from {original_last_chunk_size} to {CHUNK_SIZE} bytes")

            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] TTS: SUCCESS - Created {len(chunks)} audio chunks for transmission")
            return chunks

        except Exception as e:
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] TTS: ERROR - Failed to convert text to speech: {e}")
            raise

    def _get_available_functions(self) -> List[Dict]:
        """Return the available functions for the conversation."""
        return [
            {
                "name": "end_conversation",
                "description": "End the conversation when conversation with the agent is over.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "response": {
                            "type": "string",
                            "description": "The final response to end the conversation",
                        }
                    },
                    "required": ["response"],
                },
            }
        ]

    def _get_user_response(self, messages: List[Dict], functions: List[Dict]) -> Dict:
        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] OPENAI: Getting user response, message count: {len(messages)}")
        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] OPENAI: Last message: {messages[-1]['content'][:100] if messages else 'None'}...")
        
        openai_response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            functions=functions,
            function_call="auto",
        )

        simulated_agent_response = openai_response.choices[0].message.content or ""
        simulated_agent_message = {
            "role": (
                "user"
                if openai_response.choices[0].message.role == "assistant"
                else openai_response.choices[0].message.role
            ),
            "content": simulated_agent_response,
        }

        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] OPENAI: SUCCESS - Generated user response: '{simulated_agent_response}'")

        # Check for end conversation function call
        if (
            openai_response.choices[0].message.function_call
            and openai_response.choices[0].message.function_call.name == "end_conversation"
        ):
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] OPENAI: Function call detected - ending conversation")
            try:
                final_response = json.loads(openai_response.choices[0].message.function_call.arguments).get(
                    "response", ""
                )
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] OPENAI: Final response: '{final_response}'")
            except (json.JSONDecodeError, AttributeError) as e:
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] OPENAI: ERROR - Failed to parse function call arguments: {e}")
                final_response = ""

            return {
                "message": simulated_agent_message,
                "ended": True,
                "final_message": ({"role": "user", "content": final_response} if final_response else None),
            }

        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] OPENAI: Continuing conversation")
        return {
            "message": simulated_agent_message,
            "ended": False,
            "final_message": None,
        }

    def _format_messages_for_openai(self, message_history: List[Dict]) -> List[Dict]:
        """Format message history for OpenAI API."""
        return [
            {
                "role": x["role"],
                "content": x["content"],
            }
            for x in message_history
            if x["role"] in ["user", "assistant"]
        ]

    async def _handle_audio_stream(self, audio_state: AudioState, input_str: str, message_history: List[Dict]) -> None:
        """Manage the audio conversation with proper send/receive coordination"""
        try:
            # Start with user initiating the conversation
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: Starting conversation - user will speak first")
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: Input string: {input_str[:200]}...")
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: Initial message history length: {len(message_history)}")
            
            # Get initial user response
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: Getting initial user response...")
            user_response = self._get_user_response(
                [{"role": "system", "content": self._get_system_prompt(input_str)}]
                + self._format_messages_for_openai(message_history),
                self._get_available_functions(),
            )

            if user_response["ended"]:
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: Initial response indicates conversation should end")
                if user_response["final_message"]:
                    message_history.append(user_response["final_message"])
                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: Added final message to history")
                return

            # Add initial user message to history
            message_history.append(user_response["message"])
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: Added initial user message to history: '{user_response['message']['content']}'")

            # Send initial user audio in batches
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: Converting initial user message to audio...")
            audio_chunks = self._tts(user_response["message"]["content"])
            if audio_chunks:
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: Sending {len(audio_chunks)} audio chunks to websocket in batches...")

                # Send audio chunks in batches of 5 with responses in between
                batch_size = 5
                for i in range(0, len(audio_chunks), batch_size):
                    batch_end = min(i + batch_size, len(audio_chunks))
                    batch_chunks = audio_chunks[i:batch_end]

                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: Sending batch {i//batch_size + 1}/{(len(audio_chunks) + batch_size - 1)//batch_size} (chunks {i+1} to {batch_end})")

                    # Send all chunks in this batch
                    for j, chunk in enumerate(batch_chunks):
                        await self.websocket.send(chunk)
                        await asyncio.sleep(0.1)  # 100ms delay between chunks (like in websocket_test2.py)

                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: Batch {i//batch_size + 1} sent, waiting for response...")

                    # Wait briefly for any response after the batch
                    try:
                        response_message = await asyncio.wait_for(self.websocket.recv(), timeout=2.0)
                        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: Got intermediate response after batch {i//batch_size + 1}")
                    except asyncio.TimeoutError:
                        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: No response after batch {i//batch_size + 1}, continuing...")

                    # Brief pause before next batch
                    await asyncio.sleep(0.5)

                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: SUCCESS - Sent all initial user audio chunks in {(len(audio_chunks) + batch_size - 1)//batch_size} batches")
            else:
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: WARNING - No audio chunks generated for initial response")

            # Main conversation loop with maximum turns
            conversation_ended = False
            turn_count = 0
            max_turns = 20  # Prevent infinite loops
            
            while not conversation_ended and turn_count < max_turns:
                turn_count += 1
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: === TURN {turn_count}/{max_turns} ===")
                
                # Wait for and receive agent response
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: Waiting for agent response...")
                agent_message = await self._receive_agent_response(audio_state, message_history)
                if not agent_message:
                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: No agent response received, ending conversation")
                    break

                # Check if conversation should end
                end_indicators = ["goodbye", "thank you", "end", "bye", "that's all", "finished", "complete"]
                if any(indicator in agent_message.lower() for indicator in end_indicators):
                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: Agent indicated conversation end with: '{agent_message}'")
                    break

                # Get user response to agent
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: Getting user response to agent message...")
                user_response = self._get_user_response(
                    [{"role": "system", "content": self._get_system_prompt(input_str)}]
                    + self._format_messages_for_openai(message_history),
                    self._get_available_functions(),
                )

                if user_response["ended"]:
                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: User response indicates conversation should end")
                    if user_response["final_message"]:
                        message_history.append(user_response["final_message"])
                        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: Added final user message to history")
                    conversation_ended = True
                    break

                # Add user message to history and send audio
                message_history.append(user_response["message"])
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: Added user response to history: '{user_response['message']['content']}'")
                
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: Converting user response to audio...")
                audio_chunks = self._tts(user_response["message"]["content"])
                
                if audio_chunks:
                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: Sending {len(audio_chunks)} audio chunks to websocket in batches...")

                    # Send audio chunks in batches of 5 with responses in between
                    batch_size = 5
                    for i in range(0, len(audio_chunks), batch_size):
                        batch_end = min(i + batch_size, len(audio_chunks))
                        batch_chunks = audio_chunks[i:batch_end]

                        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: Sending batch {i//batch_size + 1}/{(len(audio_chunks) + batch_size - 1)//batch_size} (chunks {i+1} to {batch_end})")

                        # Send all chunks in this batch
                        for j, chunk in enumerate(batch_chunks):
                            await self.websocket.send(chunk)
                            await asyncio.sleep(0.1)  # 100ms delay between chunks (like in websocket_test2.py)

                        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: Batch {i//batch_size + 1} sent, waiting for response...")

                        # Wait briefly for any response after the batch
                        try:
                            response_message = await asyncio.wait_for(self.websocket.recv(), timeout=2.0)
                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: Got intermediate response after batch {i//batch_size + 1}")
                        except asyncio.TimeoutError:
                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: No response after batch {i//batch_size + 1}, continuing...")

                        # Brief pause before next batch
                        await asyncio.sleep(0.5)

                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: SUCCESS - Sent all user response audio chunks in {(len(audio_chunks) + batch_size - 1)//batch_size} batches")
                else:
                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: WARNING - No audio chunks generated for user response")

            if turn_count >= max_turns:
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: WARNING - Reached maximum conversation length ({max_turns} turns)")
                message_history.append({
                    "role": "system", 
                    "content": f"Conversation ended after {max_turns} turns due to maximum length limit"
                })
            
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: SUCCESS - Conversation completed after {turn_count} turns")
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: Final message history length: {len(message_history)}")

        except Exception as e:
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CONVERSATION: ERROR - Error in audio stream handling: {e}")
            raise

    async def _receive_agent_response(self, audio_state: AudioState, message_history: List[Dict]) -> Optional[str]:
        """Receive and process agent response, returning the transcript"""
        buffer = b""
        silence_count = 0
        max_silence_wait = 180  # 3 minutes timeout for silence
        start_time = asyncio.get_event_loop().time()
        
        try:
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: Waiting for agent response with 3-minute timeout...")
            
            while True:
                # Check timeout
                elapsed_time = asyncio.get_event_loop().time() - start_time
                if elapsed_time > max_silence_wait:
                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: Timeout after {elapsed_time:.1f}s, no valid response received")
                    return None
                
                # Wait for agent response with shorter timeout
                try:
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    silence_count += 1
                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: Silence timeout {silence_count}, elapsed: {elapsed_time:.1f}s")
                    continue
                
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: Received response type: {type(response).__name__}, size: {len(response) if hasattr(response, '__len__') else 'N/A'}")

                # Handle JSON messages
                if isinstance(response, str):
                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: Processing string response: '{response[:200]}...'")
                    
                    # First, try to parse as direct JSON
                    try:
                        json_data = json.loads(response)
                        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: Parsed JSON data: {list(json_data.keys()) if isinstance(json_data, dict) else 'not dict'}")
                        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: JSON data type: {type(json_data).__name__}")
                        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: JSON data value: {json_data}")
                        
                        # Log the complete JSON with audio_bytes replaced
                        if isinstance(json_data, dict):
                            log_json = json_data.copy()
                            if "audio_bytes" in log_json:
                                log_json["audio_bytes"] = "audio bytes here"
                            if "audio" in log_json:
                                log_json["audio"] = "audio bytes here"
                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: Complete JSON received: {json.dumps(log_json, indent=2)}")
                        else:
                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: No JSON data found in response: {response}")
                        
                        if isinstance(json_data, dict):
                            # Handle audio-related messages by converting to text
                            if "audio_bytes" in json_data or "audio" in json_data:
                                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: Found audio message, converting to text...")
                                try:
                                    # Get the audio data
                                    audio_data_str = json_data.get("audio_bytes") or json_data.get("audio", "")
                                    if audio_data_str:
                                        # Convert base64 string to bytes
                                        import base64
                                        audio_bytes = base64.b64decode(audio_data_str)
                                        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: Converted audio_bytes to {len(audio_bytes)} bytes")
                                        
                                        # Check if audio is all silence (all zeros)
                                        if all(byte == 0 for byte in audio_bytes):
                                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: Audio is all silence, skipping audio processing...")
                                            # Don't process this silence, but continue to check for text responses
                                        else:
                                            # Reset silence counter when we get non-silence audio
                                            silence_count = 0
                                            
                                            # Convert audio to text using STT
                                            transcript = self._stt(audio_bytes)
                                            if transcript and transcript.strip():
                                                message_history.append(
                                                    {
                                                        "role": "assistant",
                                                        "content": transcript,
                                                    }
                                                )
                                                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: SUCCESS - Agent Audio Response: '{transcript}'")
                                                return transcript
                                            else:
                                                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: WARNING - Audio transcript was empty, checking for text responses...")
                                    else:
                                        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: WARNING - No audio data found in message, checking for text responses...")
                                except Exception as e:
                                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: ERROR - Failed to process audio message: {e}, checking for text responses...")
                                # Don't return None here - continue to check for text responses
                            
                            # Handle conversation messages
                            if "message" in json_data or "text" in json_data:
                                content = json_data.get("message") or json_data.get("text", "")
                                if content and content.strip():
                                    message_history.append({"role": "assistant", "content": content})
                                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: SUCCESS - Agent JSON Response: '{content}'")
                                    return content
                                else:
                                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: WARNING - Empty content in JSON message")
                            
                            # Handle Checkmate dialog messages (nested structure)
                            if "data" in json_data and isinstance(json_data["data"], dict):
                                nested_data = json_data["data"]
                                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Found nested data, keys: {list(nested_data.keys())}")
                                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Nested data content: {nested_data}")
                                
                                # Check for double nesting (data.data.dialog)
                                if "data" in nested_data and isinstance(nested_data["data"], dict):
                                    double_nested = nested_data["data"]
                                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Found double nested data, keys: {list(double_nested.keys())}")
                                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Double nested data content: {double_nested}")
                                    
                                    if "dialog" in double_nested and isinstance(double_nested["dialog"], list):
                                        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Found dialog array in double nested data with {len(double_nested['dialog'])} items")
                                        # Look for the most recent AI message in the dialog
                                        for i, dialog_item in enumerate(reversed(double_nested["dialog"])):
                                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Dialog item {i}: {dialog_item}")
                                            if isinstance(dialog_item, dict) and dialog_item.get("who") == "AI" and "text" in dialog_item:
                                                content = dialog_item["text"]
                                                chrysalis = dialog_item.get("chrysalis")
                                                if content and content.strip():
                                                    message_history.append(
                                                        {
                                                            "role": "assistant",
                                                            "content": content,
                                                            "cart": (chrysalis.get("order") if chrysalis else None),
                                                        }
                                                    )
                                                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: SUCCESS - Agent Dialog Response: '{content}'")
                                                    return content
                                                else:
                                                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - AI dialog item has empty text")
                                            else:
                                                self._print_and_log(
                                                    f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Dialog item not AI or missing text: who={dialog_item.get('who') if isinstance(dialog_item, dict) else 'not dict'}, has_text={'text' in dialog_item if isinstance(dialog_item, dict) else False}"
                                                )

                                        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: INFO - Dialog found in double nested data but no AI messages with text")
                                    else:
                                        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - No dialog array found in double nested data")
                                
                                if "dialog" in nested_data and isinstance(nested_data["dialog"], list):
                                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Found dialog array with {len(nested_data['dialog'])} items")
                                    # Look for the most recent AI message in the dialog
                                    for i, dialog_item in enumerate(reversed(nested_data["dialog"])):
                                        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Dialog item {i}: {dialog_item}")
                                        if isinstance(dialog_item, dict) and dialog_item.get("who") == "AI" and "text" in dialog_item:
                                            content = dialog_item["text"]
                                            chrysalis = dialog_item.get("chrysalis")
                                            if content and content.strip():
                                                message_history.append(
                                                    {
                                                        "role": "assistant",
                                                        "content": content,
                                                        "cart": (chrysalis.get("order") if chrysalis else None),
                                                    }
                                                )
                                                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: SUCCESS - Agent Dialog Response: '{content}'")
                                                return content
                                            else:
                                                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - AI dialog item has empty text")
                                        else:
                                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Dialog item not AI or missing text: who={dialog_item.get('who') if isinstance(dialog_item, dict) else 'not dict'}, has_text={'text' in dialog_item if isinstance(dialog_item, dict) else False}")
                                    
                                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: INFO - Dialog found but no AI messages with text")
                                else:
                                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - No dialog array found in nested data")
                            else:
                                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - No nested data structure found")
                            
                            # Handle data messages (cart, items, etc.)
                            if "data" in json_data:
                                simplified_data = {
                                    "items": json_data["data"].get("items", []),
                                    "cart": json_data["data"].get("cart", ""),
                                }
                                if simplified_data["items"] or simplified_data["cart"]:
                                    message_history.append(
                                        {
                                            "role": "system",
                                            "content": json.dumps(simplified_data),
                                        }
                                    )
                                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: SUCCESS - Data message received: {simplified_data}")
                                    return "Data received"
                                else:
                                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: WARNING - Empty data in JSON message")
                    except json.JSONDecodeError as e:
                        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: JSON decode error: {e}, trying to parse as string containing JSON")
                        
                        # Try to parse as string containing JSON (remove outer quotes if present)
                        try:
                            # Remove outer quotes if the string is wrapped in quotes
                            cleaned_response = response.strip()
                            if cleaned_response.startswith('"') and cleaned_response.endswith('"'):
                                cleaned_response = cleaned_response[1:-1]
                            
                            # Unescape the JSON string
                            cleaned_response = cleaned_response.replace('\\"', '"').replace('\\\\', '\\')
                            
                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Attempting to parse cleaned response: '{cleaned_response[:200]}...'")
                            
                            json_data = json.loads(cleaned_response)
                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: SUCCESS - Parsed string-wrapped JSON: {list(json_data.keys()) if isinstance(json_data, dict) else 'not dict'}")
                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: String-wrapped JSON data type: {type(json_data).__name__}")
                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: String-wrapped JSON data value: {json_data}")
                            
                            # Log the complete JSON with audio_bytes replaced
                            if isinstance(json_data, dict):
                                log_json = json_data.copy()
                                if "audio_bytes" in log_json:
                                    log_json["audio_bytes"] = "audio bytes here"
                                if "audio" in log_json:
                                    log_json["audio"] = "audio bytes here"
                                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: Complete string-wrapped JSON received: {json.dumps(log_json, indent=2)}")
                            else:
                                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: No JSON dict found in string-wrapped response")
                            
                            if isinstance(json_data, dict):
                                # Handle audio-related messages by converting to text
                                if "audio_bytes" in json_data or "audio" in json_data:
                                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: Found audio message, converting to text...")
                                    try:
                                        # Get the audio data
                                        audio_data_str = json_data.get("audio_bytes") or json_data.get("audio", "")
                                        if audio_data_str:
                                            # Convert base64 string to bytes
                                            import base64
                                            audio_bytes = base64.b64decode(audio_data_str)
                                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: Converted audio_bytes to {len(audio_bytes)} bytes")
                                            
                                            # Check if audio is all silence (all zeros)
                                            if all(byte == 0 for byte in audio_bytes):
                                                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: Audio is all silence, skipping audio processing...")
                                                # Don't process this silence, but continue to check for text responses
                                            else:
                                                # Reset silence counter when we get non-silence audio
                                                silence_count = 0
                                                
                                                # Convert audio to text using STT
                                                transcript = self._stt(audio_bytes)
                                                if transcript and transcript.strip():
                                                    message_history.append(
                                                        {
                                                            "role": "assistant",
                                                            "content": transcript,
                                                        }
                                                    )
                                                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: SUCCESS - Agent Audio Response: '{transcript}'")
                                                    return transcript
                                                else:
                                                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: WARNING - Audio transcript was empty, checking for text responses...")
                                        else:
                                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: WARNING - No audio data found in message, checking for text responses...")
                                    except Exception as e:
                                        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: ERROR - Failed to process audio message: {e}, checking for text responses...")
                                    # Don't return None here - continue to check for text responses
                                
                                # Handle conversation messages
                                if "message" in json_data or "text" in json_data:
                                    content = json_data.get("message") or json_data.get("text", "")
                                    if content and content.strip():
                                        message_history.append({"role": "assistant", "content": content})
                                        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: SUCCESS - Agent JSON Response from string: '{content}'")
                                        return content
                                
                                # Handle Checkmate dialog messages (nested structure) from string
                                if "data" in json_data and isinstance(json_data["data"], dict):
                                    nested_data = json_data["data"]
                                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - String JSON: Found nested data, keys: {list(nested_data.keys())}")
                                    
                                    # Check for double nesting (data.data.dialog)
                                    if "data" in nested_data and isinstance(nested_data["data"], dict):
                                        double_nested = nested_data["data"]
                                        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - String JSON: Found double nested data, keys: {list(double_nested.keys())}")
                                        
                                        if "dialog" in double_nested and isinstance(double_nested["dialog"], list):
                                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - String JSON: Found dialog array in double nested data with {len(double_nested['dialog'])} items")
                                            # Look for the most recent AI message in the dialog
                                            for i, dialog_item in enumerate(reversed(double_nested["dialog"])):
                                                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - String JSON: Dialog item {i}: {dialog_item}")
                                                if isinstance(dialog_item, dict) and dialog_item.get("who") == "AI" and "text" in dialog_item:
                                                    content = dialog_item["text"]
                                                    chrysalis = dialog_item.get("chrysalis")
                                                    if content and content.strip():
                                                        message_history.append(
                                                            {
                                                                "role": "assistant",
                                                                "content": content,
                                                                "cart": (chrysalis.get("order") if chrysalis else None),
                                                            }
                                                        )
                                                        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: SUCCESS - Agent Dialog Response from string: '{content}'")
                                                        return content
                                        
                                        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: INFO - String JSON: Dialog found in double nested data but no AI messages with text")
                                    
                                    if "dialog" in nested_data and isinstance(nested_data["dialog"], list):
                                        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - String JSON: Found dialog array with {len(nested_data['dialog'])} items")
                                        # Look for the most recent AI message in the dialog
                                        for i, dialog_item in enumerate(reversed(nested_data["dialog"])):
                                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - String JSON: Dialog item {i}: {dialog_item}")
                                            if isinstance(dialog_item, dict) and dialog_item.get("who") == "AI" and "text" in dialog_item:
                                                content = dialog_item["text"]
                                                chrysalis = dialog_item.get("chrysalis")
                                                if content and content.strip():
                                                    message_history.append(
                                                        {
                                                            "role": "assistant",
                                                            "content": content,
                                                            "cart": (chrysalis.get("order") if chrysalis else None),
                                                        }
                                                    )
                                                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: SUCCESS - Agent Dialog Response from string: '{content}'")
                                                    return content
                        
                        except json.JSONDecodeError as e2:
                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: Second JSON decode error: {e2}, treating as plain text")
                        
                        # Plain text message
                        if response.strip():
                            message_history.append({"role": "assistant", "content": response})
                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: SUCCESS - Agent Text Response: '{response}'")
                            return response
                        else:
                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: WARNING - Empty plain text response")

                # Handle audio data
                if isinstance(response, bytes):
                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: Processing audio data, initial chunk size: {len(response)} bytes")
                    audio_state.agent_is_speaking = True
                    buffer = response

                    # Continue receiving audio chunks until silence
                    silence_count = 0
                    max_silence_count = 5  # Wait for 5 consecutive timeouts before considering it silence
                    chunk_count = 1
                    
                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: Starting audio chunk collection...")
                    while silence_count < max_silence_count:
                        try:
                            chunk = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                            if isinstance(chunk, bytes):
                                buffer += chunk
                                chunk_count += 1
                                silence_count = 0  # Reset silence counter
                                if chunk_count % 10 == 0:  # Log every 10th chunk
                                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: Received {chunk_count} audio chunks, total size: {len(buffer)} bytes")
                            else:
                                # Handle non-bytes response (JSON, text)
                                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: Received non-bytes chunk during audio collection: {type(chunk).__name__}")
                                if isinstance(chunk, str):
                                    try:
                                        json_data = json.loads(chunk)
                                        if "message" in json_data or "text" in json_data:
                                            content = json_data.get("message") or json_data.get("text", "")
                                            if content and content.strip():
                                                message_history.append({"role": "assistant", "content": content})
                                                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: SUCCESS - Agent JSON Response during audio: '{content}'")
                                                audio_state.agent_is_speaking = False
                                                return content
                                        
                                        # Handle Checkmate dialog messages (nested structure) during audio collection
                                        if "data" in json_data and isinstance(json_data["data"], dict):
                                            nested_data = json_data["data"]
                                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Audio collection: Found nested data, keys: {list(nested_data.keys())}")
                                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Audio collection: Nested data content: {nested_data}")
                                            
                                            # Check for double nesting (data.data.dialog) during audio collection
                                            if "data" in nested_data and isinstance(nested_data["data"], dict):
                                                double_nested = nested_data["data"]
                                                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Audio collection: Found double nested data, keys: {list(double_nested.keys())}")
                                                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Audio collection: Double nested data content: {double_nested}")
                                                
                                                if "dialog" in double_nested and isinstance(double_nested["dialog"], list):
                                                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Audio collection: Found dialog array in double nested data with {len(double_nested['dialog'])} items")
                                                    # Look for the most recent AI message in the dialog
                                                    for i, dialog_item in enumerate(reversed(double_nested["dialog"])):
                                                        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Audio collection: Dialog item {i}: {dialog_item}")
                                                        if isinstance(dialog_item, dict) and dialog_item.get("who") == "AI" and "text" in dialog_item:
                                                            content = dialog_item["text"]
                                                            chrysalis = dialog_item.get("chrysalis")
                                                            if content and content.strip():
                                                                message_history.append(
                                                                    {
                                                                        "role": "assistant",
                                                                        "content": content,
                                                                        "cart": (chrysalis.get("order") if chrysalis else None),
                                                                    }
                                                                )
                                                                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: SUCCESS - Agent Dialog Response during audio: '{content}'")
                                                                audio_state.agent_is_speaking = False
                                                                return content
                                                            else:
                                                                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Audio collection: AI dialog item has empty text")
                                                        else:
                                                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Audio collection: Dialog item not AI or missing text: who={dialog_item.get('who') if isinstance(dialog_item, dict) else 'not dict'}, has_text={'text' in dialog_item if isinstance(dialog_item, dict) else False}")
                                                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: INFO - Audio collection: Dialog found in double nested data but no AI messages with text")
                                                else:
                                                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Audio collection: No dialog array found in double nested data")
                                            
                                            if "dialog" in nested_data and isinstance(nested_data["dialog"], list):
                                                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Audio collection: Found dialog array with {len(nested_data['dialog'])} items")
                                                # Look for the most recent AI message in the dialog
                                                for i, dialog_item in enumerate(reversed(nested_data["dialog"])):
                                                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Audio collection: Dialog item {i}: {dialog_item}")
                                                    if isinstance(dialog_item, dict) and dialog_item.get("who") == "AI" and "text" in dialog_item:
                                                        content = dialog_item["text"]
                                                        chrysalis = dialog_item.get("chrysalis")
                                                        if content and content.strip():
                                                            message_history.append(
                                                                {
                                                                    "role": "assistant",
                                                                    "content": content,
                                                                    "cart": (chrysalis.get("order") if chrysalis else None),
                                                                }
                                                            )
                                                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: SUCCESS - Agent Dialog Response during audio: '{content}'")
                                                            audio_state.agent_is_speaking = False
                                                            return content
                                                        else:
                                                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Audio collection: AI dialog item has empty text")
                                                    else:
                                                        self._print_and_log(
                                                            f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Audio collection: Dialog item not AI or missing text: who={dialog_item.get('who') if isinstance(dialog_item, dict) else 'not dict'}, has_text={'text' in dialog_item if isinstance(dialog_item, dict) else False}"
                                                        )
                                            else:
                                                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Audio collection: No dialog array found in nested data")
                                        else:
                                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: DEBUG - Audio collection: No nested data structure found")
                                    except json.JSONDecodeError:
                                        if chunk.strip():
                                            message_history.append({"role": "assistant", "content": chunk})
                                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: SUCCESS - Agent Text Response during audio: '{chunk}'")
                                            audio_state.agent_is_speaking = False
                                            return chunk
                        except asyncio.TimeoutError:
                            silence_count += 1
                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: Silence detected ({silence_count}/{max_silence_count})")
                            continue

                    audio_state.agent_is_speaking = False
                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: Audio collection complete - {chunk_count} chunks, {len(buffer)} total bytes")

                    # Convert audio to text
                    if buffer and len(buffer) > 100:  # Only process if we have meaningful audio data
                        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: Converting audio to text...")
                        try:
                            # Check if audio data is all zeros
                            if all(byte == 0 for byte in buffer):
                                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: WARNING - All audio bytes are zero (silence)")
                                return None
                            else:
                                # Audio data is valid, process it
                                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: Audio data appears valid, processing with STT...")
                                # Convert audio to text using STT
                                transcript = self._stt(buffer)
                                if transcript and transcript.strip():
                                    message_history.append({"role": "assistant", "content": transcript})
                                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: SUCCESS - Agent Audio Response: '{transcript}'")
                                    return transcript
                                else:
                                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: WARNING - Audio transcript was empty")
                                    return None
                        except Exception as e:
                            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: ERROR - Error processing audio transcript: {e}")
                    else:
                        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: WARNING - Insufficient audio data received ({len(buffer)} bytes)")

            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: No valid response processed")
            return None

        except asyncio.TimeoutError:
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: ERROR - Timeout waiting for agent response")
            return None
        except websockets.exceptions.ConnectionClosed:
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: ERROR - WebSocket connection closed")
            return None
        except Exception as e:
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RECEIVE: ERROR - Error receiving agent response: {e}")
            return None

    def _simulate_conversation(self, input_str: str) -> SimulationResponse:
        """Simulate a fluid audio conversation."""
        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] SIMULATION: Starting conversation simulation")
        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] SIMULATION: Input string: {input_str[:200]}...")
        message_history = []
        audio_state = AudioState()
        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] SIMULATION: Created audio state, ready to begin conversation")

        try:
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] SIMULATION: Starting audio stream handling...")
            self.loop.run_until_complete(self._handle_audio_stream(audio_state, input_str, message_history))
            
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] SIMULATION: SUCCESS - Conversation completed successfully")
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] SIMULATION: Final message count: {len(message_history)}")
            
            return SimulationResponse(
                message_history,
                Status(StatusCode.SUCCESS, "Conversation completed successfully"),
            )

        except Exception as e:
            error_message = f"Error during simulation: {str(e)}"
            message_history.append({"role": "system", "content": error_message})
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] SIMULATION: ERROR - {error_message}")
            raise SimulationAPIError(error_message)

    def run_simulation(self, input_str: str) -> List[Dict]:
        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RUN: Starting simulation run")
        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RUN: Input string: {input_str[:200]}...")
        
        try:
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RUN: Running scenario: {input_str[:200]}...")
            model_output = self._simulate_conversation(input_str)

            message_history = model_output.message_history
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RUN: Simulation completed, status: {model_output.status.code}")
            
            if model_output.status.code != StatusCode.SUCCESS:
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RUN: WARNING - Non-success status: {model_output.status.message}")
                message_history.append(
                    {
                        "role": "system",
                        "content": f"Status: {model_output.status.message}",
                    }
                )

            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RUN: SUCCESS - Simulation run completed successfully")
            # Write transcript to file
            transcript_filename = f"checkmate_transcript_{self.run_id}.json"
            with open(transcript_filename, "w") as f:
                json.dump(message_history, f, indent=2)
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RUN: Transcript written to {transcript_filename}")
            return message_history
        finally:
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RUN: Starting cleanup...")
            self._cleanup()
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] RUN: Cleanup completed")

    def _cleanup(self):
        """Clean up resources and properly close the event loop"""
        self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CLEANUP: Starting resource cleanup")
        
        try:
            if self.websocket:
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CLEANUP: Closing websocket connection...")
                if not self.loop.is_closed():
                    self.loop.run_until_complete(self.websocket.close())
                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CLEANUP: SUCCESS - Websocket closed")
                else:
                    self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CLEANUP: WARNING - Event loop already closed, cannot close websocket")
            else:
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CLEANUP: No websocket to close")
                
        except Exception as e:
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CLEANUP: ERROR - Error during cleanup: {e}")
        finally:
            if self.loop and not self.loop.is_closed():
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CLEANUP: Closing event loop...")
                self.loop.close()
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CLEANUP: SUCCESS - Event loop closed")
            else:
                self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CLEANUP: No event loop to close or already closed")
            
            self.websocket = None
            self.loop = None
            self._print_and_log(f"[RUN:{self.run_id}][MODEL:{self.type}] CLEANUP: SUCCESS - All resources cleaned up")
