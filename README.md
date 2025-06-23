# Checkmate Simulation Runner

This repository helps you run the CheckmateModelManager without needing the rest of the pipeline. It provides a simple interface for running conversations with Checkmate agents via WebSocket.

### Files

- `CheckmateModelManager.py` - The main Checkmate model manager for voice conversations
- `run_simulation.py` - Simple simulation runner that uses CheckmateModelManager
- Various stub files for dependencies (`models/`, `llm_clients/`, etc.)

### Setup

1. Set up virtual environment

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set up .env file

```
OPENAI_API_KEY='your-open-ai-key'
ELEVENLABS_API_KEY='your-eleven-labs-key'
DEEPGRAM_API_KEY='your-deepgram-key'
WEBSOCKET_ENDPOINT='wss://your-checkmate-server.com/ws'
```

3. Run simulation

```
python run_simulation.py --duration 5 --objective "I want to schedule a meeting"
```

4. Transcript and summary files will be saved in the local directory

### Command Line Options

- `--objective`: User objective for the conversation (default: "I want to schedule a meeting")
- `--duration`: Simulation duration in minutes (default: 5)
- `--websocket-endpoint`: WebSocket endpoint for Checkmate server (overrides WEBSOCKET_ENDPOINT env var)
- `--simulation-prompt`: Custom simulation prompt for the assistant

### Example Usage

```bash
# Basic simulation
python run_simulation.py

# Custom objective and duration
python run_simulation.py --objective "Order a five dollar meal" --duration 10

# With custom websocket endpoint
python run_simulation.py --websocket-endpoint "wss://my-checkmate-server.com/ws"
```
