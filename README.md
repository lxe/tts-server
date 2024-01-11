# 🎤 TTS-Server

![](https://img.shields.io/badge/no-bugs-brightgreen.svg) ![](https://img.shields.io/badge/coverage-%F0%9F%92%AF-green.svg)

This project is a Text-to-Speech (TTS) server implemented in Python using Flask. It utilizes [StyleTTS2](https://github.com/yl4579/StyleTTS2) for high-quality speech synthesis. The server facilitates the generation of speech from text with various voice styles and parameters, and manages a queue for processing TTS requests.

## Features

- Streaming generation allows for long-form narration
- Upload your own voice style files
- Voice cache improves generation performance
- No direct dependency on espeak (but it's required to be installed on your system -- see prerequisites)
- CLI client

## Prerequisites

You need torch and [phonemizer](https://pypi.org/project/phonemizer/) installed on your system / environment:

```bash
pip install phonemizer
```

## Starting The Server

Install the server a

```bash
pip install git+https://github.com/lxe/tts-server.git
```

Run the server:

```bash
python -m tts_server.server
```

You can use `--host <host>` and `--port <port>` to specify the server's address and port. `--help` for more options.


## HTTP API

### Create New Session
- **Method**: POST
- **URL**: `/session/new`
- **Request** (Form-data or JSON):
  - `voice` (file, optional): Upload a voice file.
  - `voice` (string, optional): Predefined voice name.
- **Response** (JSON):
  - `message`: Confirmation message.
  - `voice`: Voice used in the session.
  - `session_id`: Generated session ID.

### Perform TTS
- **Method**: POST
- **URL**: `/tts`
- **Request** (JSON):
  - `sessionId` (integer, required): Session identifier.
  - `text` (string, required): Text to be synthesized.
  - `alpha`, `beta`, `diffusion_steps`, `embedding_scale` (floats/integers, optional): TTS parameters.
- **Response** (audio/wav):
  - Binary WAV file.

### Get Available Voices
- **Method**: GET
- **URL**: `/voices`
- **Response** (JSON):
  - List of available voice keys.

### Error Handling
- Custom error responses for various server exceptions.

## CLI Client

The server can be interacted with via CURL commands:

```bash
curl -X POST -H "Content-Type: application/json" -H "Accept: audio/wav" -d '{
  "sessionId": 12345,
  "text": "Embrace the chaos and let your words dance to the rhythm of imagination!",
  "alpha": 0.2,
  "beta": 0.4,
  "diffusion_steps": 10,
  "embedding_scale": 1.5
}' "http://localhost:5050/tts" | aplay
```

...or the CLI client...

```bash
python -m tts_server.cli "In a fantastical forest, flittering fireflies illuminate the night, casting a mesmerizing dance of light and shadow beneath the ancient, gnarled trees."
```

You can pass longer text to the cli for long-form narration:

```bash
echo "As the fireflies twinkle in harmonious rhythm, their gentle glow reveals the secrets of the woodland. Tiny creatures, hidden from sight by day, emerge to partake in this nocturnal spectacle. Frogs serenade with their melodic croaks, and owls, wise sentinels of the night, exchange hoots that echo through the enchanted forest." | python -m tts_server.cli -
```

## Credits

This project utilizes [StyleTTS2](https://github.com/yl4579/StyleTTS2) for its text-to-speech synthesis capabilities. Special thanks to @yl4579 for creating StyleTTS2 and @fakerybakery for the [styletts2-importable](https://github.com/fakerybakery/StyleTTS2/blob/main/styletts2/inference.py) for inspiration on `tts.py`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.