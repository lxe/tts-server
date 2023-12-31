# 🎤 TTS-Server

![](https://img.shields.io/badge/no-bugs-brightgreen.svg) ![](https://img.shields.io/badge/coverage-%F0%9F%92%AF-green.svg)

This project is a simple Text-to-Speech (TTS) server implemented in Python using Flask, along with a command-line interface (CLI) client for sending TTS requests to the server. It utilizes [StyleTTS2](https://github.com/yl4579/StyleTTS2), a powerful text-to-speech synthesis model, to provide high-quality speech synthesis. With this setup, you can convert text into speech using various voice styles and parameters, making it an ideal platform for experimenting with StyleTTS2.

The primary function of this server is to facilitate the generation of long-form text and manage a queue for its subsequent playback through a separate client. 

TODO: 
 - [ ] Add ability to pass custom voice styles

## Starting The Server

```bash
pip install espeak-phonemizer
pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu121 
pip install git+https://github.com/lxe/tts-server.git
```

Run the server like so:

```bash
python -m tts_server.server
# or tts-server
```

You can use `--host <host>` and `--port <port>`.

### Sending TTS Requests

You can CURL it like so:

```
curl -X POST -H "Content-Type: application/json" -H "Accept: audio/wav" -d '{
  "text": "Embrace the chaos and let your words dance to the rhythm of imagination!",
  "style": "en-f-1",
  "alpha": 0.2,
  "beta": 0.4,
  "diffusion_steps": 10,
  "embedding_scale": 1.5,
  "session": "my-session"
}' "http://localhost:5050/tts" | aplay

curl -X DELETE 'http://localhost:5050/destroy_session?session=my-session' 
```

### HTTP API

Perform TTS

- **Method**: POST
- **URL**: `/tts`
- **Request** (JSON):
  - `session` (string, required)
  - `text` (string, required)
  - `style` (string, optional)
  - `alpha` (float, optional)
  - `beta` (float, optional)
  - `diffusion_steps` (int, optional)
  - `embedding_scale` (float, optional)
- **Response** (audio/wav):
  - Binary wav

Note: each TTS generation in a session will follow the style of previous one for long-form narration coherency. 

### CLI Client

You can send TTS requests to the server using the CLI client, which will play them back. You can either use `python -m tts_server.cli` or the `tts-server-cli` binary.

```bash
python -m tts_server.cli <base_url> <passage> [--style <style>] [--alpha <alpha>] [--beta <beta>] [--diffusion_steps <diffusion_steps>] [--embedding_scale <embedding_scale>] [--session <session>]
```

- `<passage>`: The text you want to convert to speech. Pass '-' to read from stdin
- `--url <base_url>`: The URL of the TTS server (e.g., http://localhost:5050).
- `--session <session>`: A session identifier for continuing existing TTS requests.
- `--style <style>` (optional): The style of the voice to use (default: 'en-f-1').
- `--alpha <alpha>` (optional): The alpha parameter (default: 0.1).
- `--beta <beta>` (optional): The beta parameter (default: 0.1).
- `--diffusion_steps <diffusion_steps>` (optional): The number of diffusion steps (default: 7).
- `--embedding_scale <embedding_scale>` (optional): The embedding scale (default: 1).

Example:

```bash
python -m tts_server.cli "In a fantastical forest, flittering fireflies illuminate the night, casting a mesmerizing dance of light and shadow beneath the ancient, gnarled trees."
```

For long-form narration:

```bash
echo "As the fireflies twinkle in harmonious rhythm, their gentle glow reveals the secrets of the woodland. Tiny creatures, hidden from sight by day, emerge to partake in this nocturnal spectacle. Frogs serenade with their melodic croaks, and owls, wise sentinels of the night, exchange hoots that echo through the enchanted forest." | python -m tts_server.cli -
```

## Credits

This project utilizes [StyleTTS2](https://github.com/yl4579/StyleTTS2) for its text-to-speech synthesis capabilities. Special thanks to @yl4579 for creating StyleTTS2 and @fakerybakery for the [styletts2-importable](https://github.com/fakerybakery/StyleTTS2/blob/main/styletts2/inference.py) for inspiration on `tts.py`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.