from flask import Flask, request, jsonify, send_file
import os
import io
import soundfile as sf
import argparse
from styletts2 import TTS

app = Flask(__name__)
fdir = os.path.dirname(__file__)

# Cache for styles
styles = {}
prev_s_db = {}

tts = None


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.route('/tts', methods=['POST'])
def text_to_speech():
    try:
        if request.headers.get('Accept') != 'audio/wav':
            return jsonify({"error": "Only audio/wav is supported"}), 400

        data = request.json
        session = data.get('session')
        text = data.get('text')

        # Check if session and text are provided
        if not session:
            return jsonify({"error": "Missing session"}), 400
        if not text:
            return jsonify({"error": "Missing text"}), 400

        voice = data.get('style', 'en-f-1')
        alpha = data.get('alpha', 0.1)
        beta = data.get('beta', 0.3)
        diffusion_steps = data.get('diffusion_steps', 5)
        embedding_scale = data.get('embedding_scale', 1)

        # Retrieve or compute style
        if voice not in styles:
            voice_fname = os.path.join(fdir, 'voices', f'{voice}.wav')
            styles[voice] = tts.compute_style(voice_fname)

        style = styles[voice]

        # Generate key for previous voice state
        vkey = f"{session}/{voice}"
        prev_s = prev_s_db.get(vkey, None)

        # Generate audio
        wav, prev_s = tts.inference(
            text,
            style,
            prev_s=prev_s,
            alpha=alpha,
            beta=beta,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale
        )

        # Update previous voice state
        prev_s_db[vkey] = prev_s

        byte_io = io.BytesIO()
        sf.write(byte_io, wav, 24000, format='WAV')
        byte_io.seek(0)
        return send_file(byte_io, mimetype='audio/wav')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    global tts
    parser = argparse.ArgumentParser(
        description='Run the Text-to-Speech Server')

    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--host', default='127.0.0.1',
                        help='Host IP address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5050,
                        help='Port number (default: 5050)')

    args = parser.parse_args()

    tts = TTS.load_model(
        config_path="hf://yl4579/StyleTTS2-LibriTTS/Models/LibriTTS/config.yml",
        checkpoint_path="hf://yl4579/StyleTTS2-LibriTTS/Models/LibriTTS/epochs_2nd_00020.pth"
    )

    app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
