import queue
import threading
from flask import Flask, request, jsonify, send_file
import os
import io
import soundfile as sf
import argparse
from styletts2 import TTS
import subprocess
import traceback
import random

app = Flask(__name__)
fdir = os.path.dirname(__file__)

# Cache for voices
voices = {}
prev_s_db = {}

tts = None


class HotProcessQueue:
    def __init__(self, *popen_args, **popen_kwargs):
        self.popen_args = popen_args
        self.popen_kwargs = popen_kwargs
        self.process_queue = queue.Queue()
        self.lock = threading.Lock()
        # Initial process spawning
        self.spawn_new_process()

    def spawn_new_process(self):
        # Spawn a new process and add it to the queue
        process = subprocess.Popen(
            *self.popen_args,
            **self.popen_kwargs,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.process_queue.put(process)

    def consume(self, input_string):
        # Wait for an available process
        process = self.process_queue.get()

        # Communicate with the process
        stdout, stderr = process.communicate(input=input_string.encode())
        process.terminate()

        # Spawn a new process asynchronously
        threading.Thread(target=self.spawn_new_process).start()

        return stdout.decode()

    def __del__(self):
        # Cleanup: close all processes
        while not self.process_queue.empty():
            process = self.process_queue.get()
            process.terminate()


class Phonemizer:
    def __init__(self, language='en-us'):
        self.language = language
        self.queue = HotProcessQueue([
            'phonemize',
            '--preserve-punctuation', '--with-stress', '--language', self.language
        ])

    def phonemize(self, texts):
        try:
            passage = texts[0]
            output = self.queue.consume(passage)
            return [output]
        except Exception as e:
            print(e)
            return None


phonemizer = Phonemizer()


class ServerException(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@app.errorhandler(ServerException)
def handle_server_exception(error):
    traceback.print_exc()
    response = jsonify({"error": error.message})
    response.status_code = error.status_code
    return response


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


def compute_voice(voice_file_path):
    print(f'Computing style for {voice_file_path}')
    voice = tts.compute_style(voice_file_path)
    return voice


@app.route('/voices', methods=['GET'])
def get_voices():
    return jsonify(list(voices.keys()))


@app.route('/session/new', methods=['POST'])
def new_session():
    try:
        # random session id between 0 and 2^32
        session_id = random.randint(0, 4294967295)

        if 'voice' in request.files and request.files['voice'].filename != '':
            voice_file = request.files['voice']
            voice_name = os.path.splitext(voice_file.filename)[0]

            # Read file into memory
            voice_data = io.BytesIO()
            voice_file.save(voice_data)
            voice_data.seek(0)

            # Convert style to wav using FFmpeg, reading from and writing to memory
            process = subprocess.Popen(
                ['ffmpeg', '-i', '-', '-ac', '1', '-ar', '24000', '-f', 'wav', '-'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(input=voice_data.read())

            # Check for errors
            if process.returncode != 0:
                return jsonify({"error": "FFmpeg error", "cause": stderr}), 500

            # Convert stdout bytes to a BytesIO object
            style_wav_data = io.BytesIO(stdout)

            # Compute style and store in voices dictionary
            voices[session_id] = compute_voice(style_wav_data)
        else:
            voice_name = request.form['voice']
            if voice_name == '':
                voice_name = 'en-f-1'
            voice_file_path = os.path.join(fdir, 'voices', f'{voice_name}.wav')
            voice = compute_voice(voice_file_path)
            voices[session_id] = voice
            voices[voice_name] = voice

        return jsonify({
            "message": "New session created",
            "voice": voice_name,
            "session_id": session_id
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Failed to create new session"}), 500


@app.route('/tts', methods=['POST'])
def text_to_speech():
    try:
        # Accept header validation
        if request.headers.get('Accept') != 'audio/wav':
            raise ServerException("Only audio/wav is supported", 406)

        data = request.json
        session_id = int(data.get('sessionId', -1))  # -1 if not found

        # Session ID validation
        if session_id < 0 or session_id > 4294967295:
            raise ServerException(f"Invalid session ID: {session_id}", 400)

        # Compute or retrieve style
        if session_id not in voices:
            voice_name = data.get('voice', 'en-f-1')
            if voice_name in voices:
                voice = voices[voice_name]
            else:
                voice_file_path = os.path.join(
                    fdir, 'voices', f'{voice_name}.wav')
                voice = compute_voice(voice_file_path)
                voices[session_id] = voice
                voices[voice_name] = voice
        else:
            voice = voices[session_id]

        text = data.get('text')
        if not text:
            raise ServerException("Text is required", 400)

        seed = data.get('seed', session_id)
        if (seed == -1):
            seed = random.randint(0, 4294967295)
        tts.set_seed(int(seed))

        # Generate audio
        wav, prev_s = tts.inference(
            text,
            voice,
            phonemizer=phonemizer,
            prev_s=prev_s_db.get(session_id, None),
            alpha=data.get('alpha', 0.3),
            beta=data.get('beta', 0.7),
            diffusion_steps=data.get('diffusion_steps', 20),
            embedding_scale=data.get('embedding_scale', 1.5)
        )

        # Update previous voice state
        prev_s_db[session_id] = prev_s

        byte_io = io.BytesIO()
        sf.write(byte_io, wav, 24000, format='WAV')
        byte_io.seek(0)
        return send_file(byte_io, mimetype='audio/wav')

    except ServerException as e:
        raise e

    except Exception as e:
        raise ServerException(str(e), 500)


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
