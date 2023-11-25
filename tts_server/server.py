from flask import Flask, request, jsonify, send_from_directory
import os
import time
import soundfile as sf
import argparse
from . import tts # import here to avoid loading it when running --help

app = Flask(__name__)
cwd = os.getcwd()
fdir = os.path.dirname(__file__)

# Cache for styles
styles = {}
s_prevs = {}

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.route('/destroy_session', methods=['DELETE'])
def destroy_session():
    session = request.args.get('session')

    if not session:
        return jsonify({"error": "Missing session parameter"}), 400

    session_dir = os.path.join(cwd, 'sessions', session)
    if os.path.exists(session_dir):
        os.system(f'rm -rf {session_dir}')

    return jsonify({"success": True}), 200

@app.route('/download', methods=['GET'])
def download_file():
    session = request.args.get('session')
    filename = request.args.get('filename')

    if not session or not filename:
        return jsonify({"error": "Missing session or filename parameter"}), 400

    session_dir = os.path.join(cwd, 'sessions', session)
    file_path = os.path.join(session_dir, filename)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    return send_from_directory(session_dir, filename, as_attachment=True)

@app.route('/tts', methods=['POST'])
def text_to_speech():
    try:
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

        # Create session directory if not exists
        session_dir = os.path.join(cwd, 'sessions', session)
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)

        # Retrieve or compute style
        if voice not in styles:
            voice_fname = os.path.join(fdir, 'voices', f'{voice}.wav')
            styles[voice] = tts.compute_style(voice_fname)
 
        style = styles[voice]
        
        # Generate key for previous voice state
        vkey = f"{session}/{voice}"
        s_prev = s_prevs.get(vkey, None)

        # Generate audio
        wav, s_prev = tts.LFinference(
            text, 
            s_prev,
            style, 
            alpha=alpha, 
            beta=beta, 
            diffusion_steps=diffusion_steps, 
            embedding_scale=embedding_scale
        )
        
        # Update previous voice state
        s_prevs[vkey] = s_prev

        # Save audio
        filename = f"{int(time.time())}.wav"
        filepath = os.path.join(session_dir, filename)
        sf.write(filepath, wav, 24000)
        
        # If we are accepting wav's, return the wav
        if request.headers.get('Accept') == 'audio/wav':
            return send_from_directory(session_dir, filename, as_attachment=True)
        else:
            return jsonify({"filename": filename, "session": session}), 200
    except Exception as e:
        raise
        return jsonify({"error": str(e)}), 500
        
def main():
    parser = argparse.ArgumentParser(description='Run the Text-to-Speech Server')

    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--host', default='127.0.0.1', help='Host IP address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5050, help='Port number (default: 5050)')

    args = parser.parse_args()
    

    app.run(debug=args.debug, host=args.host, port=args.port)
    
if __name__ == '__main__':
    main()
