import requests
import argparse
import sounddevice as sd
import numpy as np
import time
import sys
import threading
import queue
import signal
import nltk.data

sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')

base_url = 'http://localhost:5050'
session = str(int(time.time()))
audio_queue = queue.Queue()


def play_audio():
    while True:
        wav = audio_queue.get()
        if wav is None:
            break
        sd.play(wav, 24000)
        sd.wait()


def play_sentence(sentence, style, alpha, beta, diffusion_steps, embedding_scale):
    _sentence = sentence
    if len(sentence) > 50:
        _sentence = sentence[:50] + '...'
    print(f'Generating "{_sentence}"', end='')

    start = time.time()
    data = {
        "session": session,
        "text": sentence,
        "style": style,
        "alpha": alpha,
        "beta": beta,
        "diffusion_steps": diffusion_steps,
        "embedding_scale": embedding_scale
    }

    try:
        response = requests.post(
            f'{base_url}/tts',
            headers={'Accept': 'audio/wav'},
            json=data
        )
    except Exception as e:
        print(e)
        return

    finish = time.time()
    print(f" {finish - start} seconds")

    wav = np.frombuffer(response.content, dtype=np.int16)

    wav = wav[50:]  # there's a click in the beginning, so remove it
    audio_queue.put(wav)


def main():
    global base_url
    global session

    parser = argparse.ArgumentParser(
        description='Send TTS requests to a server.')
    parser.add_argument('base_url', help='The URL of the TTS server')
    parser.add_argument('passage', help='Passage to convert to speech')
    parser.add_argument('--session', help='Session identifier')
    parser.add_argument('--style', default='en-f-1', help='Style of the voice')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Alpha parameter')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Beta parameter')
    parser.add_argument('--diffusion_steps', type=int,
                        default=7, help='Number of diffusion steps')
    parser.add_argument('--embedding_scale', type=float,
                        default=1, help='Embedding scale')

    args = parser.parse_args()

    # Queue audio to be played in a separate thread.
    thread = threading.Thread(target=play_audio)
    thread.start()

    if args.session is not None:
        session = args.session

    if args.base_url is not None:
        base_url = args.base_url

    if args.passage == '-':
        args.passage = sys.stdin.read()

    sentences = sentence_detector.tokenize(args.passage.strip())

    for sentence in sentences:
        play_sentence(
            sentence,
            args.style,
            args.alpha,
            args.beta,
            args.diffusion_steps,
            args.embedding_scale
        )

    audio_queue.put(None)


def signal_handler(sig, frame):
    audio_queue.put(None)
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    main()
