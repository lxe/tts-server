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
import random
import threading

sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')

url = 'http://localhost:5050'
session_id = random.randint(0, 4294967295)
audio_queue = queue.Queue()

def play_audio():
    while True:
        wav = audio_queue.get()
        if wav is None:
            break
        sd.play(wav, 24000)
        sd.wait()

def signal_handler(signal, frame):
    audio_queue.queue.clear()
    audio_queue.put(None)
    sd.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def play_sentence(sentence, voice, alpha, beta, diffusion_steps, embedding_scale, seed):
    _sentence = sentence
    # if len(sentence) > 50:
    #     _sentence = sentence[:50] + '...'
    print(f'Generating "{_sentence}"', end='')

    start = time.time()
    data = {
        "sessionId": session_id,
        "text": sentence,
        "voice": voice,
        "alpha": alpha,
        "beta": beta,
        "diffusion_steps": diffusion_steps,
        "embedding_scale": embedding_scale,
        "seed": seed
    }

    try:
        response = requests.post(f'{url}/tts', headers={'Accept': 'audio/wav'}, json=data)
    except Exception as e:
        print(e)
        return

    finish = time.time()
    print(f" {finish - start} seconds")

    wav = np.frombuffer(response.content, dtype=np.int16)

    wav = wav[50:]  # there's a click in the beginning, so remove it
    audio_queue.put(wav)


def main():
    global url
    global session_id

    parser = argparse.ArgumentParser(description='Send TTS requests to a server and play the resulting audio.')
    parser.add_argument('passage', help='Passage to convert to speech')
    parser.add_argument('--url', help='Server base URL')
    parser.add_argument('--sessionId', help='Session identifier')
    parser.add_argument('--voice', default='en-f-1', help='Reference voice / style')
    parser.add_argument('--alpha', type=float, default=0.3, help='Alpha parameter')
    parser.add_argument('--beta', type=float, default=0.7, help='Beta parameter')
    parser.add_argument('--diffusion_steps', type=int, default=25, help='Number of diffusion steps')
    parser.add_argument('--embedding_scale', type=float, default=2, help='Embedding scale')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')

    args = parser.parse_args()

    # Queue audio to be played in a separate thread.
    thread = threading.Thread(target=play_audio)
    thread.start()

    if args.sessionId is not None:
        session_id = args.sessionId

    if args.url is not None:
        url = args.url
        
    if (args.seed is None):
        args.seed = session_id  

    if args.passage == '-':
        args.passage = sys.stdin.read()

    sentences = sentence_detector.tokenize(args.passage.strip())

    for sentence in sentences:
        # if the sentence is too long, split it by punctuation, preserving the punctuation, then call play_sentence on each
        if len(sentence) > 250:
            split_sentences = sentence_detector.tokenize(sentence)
            for split_sentence in split_sentences:
                play_sentence(
                    split_sentence,
                    args.voice,
                    args.alpha,
                    args.beta,
                    args.diffusion_steps,
                    args.embedding_scale,
                    args.seed
                )
        else:
            play_sentence(
                sentence,
                args.voice,
                args.alpha,
                args.beta,
                args.diffusion_steps,
                args.embedding_scale,
                args.seed
            )

    audio_queue.put(None)

if __name__ == "__main__":
    main()
