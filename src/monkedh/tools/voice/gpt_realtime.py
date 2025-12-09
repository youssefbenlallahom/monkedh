"""
Azure GPT-Realtime Voice Handler.
Uses a single WebSocket connection for both STT and TTS.
Provides seamless bidirectional voice conversation.
"""

import os
import json
import base64
import asyncio
import struct
import threading
import queue
from typing import Callable, Optional
from dotenv import load_dotenv

load_dotenv()

# Check for required packages
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("⚠️ pyaudio non installé. Exécutez: pip install pyaudio")

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("⚠️ websockets non installé. Exécutez: pip install websockets")


class GPTRealtimeVoice:
    """
    Full-duplex voice handler using Azure GPT-Realtime API.
    Single WebSocket for both speech recognition and audio synthesis.
    
    Available voices:
    - "alloy"   : Neutral, balanced (default)
    - "shimmer" : Female, warm and friendly ⭐
    - "nova"    : Female, professional ⭐
    - "echo"    : Male, clear
    - "fable"   : Male, British accent
    - "onyx"    : Male, deep and authoritative
    """
    
    # Voice options
    VOICES = {
        "alloy": "Neutral, balanced",
        "shimmer": "Female, warm and friendly ⭐",
        "nova": "Female, professional ⭐", 
        "echo": "Male, clear",
        "fable": "Male, British accent",
        "onyx": "Male, deep and authoritative"
    }
    
    def __init__(self, voice: str = "nova"):
        """
        Initialize GPT-Realtime voice handler.
        
        Args:
            voice: Voice to use for TTS. Options:
                   - "nova" (female, professional) ⭐ DEFAULT - Best for emergencies
                   - "shimmer" (female, warm)
                   - "alloy" (neutral)
                   - "echo" (male, clear)
                   - "fable" (male, British)
                   - "onyx" (male, deep)
        """
        self.api_key = os.getenv("AZURE_REALTIME_API_KEY")
        self.api_base = os.getenv("AZURE_REALTIME_API_BASE")
        
        if not self.api_key or not self.api_base:
            raise ValueError(
                "Missing Azure GPT-Realtime credentials. "
                "Set AZURE_REALTIME_API_KEY and AZURE_REALTIME_API_BASE in .env"
            )
        
        # Voice selection
        self.voice = voice.lower() if voice.lower() in self.VOICES else "nova"
        
        # WebSocket URL
        self.ws_url = self.api_base.replace("https://", "wss://").replace("http://", "ws://")
        
        # Audio settings (GPT-Realtime uses 24kHz mono PCM16)
        self.sample_rate = 24000
        self.channels = 1
        self.chunk_size = 2400  # 100ms at 24kHz
        
        # State
        self.ws = None
        self.is_listening = False
        self.is_speaking = False
        self.audio_queue = queue.Queue()
        self.transcript_queue = queue.Queue()
        
        print(f"✅ GPT-Realtime Voice initialisé (voix: {self.voice} - {self.VOICES[self.voice]})")
    
    def is_available(self) -> bool:
        """Check if all dependencies are available."""
        return PYAUDIO_AVAILABLE and WEBSOCKETS_AVAILABLE
    
    async def _connect(self):
        """Establish WebSocket connection."""
        headers = {"api-key": self.api_key}
        
        self.ws = await websockets.connect(
            self.ws_url,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=10
        )
        
        # Configure session
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": """Tu es un assistant médical d'urgence SAMU. 
                Tu aides les gens en situation d'urgence médicale.
                Réponds de manière claire, calme et concise en français.
                Pose des questions pour évaluer la situation.
                Donne des instructions de premiers secours si nécessaire.""",
                "voice": self.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 800
                }
            }
        }
        
        await self.ws.send(json.dumps(session_config))
        
        # Wait for session confirmation
        while True:
            response = await self.ws.recv()
            data = json.loads(response)
            if data.get("type") in ["session.created", "session.updated"]:
                print("✅ Session GPT-Realtime établie")
                break
    
    async def _disconnect(self):
        """Close WebSocket connection."""
        if self.ws:
            await self.ws.close()
            self.ws = None
    
    def _record_audio_chunk(self, stream) -> bytes:
        """Record a single chunk of audio."""
        try:
            data = stream.read(self.chunk_size, exception_on_overflow=False)
            return data
        except Exception:
            return b''
    
    def _play_audio_chunk(self, stream, audio_data: bytes):
        """Play a chunk of audio."""
        try:
            stream.write(audio_data)
        except Exception as e:
            print(f"Audio playback error: {e}")
    
    async def _audio_input_loop(self, p: 'pyaudio.PyAudio'):
        """Continuously capture audio and send to WebSocket."""
        stream = p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        silence_threshold = 300
        
        try:
            while self.is_listening:
                # Don't send audio while assistant is speaking
                if self.is_speaking:
                    await asyncio.sleep(0.05)
                    continue
                
                # Record chunk
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                
                # Calculate RMS for visual feedback
                try:
                    samples = struct.unpack(f'{self.chunk_size}h', data)
                    rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
                    if rms > silence_threshold:
                        print("█", end="", flush=True)
                except:
                    pass
                
                # Send to WebSocket
                if self.ws:
                    audio_b64 = base64.b64encode(data).decode('utf-8')
                    msg = {
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64
                    }
                    await self.ws.send(json.dumps(msg))
                
                await asyncio.sleep(0.01)
                
        finally:
            stream.stop_stream()
            stream.close()
    
    async def _audio_output_loop(self, p: 'pyaudio.PyAudio'):
        """Play audio chunks from the queue."""
        stream = p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.chunk_size
        )
        
        try:
            while self.is_listening:
                try:
                    audio_data = self.audio_queue.get_nowait()
                    stream.write(audio_data)
                except queue.Empty:
                    await asyncio.sleep(0.01)
        finally:
            stream.stop_stream()
            stream.close()
    
    async def _message_handler(self):
        """Handle incoming WebSocket messages."""
        user_transcript = ""
        assistant_transcript = ""
        
        try:
            while self.is_listening and self.ws:
                try:
                    response = await asyncio.wait_for(self.ws.recv(), timeout=0.1)
                    data = json.loads(response)
                    msg_type = data.get("type", "")
                    
                    # User's speech transcription
                    if msg_type == "conversation.item.input_audio_transcription.completed":
                        user_transcript = data.get("transcript", "")
                        if user_transcript:
                            print(f"\n\n👤 Vous: {user_transcript}")
                            self.transcript_queue.put(("user", user_transcript))
                    
                    # Assistant starting to respond
                    elif msg_type == "response.audio.delta":
                        self.is_speaking = True
                        audio_b64 = data.get("delta", "")
                        if audio_b64:
                            audio_data = base64.b64decode(audio_b64)
                            self.audio_queue.put(audio_data)
                    
                    # Assistant's text response
                    elif msg_type == "response.audio_transcript.delta":
                        delta = data.get("delta", "")
                        assistant_transcript += delta
                        print(delta, end="", flush=True)
                    
                    # Assistant finished speaking
                    elif msg_type == "response.audio_transcript.done":
                        full_text = data.get("transcript", assistant_transcript)
                        if full_text:
                            print(f"\n🤖 Assistant: {full_text[:100]}...")
                            self.transcript_queue.put(("assistant", full_text))
                        assistant_transcript = ""
                    
                    elif msg_type == "response.done":
                        self.is_speaking = False
                        print("\n" + "-"*50)
                        print("🎤 C'EST VOTRE TOUR DE PARLER...")
                        print("-"*50)
                    
                    elif msg_type == "input_audio_buffer.speech_started":
                        print("\n🎙️ Parole détectée... ", end="", flush=True)
                    
                    elif msg_type == "input_audio_buffer.speech_stopped":
                        print(" ✓")
                    
                    elif msg_type == "error":
                        error = data.get("error", {})
                        print(f"\n❌ Erreur API: {error.get('message', 'Unknown')}")
                    
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("\n⚠️ Connexion fermée")
                    break
                    
        except Exception as e:
            print(f"\n❌ Erreur message handler: {e}")
    
    async def _run_conversation(self, on_exit: Callable[[], bool] = None):
        """Run the full-duplex conversation."""
        p = pyaudio.PyAudio()
        
        try:
            await self._connect()
            self.is_listening = True
            self.is_speaking = False
            
            print("\n" + "="*50)
            print("🎤 CONVERSATION VOCALE EN TEMPS RÉEL")
            print("="*50)
            print("Parlez naturellement - l'IA vous répondra automatiquement")
            print("Appuyez sur Ctrl+C pour quitter")
            print("="*50 + "\n")
            
            # Start all async tasks
            tasks = [
                asyncio.create_task(self._audio_input_loop(p)),
                asyncio.create_task(self._audio_output_loop(p)),
                asyncio.create_task(self._message_handler())
            ]
            
            # Wait for all tasks
            await asyncio.gather(*tasks)
            
        except KeyboardInterrupt:
            print("\n\n👋 Au revoir!")
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
        finally:
            self.is_listening = False
            await self._disconnect()
            p.terminate()
    
    def start_conversation(self):
        """Start the voice conversation (blocking)."""
        if not self.is_available():
            print("❌ Dépendances manquantes (pyaudio, websockets)")
            return
        
        asyncio.run(self._run_conversation())
    
    # === Single-turn methods for compatibility ===
    
    def listen_once(self, timeout_seconds: int = 10) -> str:
        """
        Listen for a single utterance and return text.
        Uses the realtime API for transcription.
        """
        if not self.is_available():
            return self._text_fallback()
        
        print("🎤 Parlez maintenant... (s'arrête après 2s de silence)")
        
        audio_data = self._record_audio_simple(timeout_seconds)
        if not audio_data:
            return ""
        
        print("🔄 Transcription en cours...")
        transcript = asyncio.run(self._transcribe_audio(audio_data))
        
        if transcript:
            print(f"\n{'='*50}")
            print(f"📝 VOUS AVEZ DIT: {transcript}")
            print(f"{'='*50}\n")
        
        return transcript
    
    def _record_audio_simple(self, max_seconds: int = 10) -> bytes:
        """Record audio with silence detection."""
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            frames = []
            silence_threshold = 300
            silence_chunks = 0
            max_silence = int(2.0 * self.sample_rate / self.chunk_size)
            started = False
            min_chunks = int(1.5 * self.sample_rate / self.chunk_size)
            max_chunks = int(max_seconds * self.sample_rate / self.chunk_size)
            
            print("   [", end="", flush=True)
            
            for i in range(max_chunks):
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)
                
                try:
                    samples = struct.unpack(f'{self.chunk_size}h', data)
                    rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
                except:
                    rms = 0
                
                if i % 10 == 0:
                    print("█" if rms > silence_threshold else "░", end="", flush=True)
                
                if rms > silence_threshold:
                    started = True
                    silence_chunks = 0
                else:
                    silence_chunks += 1
                
                if len(frames) >= min_chunks and started and silence_chunks > max_silence:
                    break
            
            print("]")
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            print("✅ Enregistrement terminé")
            return b''.join(frames)
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return b''
    
    async def _transcribe_audio(self, audio_data: bytes) -> str:
        """Send audio to GPT-Realtime for transcription only."""
        try:
            headers = {"api-key": self.api_key}
            
            async with websockets.connect(
                self.ws_url,
                additional_headers=headers,
                ping_interval=None
            ) as ws:
                # Configure for transcription
                config = {
                    "type": "session.update",
                    "session": {
                        "modalities": ["text"],
                        "input_audio_format": "pcm16",
                        "input_audio_transcription": {"model": "whisper-1"},
                        "turn_detection": None
                    }
                }
                await ws.send(json.dumps(config))
                
                # Wait for session ready
                while True:
                    resp = await ws.recv()
                    if json.loads(resp).get("type") in ["session.created", "session.updated"]:
                        break
                
                # Send audio
                chunk_size = 4800
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    msg = {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode('utf-8')
                    }
                    await ws.send(json.dumps(msg))
                
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                await ws.send(json.dumps({"type": "response.create"}))
                
                # Get transcription
                transcript = ""
                while True:
                    try:
                        resp = await asyncio.wait_for(ws.recv(), timeout=10)
                        data = json.loads(resp)
                        msg_type = data.get("type", "")
                        
                        if msg_type == "conversation.item.input_audio_transcription.completed":
                            transcript = data.get("transcript", "")
                            break
                        elif msg_type == "response.done":
                            break
                        elif msg_type == "error":
                            print(f"❌ {data.get('error', {}).get('message')}")
                            break
                    except asyncio.TimeoutError:
                        break
                
                return transcript.strip()
                
        except Exception as e:
            print(f"❌ Erreur transcription: {e}")
            return ""
    
    def speak(self, text: str) -> bool:
        """
        Speak text using GPT-Realtime TTS.
        """
        if not self.is_available():
            print(f"🔊 {text}")
            return False
        
        asyncio.run(self._speak_async(text))
        return True
    
    async def _speak_async(self, text: str):
        """Generate and play speech for text."""
        try:
            headers = {"api-key": self.api_key}
            
            async with websockets.connect(
                self.ws_url,
                additional_headers=headers,
                ping_interval=None
            ) as ws:
                # Configure for audio output
                config = {
                    "type": "session.update",
                    "session": {
                        "modalities": ["text", "audio"],
                        "voice": "alloy",
                        "output_audio_format": "pcm16"
                    }
                }
                await ws.send(json.dumps(config))
                
                # Wait for session ready
                while True:
                    resp = await ws.recv()
                    if json.loads(resp).get("type") in ["session.created", "session.updated"]:
                        break
                
                # Send text to speak
                msg = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": f"Dis exactement ceci: {text}"}]
                    }
                }
                await ws.send(json.dumps(msg))
                await ws.send(json.dumps({"type": "response.create"}))
                
                # Collect and play audio
                p = pyaudio.PyAudio()
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.sample_rate,
                    output=True
                )
                
                try:
                    while True:
                        try:
                            resp = await asyncio.wait_for(ws.recv(), timeout=10)
                            data = json.loads(resp)
                            msg_type = data.get("type", "")
                            
                            if msg_type == "response.audio.delta":
                                audio_b64 = data.get("delta", "")
                                if audio_b64:
                                    audio_data = base64.b64decode(audio_b64)
                                    stream.write(audio_data)
                            
                            elif msg_type == "response.done":
                                break
                                
                        except asyncio.TimeoutError:
                            break
                finally:
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    
        except Exception as e:
            print(f"❌ Erreur TTS: {e}")
    
    def _text_fallback(self) -> str:
        """Fallback to text input."""
        try:
            return input("🎤 Votre message (texte): ").strip()
        except:
            return ""


if __name__ == "__main__":
    print("=== Test GPT-Realtime Voice ===\n")
    
    try:
        voice = GPTRealtimeVoice()
        
        if voice.is_available():
            print("Choisissez un mode:")
            print("1. Conversation temps réel (recommandé)")
            print("2. Test STT seul")
            print("3. Test TTS seul")
            
            choice = input("\nVotre choix (1/2/3): ").strip()
            
            if choice == "1":
                voice.start_conversation()
            elif choice == "2":
                text = voice.listen_once()
                print(f"Résultat: {text}")
            elif choice == "3":
                voice.speak("Bonjour, je suis l'assistant médical d'urgence.")
        else:
            print("❌ Dépendances manquantes")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
