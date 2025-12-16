---
sidebar_position: 3
title: "Lesson 3: Voice Interaction"
description: "Building voice-controlled robot interfaces"
---

# Voice Interaction

## Learning Objectives

By the end of this lesson, you will be able to:

1. Implement speech recognition for robot commands
2. Add text-to-speech for robot responses
3. Build a conversational interface
4. Handle multi-turn dialogues

## Prerequisites

- Completed Lessons 1-2 of this chapter
- Microphone and speaker hardware
- Understanding of audio processing basics

## Voice Interaction Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Voice Interaction Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User Speech                                                │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Automatic Speech Recognition (ASR)          │   │
│  │    Whisper / Google Speech / Azure Speech           │   │
│  └─────────────────────────────────────────────────────┘   │
│       │                                                     │
│       │  Transcription                                      │
│       ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Intent Processing (LLM)                      │   │
│  │    Parse command, extract entities                   │   │
│  └─────────────────────────────────────────────────────┘   │
│       │                                                     │
│       │  Structured Command                                 │
│       ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Task Planning & Execution                    │   │
│  │    Generate plan, execute actions                    │   │
│  └─────────────────────────────────────────────────────┘   │
│       │                                                     │
│       │  Result / Response                                  │
│       ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Text-to-Speech (TTS)                         │   │
│  │    Piper / XTTS / Cloud TTS                         │   │
│  └─────────────────────────────────────────────────────┘   │
│       │                                                     │
│       ▼                                                     │
│  Robot Speech                                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Speech Recognition

### Whisper Integration

```python
#!/usr/bin/env python3
"""
Speech Recognition Node
Physical AI Book - Chapter 6

Uses OpenAI Whisper for speech-to-text.

Usage:
    ros2 run physical_ai_examples speech_recognition

Expected Output:
    [INFO] [speech_recognition]: Speech recognition ready
    [INFO] [speech_recognition]: Listening...
    [INFO] [speech_recognition]: Recognized: "pick up the red cup"

Dependencies:
    - rclpy
    - whisper (openai-whisper)
    - sounddevice
    - numpy
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import sounddevice as sd
import queue
import threading
from typing import Optional
import time


class SpeechRecognitionNode(Node):
    """
    Speech recognition using Whisper.

    Supports continuous listening with wake word detection.
    """

    def __init__(self):
        super().__init__('speech_recognition')

        # Parameters
        self.declare_parameter('model_size', 'base')
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('wake_word', 'hey robot')
        self.declare_parameter('silence_threshold', 0.01)
        self.declare_parameter('silence_duration', 1.0)

        model_size = self.get_parameter('model_size').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.wake_word = self.get_parameter('wake_word').value.lower()
        self.silence_threshold = self.get_parameter('silence_threshold').value
        self.silence_duration = self.get_parameter('silence_duration').value

        # Load Whisper model
        try:
            import whisper
            self.get_logger().info(f'Loading Whisper {model_size} model...')
            self.model = whisper.load_model(model_size)
            self.get_logger().info('Whisper model loaded')
        except ImportError:
            self.get_logger().error('Whisper not installed. Run: pip install openai-whisper')
            self.model = None

        # Audio queue for streaming
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.is_recording = False

        # Publishers
        self.transcript_pub = self.create_publisher(
            String, '/speech/transcript', 10
        )
        self.command_pub = self.create_publisher(
            String, '/speech/command', 10
        )

        # Subscribers
        self.create_subscription(
            String, '/speech/start_listening',
            self._start_listening_callback, 10
        )
        self.create_subscription(
            String, '/speech/stop_listening',
            self._stop_listening_callback, 10
        )

        # Start listening thread
        self.listen_thread = threading.Thread(target=self._listen_loop)
        self.listen_thread.daemon = True
        self.listen_thread.start()

        self.get_logger().info('Speech recognition ready')

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream."""
        if status:
            self.get_logger().warn(f'Audio status: {status}')
        self.audio_queue.put(indata.copy())

    def _start_listening_callback(self, msg):
        """Start listening for commands."""
        self.is_listening = True
        self.get_logger().info('Started listening')

    def _stop_listening_callback(self, msg):
        """Stop listening for commands."""
        self.is_listening = False
        self.get_logger().info('Stopped listening')

    def _listen_loop(self):
        """Main listening loop."""
        while rclpy.ok():
            if not self.is_listening or self.model is None:
                time.sleep(0.1)
                continue

            try:
                # Record audio until silence
                audio = self._record_until_silence()

                if audio is not None and len(audio) > 0:
                    # Transcribe
                    text = self._transcribe(audio)

                    if text:
                        self.get_logger().info(f'Recognized: "{text}"')

                        # Publish transcript
                        msg = String()
                        msg.data = text
                        self.transcript_pub.publish(msg)

                        # Check for wake word
                        if self._check_wake_word(text):
                            # Extract command after wake word
                            command = self._extract_command(text)
                            if command:
                                cmd_msg = String()
                                cmd_msg.data = command
                                self.command_pub.publish(cmd_msg)

            except Exception as e:
                self.get_logger().error(f'Listen error: {e}')

    def _record_until_silence(self) -> Optional[np.ndarray]:
        """Record audio until silence detected."""
        self.get_logger().debug('Recording...')

        audio_chunks = []
        silence_samples = 0
        silence_needed = int(self.silence_duration * self.sample_rate)

        # Clear queue
        while not self.audio_queue.empty():
            self.audio_queue.get()

        # Start stream
        with sd.InputStream(samplerate=self.sample_rate, channels=1,
                           callback=self._audio_callback, blocksize=1024):
            max_duration = 30  # Maximum recording duration
            start_time = time.time()

            while time.time() - start_time < max_duration:
                try:
                    chunk = self.audio_queue.get(timeout=0.5)
                    audio_chunks.append(chunk)

                    # Check for silence
                    rms = np.sqrt(np.mean(chunk**2))
                    if rms < self.silence_threshold:
                        silence_samples += len(chunk)
                    else:
                        silence_samples = 0

                    # Stop on silence
                    if silence_samples >= silence_needed and len(audio_chunks) > 5:
                        break

                except queue.Empty:
                    continue

        if len(audio_chunks) < 3:
            return None

        audio = np.concatenate(audio_chunks)
        return audio.flatten().astype(np.float32)

    def _transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio using Whisper."""
        try:
            result = self.model.transcribe(
                audio,
                language='en',
                fp16=False
            )
            return result['text'].strip()
        except Exception as e:
            self.get_logger().error(f'Transcription error: {e}')
            return ''

    def _check_wake_word(self, text: str) -> bool:
        """Check if text contains wake word."""
        return self.wake_word in text.lower()

    def _extract_command(self, text: str) -> str:
        """Extract command from text after wake word."""
        lower = text.lower()
        idx = lower.find(self.wake_word)
        if idx >= 0:
            command = text[idx + len(self.wake_word):].strip()
            # Remove punctuation at start
            command = command.lstrip(',. ')
            return command
        return text


def main(args=None):
    rclpy.init(args=args)
    node = SpeechRecognitionNode()

    # Enable listening immediately
    node.is_listening = True

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Text-to-Speech

### Piper TTS Integration

```python
#!/usr/bin/env python3
"""
Text-to-Speech Node
Physical AI Book - Chapter 6

Uses Piper for fast, local text-to-speech.

Usage:
    ros2 run physical_ai_examples text_to_speech

    # Speak a message
    ros2 topic pub /speech/say std_msgs/String "data: 'Hello, I am your robot assistant'"

Expected Output:
    [INFO] [text_to_speech]: TTS ready
    [INFO] [text_to_speech]: Speaking: "Hello, I am your robot assistant"

Dependencies:
    - rclpy
    - piper-tts
    - sounddevice
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import numpy as np
import sounddevice as sd
import queue
import threading
from typing import Optional
import subprocess
import tempfile
import os


class TextToSpeechNode(Node):
    """
    Text-to-speech using Piper.

    Provides fast, natural-sounding speech synthesis.
    """

    def __init__(self):
        super().__init__('text_to_speech')

        # Parameters
        self.declare_parameter('voice', 'en_US-lessac-medium')
        self.declare_parameter('rate', 1.0)
        self.declare_parameter('volume', 0.8)

        self.voice = self.get_parameter('voice').value
        self.rate = self.get_parameter('rate').value
        self.volume = self.get_parameter('volume').value

        # Speech queue
        self.speech_queue = queue.Queue()
        self.is_speaking = False

        # Publishers
        self.speaking_pub = self.create_publisher(
            Bool, '/speech/is_speaking', 10
        )

        # Subscribers
        self.create_subscription(
            String, '/speech/say',
            self._say_callback, 10
        )
        self.create_subscription(
            String, '/speech/stop',
            self._stop_callback, 10
        )

        # Start speech thread
        self.speech_thread = threading.Thread(target=self._speech_loop)
        self.speech_thread.daemon = True
        self.speech_thread.start()

        # Check if Piper is available
        self._check_piper()

        self.get_logger().info('TTS ready')

    def _check_piper(self):
        """Check if Piper TTS is available."""
        try:
            result = subprocess.run(['piper', '--help'],
                                  capture_output=True, text=True)
            self.piper_available = True
            self.get_logger().info('Piper TTS available')
        except FileNotFoundError:
            self.piper_available = False
            self.get_logger().warn(
                'Piper not installed. Using fallback espeak.'
            )

    def _say_callback(self, msg: String):
        """Queue text for speech."""
        self.speech_queue.put(msg.data)

    def _stop_callback(self, msg: String):
        """Stop current speech."""
        # Clear queue
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break
        self.is_speaking = False

    def _speech_loop(self):
        """Main speech synthesis loop."""
        while rclpy.ok():
            try:
                text = self.speech_queue.get(timeout=0.5)

                self.is_speaking = True
                self._publish_speaking_status(True)

                self.get_logger().info(f'Speaking: "{text}"')
                self._speak(text)

                self.is_speaking = False
                self._publish_speaking_status(False)

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Speech error: {e}')
                self.is_speaking = False

    def _speak(self, text: str):
        """Synthesize and play speech."""
        if self.piper_available:
            self._speak_piper(text)
        else:
            self._speak_espeak(text)

    def _speak_piper(self, text: str):
        """Speak using Piper TTS."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wav_path = f.name

        try:
            # Generate audio with Piper
            process = subprocess.run(
                ['piper', '--model', self.voice, '--output_file', wav_path],
                input=text,
                capture_output=True,
                text=True
            )

            if process.returncode == 0:
                # Play audio
                self._play_wav(wav_path)
            else:
                self.get_logger().error(f'Piper error: {process.stderr}')

        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    def _speak_espeak(self, text: str):
        """Fallback to espeak."""
        try:
            subprocess.run(
                ['espeak', '-v', 'en', '-s', str(int(150 * self.rate)), text],
                capture_output=True
            )
        except FileNotFoundError:
            self.get_logger().error('espeak not installed')

    def _play_wav(self, wav_path: str):
        """Play a WAV file."""
        import wave

        with wave.open(wav_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            audio = np.frombuffer(wf.readframes(-1), dtype=np.int16)

            if channels > 1:
                audio = audio.reshape(-1, channels)

            # Normalize and apply volume
            audio = audio.astype(np.float32) / 32768.0 * self.volume

            sd.play(audio, sample_rate)
            sd.wait()

    def _publish_speaking_status(self, speaking: bool):
        """Publish speaking status."""
        msg = Bool()
        msg.data = speaking
        self.speaking_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TextToSpeechNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Conversational Interface

### Dialogue Manager

```python
"""
Dialogue Manager
Physical AI Book - Chapter 6

Manages multi-turn conversations with the robot.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import json
import time


class DialogueState(Enum):
    """States of the dialogue system."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    CONFIRMING = "confirming"
    EXECUTING = "executing"
    CLARIFYING = "clarifying"


@dataclass
class DialogueTurn:
    """A single turn in the dialogue."""
    speaker: str  # 'user' or 'robot'
    text: str
    timestamp: float
    intent: Optional[str] = None
    entities: Optional[Dict[str, Any]] = None


@dataclass
class DialogueContext:
    """Context maintained across dialogue turns."""
    history: List[DialogueTurn] = field(default_factory=list)
    current_task: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    clarification_needed: Optional[str] = None
    max_history: int = 10

    def add_turn(self, speaker: str, text: str,
                 intent: str = None, entities: Dict = None):
        """Add a turn to history."""
        turn = DialogueTurn(
            speaker=speaker,
            text=text,
            timestamp=time.time(),
            intent=intent,
            entities=entities
        )
        self.history.append(turn)

        # Trim old history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        # Update entities
        if entities:
            self.entities.update(entities)

    def get_history_for_prompt(self) -> str:
        """Format history for LLM prompt."""
        lines = []
        for turn in self.history[-5:]:  # Last 5 turns
            role = "Human" if turn.speaker == "user" else "Robot"
            lines.append(f"{role}: {turn.text}")
        return "\n".join(lines)


class DialogueManager:
    """
    Manages conversational interaction.

    Handles multi-turn dialogues, clarifications, and confirmations.
    """

    DIALOGUE_PROMPT = """You are a helpful robot assistant in a conversation.

CONVERSATION HISTORY:
{history}

CURRENT CONTEXT:
- Task in progress: {current_task}
- Known entities: {entities}

USER INPUT: {user_input}

INSTRUCTIONS:
1. Understand the user's intent
2. If clarification needed, ask ONE specific question
3. If ready to execute, confirm the action
4. Keep responses brief and natural (1-2 sentences)

Respond with JSON:
{{
  "response": "What to say to user",
  "intent": "understood_intent or 'clarify'",
  "entities": {{}},
  "action": "none" | "execute" | "clarify" | "confirm",
  "clarification_question": "question if action is clarify"
}}

JSON:"""

    def __init__(self, llm_provider, task_executor):
        """
        Initialize dialogue manager.

        Args:
            llm_provider: LLM for understanding
            task_executor: For executing commands
        """
        self.llm = llm_provider
        self.executor = task_executor
        self.context = DialogueContext()
        self.state = DialogueState.IDLE

        # Response callbacks
        self.on_response = None  # (text) -> None
        self.on_execute = None   # (command, entities) -> None

    def process_input(self, user_input: str) -> str:
        """
        Process user input and generate response.

        Args:
            user_input: User's spoken/typed input

        Returns:
            Robot's response text
        """
        self.state = DialogueState.PROCESSING

        # Add to history
        self.context.add_turn('user', user_input)

        # Generate LLM response
        prompt = self.DIALOGUE_PROMPT.format(
            history=self.context.get_history_for_prompt(),
            current_task=self.context.current_task or "None",
            entities=json.dumps(self.context.entities),
            user_input=user_input
        )

        llm_response = self.llm.generate(prompt)

        # Parse response
        result = self._parse_response(llm_response.content)

        # Handle based on action
        response_text = result.get('response', "I didn't understand that.")

        action = result.get('action', 'none')

        if action == 'clarify':
            self.state = DialogueState.CLARIFYING
            self.context.clarification_needed = result.get('clarification_question')

        elif action == 'confirm':
            self.state = DialogueState.CONFIRMING
            self.context.current_task = result.get('intent')

        elif action == 'execute':
            self.state = DialogueState.EXECUTING
            if self.on_execute:
                self.on_execute(
                    result.get('intent'),
                    result.get('entities', {})
                )
            self.state = DialogueState.IDLE

        else:
            self.state = DialogueState.IDLE

        # Update context
        if result.get('entities'):
            self.context.entities.update(result['entities'])

        # Add robot response to history
        self.context.add_turn(
            'robot', response_text,
            intent=result.get('intent'),
            entities=result.get('entities')
        )

        # Callback
        if self.on_response:
            self.on_response(response_text)

        return response_text

    def handle_confirmation(self, confirmed: bool) -> str:
        """
        Handle user confirmation.

        Args:
            confirmed: Whether user confirmed

        Returns:
            Response text
        """
        if confirmed and self.context.current_task:
            if self.on_execute:
                self.on_execute(
                    self.context.current_task,
                    self.context.entities
                )
            response = "Okay, executing now."
            self.context.current_task = None
        else:
            response = "Okay, cancelled. What else can I help with?"
            self.context.current_task = None

        self.state = DialogueState.IDLE
        self.context.add_turn('robot', response)

        return response

    def _parse_response(self, response: str) -> Dict:
        """Parse LLM JSON response."""
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0:
                return json.loads(response[json_start:json_end])
        except json.JSONDecodeError:
            pass

        return {
            'response': response,
            'action': 'none'
        }

    def reset(self):
        """Reset dialogue context."""
        self.context = DialogueContext()
        self.state = DialogueState.IDLE
```

## Complete Voice Interface

```python
#!/usr/bin/env python3
"""
Voice Interface Node
Physical AI Book - Chapter 6

Complete voice-controlled robot interface.

Usage:
    ros2 run physical_ai_examples voice_interface

Expected Output:
    [INFO] [voice_interface]: Voice interface ready
    [INFO] [voice_interface]: Say "Hey robot" to start
    User: "Hey robot, pick up the cup"
    Robot: "I'll pick up the cup for you."
    [Executing task...]
    Robot: "Done! The cup has been picked up."

Dependencies:
    - All speech recognition and TTS dependencies
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import threading


class VoiceInterfaceNode(Node):
    """
    Complete voice interface for robot control.

    Integrates ASR, TTS, dialogue management, and task execution.
    """

    def __init__(self):
        super().__init__('voice_interface')

        # State
        self.is_active = False
        self.is_processing = False

        # Subscribe to speech input
        self.create_subscription(
            String, '/speech/command',
            self._command_callback, 10
        )

        # Subscribe to speaking status
        self.create_subscription(
            Bool, '/speech/is_speaking',
            self._speaking_callback, 10
        )

        # Publish speech output
        self.say_pub = self.create_publisher(
            String, '/speech/say', 10
        )

        # Publish listening control
        self.listen_pub = self.create_publisher(
            String, '/speech/start_listening', 10
        )

        # Task status subscription
        self.create_subscription(
            String, '/task/status',
            self._task_status_callback, 10
        )

        # Initialize dialogue manager (simplified for demo)
        self.conversation_history = []

        self.get_logger().info('Voice interface ready')
        self.get_logger().info('Say "Hey robot" to start')

        # Start listening
        self._start_listening()

    def _command_callback(self, msg: String):
        """Handle recognized command."""
        command = msg.data.strip()
        if not command:
            return

        self.get_logger().info(f'Command received: "{command}"')

        # Process command
        self.is_processing = True
        response = self._process_command(command)

        # Speak response
        self._speak(response)

    def _process_command(self, command: str) -> str:
        """
        Process voice command.

        In production, would use full dialogue manager.
        """
        command_lower = command.lower()

        # Simple command handling
        if 'hello' in command_lower or 'hi' in command_lower:
            return "Hello! How can I help you today?"

        elif 'pick up' in command_lower or 'grab' in command_lower:
            # Extract object
            words = command_lower.split()
            obj = "the object"
            for i, word in enumerate(words):
                if word in ['pick', 'grab']:
                    obj = ' '.join(words[i+1:])
                    break

            return f"I'll pick up {obj} for you. Starting now."

        elif 'go to' in command_lower or 'navigate' in command_lower:
            # Extract location
            if 'kitchen' in command_lower:
                return "Navigating to the kitchen."
            elif 'living room' in command_lower:
                return "Navigating to the living room."
            else:
                return "Where would you like me to go?"

        elif 'stop' in command_lower:
            return "Stopping current action."

        elif 'help' in command_lower:
            return ("I can pick up objects, navigate to locations, "
                   "and answer questions. What would you like me to do?")

        elif 'thank' in command_lower:
            return "You're welcome! Let me know if you need anything else."

        elif 'bye' in command_lower or 'goodbye' in command_lower:
            return "Goodbye! Have a great day."

        else:
            return f"I heard '{command}'. Could you please rephrase that?"

    def _speak(self, text: str):
        """Send text to TTS."""
        msg = String()
        msg.data = text
        self.say_pub.publish(msg)
        self.get_logger().info(f'Robot: "{text}"')

    def _speaking_callback(self, msg: Bool):
        """Handle speaking status changes."""
        if not msg.data and self.is_processing:
            # Done speaking, resume listening
            self.is_processing = False
            self._start_listening()

    def _task_status_callback(self, msg: String):
        """Handle task execution status."""
        status = msg.data
        if status == 'complete':
            self._speak("Done! Task completed successfully.")
        elif status == 'failed':
            self._speak("I'm sorry, I couldn't complete that task.")

    def _start_listening(self):
        """Start speech recognition."""
        msg = String()
        msg.data = 'start'
        self.listen_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VoiceInterfaceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Summary

Key takeaways from this lesson:

1. **ASR** converts speech to text for robot commands
2. **TTS** provides natural robot voice responses
3. **Dialogue management** handles multi-turn conversations
4. **Context tracking** enables referential understanding
5. **Confirmation** adds safety for important actions

## Next Steps

Continue to [Chapter 7: Vision-Language-Action](../chapter-07-vla/lesson-01-vla-introduction.md) to learn:
- End-to-end vision-language-action models
- Multimodal robot control
- Learning from demonstrations

## Additional Resources

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Piper TTS](https://github.com/rhasspy/piper)
- [Vosk Speech Recognition](https://alphacephei.com/vosk/)
