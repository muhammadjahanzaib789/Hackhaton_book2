#!/usr/bin/env python3
"""
Voice Commander Node
Physical AI Book - Chapter 6: LLM Integration

Voice-controlled robot interface combining speech recognition,
LLM understanding, and text-to-speech feedback.

Usage:
    ros2 run physical_ai_examples voice_commander

Expected Output:
    [INFO] [voice_commander]: Voice commander ready
    [INFO] [voice_commander]: Listening for wake word "hey robot"
    [INFO] [voice_commander]: Wake word detected!
    [INFO] [voice_commander]: Recognized: "pick up the cup"
    [INFO] [voice_commander]: Robot says: "I'll pick up the cup for you"

Dependencies:
    - rclpy
    - sounddevice (optional, for audio)

Author: Physical AI Book
License: MIT
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from typing import Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
import time
import json


class VoiceState(Enum):
    """Voice interface states."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    WAITING_CONFIRMATION = "waiting_confirmation"


@dataclass
class ConversationTurn:
    """A turn in the conversation."""
    speaker: str
    text: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationContext:
    """Conversation context tracking."""
    turns: List[ConversationTurn] = field(default_factory=list)
    current_task: Optional[str] = None
    entities: dict = field(default_factory=dict)

    def add_turn(self, speaker: str, text: str):
        """Add a turn to the conversation."""
        self.turns.append(ConversationTurn(speaker=speaker, text=text))
        # Keep only last 10 turns
        if len(self.turns) > 10:
            self.turns = self.turns[-10:]

    def get_history_string(self) -> str:
        """Get conversation history as string."""
        return "\n".join([
            f"{'User' if t.speaker == 'user' else 'Robot'}: {t.text}"
            for t in self.turns[-5:]
        ])


class VoiceCommanderNode(Node):
    """
    Voice-controlled robot interface.

    Combines:
    - Wake word detection
    - Speech recognition
    - LLM intent understanding
    - Text-to-speech responses
    - Task execution
    """

    WAKE_WORD = "hey robot"

    # Simple intent patterns for demo (production would use LLM)
    INTENT_PATTERNS = {
        'pick': ['pick up', 'grab', 'get', 'take'],
        'place': ['put', 'place', 'set down', 'drop'],
        'navigate': ['go to', 'move to', 'navigate', 'walk to'],
        'find': ['find', 'locate', 'where is', 'look for'],
        'stop': ['stop', 'halt', 'cancel', 'abort'],
        'help': ['help', 'what can you', 'capabilities'],
        'greet': ['hello', 'hi', 'hey', 'good morning', 'good afternoon'],
        'goodbye': ['bye', 'goodbye', 'see you', 'later'],
        'thanks': ['thank', 'thanks', 'appreciate'],
    }

    def __init__(self):
        super().__init__('voice_commander')

        # Parameters
        self.declare_parameter('wake_word_enabled', True)
        self.declare_parameter('confirmation_required', True)
        self.declare_parameter('voice_feedback', True)

        self.wake_word_enabled = self.get_parameter('wake_word_enabled').value
        self.confirmation_required = self.get_parameter('confirmation_required').value
        self.voice_feedback = self.get_parameter('voice_feedback').value

        # State
        self.state = VoiceState.IDLE
        self.context = ConversationContext()
        self.pending_command = None

        # Command queue
        self.command_queue = queue.Queue()

        # Publishers
        self.speech_pub = self.create_publisher(String, '/speech/say', 10)
        self.task_pub = self.create_publisher(String, '/task/request', 10)
        self.status_pub = self.create_publisher(String, '/voice/status', 10)

        # Subscribers
        self.create_subscription(
            String, '/speech/transcript',
            self._transcript_callback, 10
        )
        self.create_subscription(
            Bool, '/speech/is_speaking',
            self._speaking_callback, 10
        )
        self.create_subscription(
            String, '/task/status',
            self._task_status_callback, 10
        )

        # Simulated input for demo
        self.create_subscription(
            String, '/voice/simulate_input',
            self._simulated_input_callback, 10
        )

        # Processing thread
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.daemon = True
        self.process_thread.start()

        self._publish_status()
        self.get_logger().info('Voice commander ready')
        self.get_logger().info(f'Listening for wake word "{self.WAKE_WORD}"')

    def _transcript_callback(self, msg: String):
        """Handle speech recognition result."""
        text = msg.data.strip()
        if not text:
            return

        self.command_queue.put(text)

    def _simulated_input_callback(self, msg: String):
        """Handle simulated text input (for testing)."""
        self.command_queue.put(msg.data.strip())

    def _speaking_callback(self, msg: Bool):
        """Handle TTS status changes."""
        if msg.data:
            self.state = VoiceState.SPEAKING
        elif self.state == VoiceState.SPEAKING:
            self.state = VoiceState.IDLE
        self._publish_status()

    def _task_status_callback(self, msg: String):
        """Handle task execution status."""
        status = msg.data

        if status == 'complete':
            self._speak("Done! Task completed successfully.")
        elif status == 'failed':
            self._speak("I'm sorry, I couldn't complete that task.")

        self.context.current_task = None

    def _process_loop(self):
        """Main processing loop."""
        while rclpy.ok():
            try:
                text = self.command_queue.get(timeout=0.5)
                self._process_input(text)
            except queue.Empty:
                continue

    def _process_input(self, text: str):
        """Process voice input."""
        self.get_logger().info(f'Processing: "{text}"')

        # Check for wake word if enabled
        text_lower = text.lower()

        if self.wake_word_enabled and self.state == VoiceState.IDLE:
            if self.WAKE_WORD not in text_lower:
                return  # Ignore without wake word

            self.get_logger().info('Wake word detected!')
            # Extract command after wake word
            idx = text_lower.find(self.WAKE_WORD)
            text = text[idx + len(self.WAKE_WORD):].strip()
            text = text.lstrip(',. ')

            if not text:
                self._speak("Yes? How can I help you?")
                return

        # Add to conversation
        self.context.add_turn('user', text)

        # Handle confirmation state
        if self.state == VoiceState.WAITING_CONFIRMATION:
            self._handle_confirmation(text)
            return

        # Detect intent
        intent, entities = self._detect_intent(text)
        self.get_logger().info(f'Intent: {intent}, Entities: {entities}')

        # Generate response and action
        response, should_execute = self._generate_response(intent, entities, text)

        # Speak response
        self._speak(response)
        self.context.add_turn('robot', response)

        # Execute if appropriate
        if should_execute and intent not in ['greet', 'goodbye', 'thanks', 'help']:
            if self.confirmation_required:
                self.pending_command = text
                self.state = VoiceState.WAITING_CONFIRMATION
            else:
                self._execute_command(text)

    def _detect_intent(self, text: str) -> tuple:
        """
        Detect intent from text.

        In production, would use LLM for better understanding.
        """
        text_lower = text.lower()
        detected_intent = 'unknown'
        entities = {}

        # Check patterns
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if pattern in text_lower:
                    detected_intent = intent
                    break
            if detected_intent != 'unknown':
                break

        # Extract entities (simplified)
        words = text_lower.split()

        # Color entities
        colors = ['red', 'blue', 'green', 'yellow', 'white', 'black']
        for color in colors:
            if color in words:
                entities['color'] = color
                break

        # Object entities
        objects = ['cup', 'book', 'ball', 'box', 'bottle', 'plate']
        for obj in objects:
            if obj in words:
                entities['object'] = obj
                break

        # Location entities
        locations = ['table', 'shelf', 'kitchen', 'living room', 'desk', 'floor']
        for loc in locations:
            if loc in text_lower:
                entities['location'] = loc
                break

        return detected_intent, entities

    def _generate_response(self, intent: str, entities: dict, original: str) -> tuple:
        """Generate response based on intent."""
        should_execute = False

        if intent == 'greet':
            responses = [
                "Hello! How can I help you today?",
                "Hi there! What would you like me to do?",
                "Hey! Ready to help.",
            ]
            import random
            response = random.choice(responses)

        elif intent == 'goodbye':
            response = "Goodbye! Let me know if you need anything."

        elif intent == 'thanks':
            response = "You're welcome! Happy to help."

        elif intent == 'help':
            response = ("I can pick up objects, navigate to locations, "
                       "and find things for you. Just tell me what you need.")

        elif intent == 'stop':
            response = "Stopping current action."
            should_execute = True

        elif intent == 'pick':
            obj = entities.get('object', 'that')
            color = entities.get('color', '')
            full_obj = f"the {color} {obj}".strip() if color else f"the {obj}"
            response = f"I'll pick up {full_obj} for you."
            should_execute = True

        elif intent == 'place':
            location = entities.get('location', 'there')
            response = f"I'll place it on the {location}."
            should_execute = True

        elif intent == 'navigate':
            location = entities.get('location', 'that location')
            response = f"Navigating to the {location}."
            should_execute = True

        elif intent == 'find':
            obj = entities.get('object', 'that')
            color = entities.get('color', '')
            full_obj = f"{color} {obj}".strip() if color else obj
            response = f"I'll look for the {full_obj}."
            should_execute = True

        else:
            response = f"I heard '{original}'. Could you please rephrase that?"

        return response, should_execute

    def _handle_confirmation(self, text: str):
        """Handle confirmation response."""
        text_lower = text.lower()

        confirmed = any(word in text_lower for word in
                       ['yes', 'yeah', 'yep', 'sure', 'okay', 'ok', 'do it', 'proceed'])
        denied = any(word in text_lower for word in
                    ['no', 'nope', 'cancel', 'stop', 'don\'t', 'nevermind'])

        if confirmed and self.pending_command:
            self._speak("Okay, executing now.")
            self._execute_command(self.pending_command)
        elif denied:
            self._speak("Okay, cancelled. What else can I do for you?")
        else:
            self._speak("Please say yes or no to confirm.")
            return  # Stay in confirmation state

        self.pending_command = None
        self.state = VoiceState.IDLE
        self._publish_status()

    def _execute_command(self, command: str):
        """Send command for execution."""
        self.context.current_task = command

        task_msg = String()
        task_msg.data = command
        self.task_pub.publish(task_msg)

        self.get_logger().info(f'Executing: "{command}"')

    def _speak(self, text: str):
        """Send text to TTS."""
        if self.voice_feedback:
            msg = String()
            msg.data = text
            self.speech_pub.publish(msg)

        self.get_logger().info(f'Robot says: "{text}"')

    def _publish_status(self):
        """Publish current state."""
        msg = String()
        msg.data = json.dumps({
            'state': self.state.value,
            'current_task': self.context.current_task,
            'pending_command': self.pending_command
        })
        self.status_pub.publish(msg)


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    node = VoiceCommanderNode()

    # Demo: Simulate voice commands
    time.sleep(1.0)

    demo_commands = [
        "hey robot, pick up the red cup",
        "yes",
        "hey robot, go to the kitchen",
        "no",
        "hey robot, what can you do?",
    ]

    def send_demo_commands():
        for cmd in demo_commands:
            time.sleep(2.0)
            node.get_logger().info(f'\n--- Simulating: "{cmd}" ---')
            msg = String()
            msg.data = cmd
            node._simulated_input_callback(msg)

    demo_thread = threading.Thread(target=send_demo_commands)
    demo_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
