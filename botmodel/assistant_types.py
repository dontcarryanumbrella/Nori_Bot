from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class AssistantState(Enum):
    IDLE = auto()
    LISTENING = auto()
    WAITING = auto()
    THINKING = auto()
    SPEAKING = auto()
    STOPPING = auto()
    SHUTDOWN = auto()


@dataclass
class AssistantTurn:
    turn_id: int
    user_text: str
    assistant_text: Optional[str] = None
    delivered: bool = False
    interrupted: bool = False
    system_generated: bool = False
    posture_state: Optional[str] = None
    pending_music_file: Optional[str] = None
    track_interruption: bool = True
    emotion: Optional[str] = None
    action: Optional[str] = None
