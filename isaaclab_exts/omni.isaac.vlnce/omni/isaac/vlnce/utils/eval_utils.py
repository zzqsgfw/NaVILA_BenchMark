from typing import List, Optional

import cv2
import attr
import textwrap
import gzip
import json
import numpy as np


@attr.s(auto_attribs=True)
class InstructionData:
    instruction_text: str
    instruction_tokens: Optional[List[str]] = None


def skip(*args, **kwargs):
    pass


def read_episodes(file_path):
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    
    return data["episodes"]


def add_instruction_on_img(img: np.ndarray, text: str, start_y=0) -> None:
    font_size = 0.6
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    char_size = cv2.getTextSize(" ", font, font_size, thickness)[0]
    wrapped_text = textwrap.wrap(
        text, width=int((img.shape[1] - 15) / char_size[0])
    )
    if len(wrapped_text) < 8:
        wrapped_text.insert(0, "")

    y = start_y
    start_x = 15
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, thickness)[0]
        y += textsize[1] + 25
        cv2.putText(
            img,
            line,
            (start_x, y),
            font,
            font_size,
            (0, 0, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )


def get_vel_command(text):
    """Parse the VLM free-text into a velocity command.

    Returns (vel, time_to_go, parse): `parse` is a dict for offline failure
    attribution / IPSR:
      action        : turn_left | turn_right | move_forward | stop | unparsed
      mag_matched   : whether an expected magnitude token (15/30/45 deg or
                      25/50/75 cm) was found; False means the magnitude was
                      guessed (brittle-parse near-miss)
      fallthrough   : True iff none of the action keywords matched at all
                      (the silent [0.5,0,0] default -> class-(c) parse failure)
    """
    t = text.lower()
    if "turn left" in t:
        if "45" in t:
            return [0.0, 0.0, np.pi/6.0], 1.5, {"action": "turn_left", "mag_matched": True, "fallthrough": False}
        elif "30" in t:
            return [0.0, 0.0, np.pi/6.0], 1.0, {"action": "turn_left", "mag_matched": True, "fallthrough": False}
        elif "15" in t:
            return [0.0, 0.0, np.pi/6.0], 0.5, {"action": "turn_left", "mag_matched": True, "fallthrough": False}
        return [0.0, 0.0, np.pi/6.0], 0.5, {"action": "turn_left", "mag_matched": False, "fallthrough": False}
    elif "turn right" in t:
        if "45" in t:
            return [0.0, 0.0, -np.pi/6.0], 1.5, {"action": "turn_right", "mag_matched": True, "fallthrough": False}
        elif "30" in t:
            return [0.0, 0.0, -np.pi/6.0], 1.0, {"action": "turn_right", "mag_matched": True, "fallthrough": False}
        elif "15" in t:
            return [0.0, 0.0, -np.pi/6.0], 0.5, {"action": "turn_right", "mag_matched": True, "fallthrough": False}
        return [0.0, 0.0, -np.pi/6.0], 0.5, {"action": "turn_right", "mag_matched": False, "fallthrough": False}
    elif "move forward" in t or "move" in t:
        if "75" in t:
            return [0.5, 0.0, 0.0], 1.5, {"action": "move_forward", "mag_matched": True, "fallthrough": False}
        elif "50" in t:
            return [0.5, 0.0, 0.0], 1.0, {"action": "move_forward", "mag_matched": True, "fallthrough": False}
        elif "25" in t:
            return [0.5, 0.0, 0.0], 0.5, {"action": "move_forward", "mag_matched": True, "fallthrough": False}
        return [0.5, 0.0, 0.0], 0.5, {"action": "move_forward", "mag_matched": False, "fallthrough": False}
    elif "stop" in t:
        return [0.0, 0.0, 0.0], 0.0, {"action": "stop", "mag_matched": True, "fallthrough": False}
    else:
        # silent default: VLM said something the parser cannot map -> class-(c)
        return [0.5, 0.0, 0.0], 0.5, {"action": "unparsed", "mag_matched": False, "fallthrough": True}