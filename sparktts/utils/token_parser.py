TASK_TOKEN_MAP = {
    "vc": "<|task_vc|>",
    "tts": "<|task_tts|>",
    "asr": "<|task_asr|>",
    "s2s": "<|task_s2s|>",
    "t2s": "<|task_t2s|>",
    "understand": "<|task_understand|>",
    "caption": "<|task_cap|>",
    "controllable_tts": "<|task_controllable_tts|>",
    "prompt_tts": "<|task_prompt_tts|>",
    "speech_edit": "<|task_edit|>",
}

LEVELS_MAP = {
    "very_low": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "very_high": 4,
}

LEVELS_MAP_UI = {
    1: 'very_low',
    2: 'low',
    3: 'moderate',
    4: 'high',
    5: 'very_high'
}

GENDER_MAP = {
    "female": 0,
    "male": 1,
}

AGE_MAP = {"Child": 0, "Teenager": 1, "Youth-Adult": 2, "Middle-aged": 3, "Elderly": 4}

EMO_MAP = {
    "UNKNOWN": 0,
    "NEUTRAL": 1,
    "ANGRY": 2,
    "HAPPY": 3,
    "SAD": 4,
    "FEARFUL": 5,
    "DISGUSTED": 6,
    "SURPRISED": 7,
    "SARCASTIC": 8,
    "EXCITED": 9,
    "SLEEPY": 10,
    "CONFUSED": 11,
    "EMPHASIS": 12,
    "LAUGHING": 13,
    "SINGING": 14,
    "WORRIED": 15,
    "WHISPER": 16,
    "ANXIOUS": 17,
    "NO-AGREEMENT": 18,
    "APOLOGETIC": 19,
    "CONCERNED": 20,
    "ENUNCIATED": 21,
    "ASSERTIVE": 22,
    "ENCOURAGING": 23,
    "CONTEMPT": 24,
}


class TokenParser:
    """Turn label to special token"""

    def __init__(self):
        pass

    """Parse the attributes of a person."""

    def __init__(self):
        pass

    @staticmethod
    def age(age: str) -> str:
        """Turn age token."""
        age_id = AGE_MAP[age]
        return f"<|age_{age_id}|>"

    @staticmethod
    def gender(gender: str) -> str:
        """Turn gender token."""
        gender_id = GENDER_MAP[gender]
        return f"<|gender_{gender_id}|>"

    @staticmethod
    def mel_value(mel: int):
        """Turn special token of mel scale pitch."""
        mel = max(0, int(mel))
        mel = min(1000, int(mel))
        return f"<|pitch_value_{mel}|>"

    @staticmethod
    def mel_level(level: str):
        """Turn special token of mel level."""
        level_tag = LEVELS_MAP[level]
        return f"<|pitch_label_{level_tag}|>"

    @staticmethod
    def pitch_var_value(pitch_std: int):
        """Turn special token of pitch_std value."""
        assert isinstance(pitch_std, int)
        pitch_std = max(0, int(pitch_std))
        pitch_std = min(10, int(pitch_std))
        return f"<|pitch_var_value_{pitch_std}|>"

    @staticmethod
    def pitch_var_level(level: str):
        """Turn special token of pitch std level."""
        level_tag = LEVELS_MAP[level]
        return f"<|pitch_var_label_{level_tag}|>"

    @staticmethod
    def loudness_value(loudness: int):
        """Turn special toak of loudness value [0, 30]"""
        assert loudness >= 0
        loudness = max(0, int(loudness))
        loudness = min(30, int(loudness))
        return f"<|loudness_value_{loudness}|>"

    @staticmethod
    def loudness_level(level: str):
        """Turn special token of loudness level."""
        level_tag = LEVELS_MAP[level]
        return f"<|loudness_label_{level_tag}|>"

    @staticmethod
    def speed_value(speed: int):
        """Turn special token of speed value."""
        speed = max(0, int(speed))
        speed = min(10, int(speed))
        return f"<|speed_value_{speed}|>"

    @staticmethod
    def speed_level(level: str):
        """Turn special token of speed level."""
        level_tag = LEVELS_MAP[level]
        return f"<|speed_label_{level_tag}|>"

    @staticmethod
    def task(task: str) -> str:
        """Turn special token of task."""
        assert task in TASK_TOKEN_MAP.keys()

        return TASK_TOKEN_MAP[task]

    @staticmethod
    def emotion(emotion: str):
        emo_id = EMO_MAP[emotion]

        return f"<|emotion_{emo_id}|>"


# test
if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "/aifs4su/xinshengwang/code/StyleCraft/tokenizer/stylecraft-bicodec-pitch-loudness-speed-emotion-tokenizer"
    )

    tasks = ["tts", "tts", "understand", "controllable_tts", "prompt_tts"]
    ages = ["Child", "Teenager", "Youth-Adult", "Middle-aged", "Elderly"]
    genders = ["female", "female", "female", "male", "male"]
    mels = [100, 200, 300, 400, 500]
    mel_levels = ["very_low", "low", "moderate", "high", "very_high"]
    loudnesses = [1, 10, 23, 19, 30]
    loudness_levels = ["very_low", "low", "moderate", "high", "very_high"]
    emotions = ["UNKNOWN", "NEUTRAL", "ANGRY", "HAPPY", "SAD"]

    for i in range(5):
        task = TokenParser.task(tasks[i])
        age = TokenParser.age(ages[i])
        gender = TokenParser.gender(genders[i])
        mel = TokenParser.mel_value(mels[i])
        mel_level = TokenParser.mel_level(mel_levels[i])
        loudness = TokenParser.loudness_value(loudnesses[i])
        loudness_level = TokenParser.loudness_level(loudness_levels[i])
        emotion = TokenParser.emotion(emotions[i])
        inputs = [task, age, gender, mel, mel_level, loudness, loudness_level, emotion]
        inputs = "".join(inputs)
        ids = tokenizer.encode(inputs, add_special_tokens=False)
        print(ids)
        print("decode", tokenizer.decode(ids))
