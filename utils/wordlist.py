"""
Word list utilities for word-based Codenames.
"""

from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)
curr_dir = os.path.dirname(os.path.abspath(__file__))


# Default wordlist file
WORD_LIST_FILE = os.path.join(curr_dir, "wordlist.txt")


def load_wordlist(filepath: str = "wordlist.txt") -> list[str]:
    """
    Load word list from file.

    Args:
        filepath: Path to wordlist file (one word per line).

    Returns:
        List of words in uppercase.

    Raises:
        FileNotFoundError: If wordlist file doesn't exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Wordlist file not found: {filepath}")

    words = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:  # Skip empty lines
                words.append(word.upper())

    return words


# Load the default word pool
try:
    WORD_POOL = load_wordlist(WORD_LIST_FILE)
except FileNotFoundError:
    # If wordlist.txt doesn't exist, provide a minimal default
    logger.warning(f"{WORD_LIST_FILE} not found, using minimal default word pool")
    WORD_POOL = [
        "AFRICA", "AGENT", "AIR", "ALIEN", "ALPS", "AMAZON", "AMBULANCE", "AMERICA",
        "ANGEL", "ANTARCTICA", "APPLE", "ARM", "ATLANTIS", "AUSTRALIA", "AZTEC",
        "BACK", "BALL", "BAND", "BANK", "BAR", "BARK", "BAT", "BATTERY", "BEACH",
        "BEAR", "BEAT", "BED", "BEIJING", "BELL", "BELT", "BERLIN", "BERMUDA",
        "BERRY", "BILL", "BLOCK", "BOARD", "BOLT", "BOMB", "BOND", "BOOM",
        "BOOT", "BOTTLE", "BOW", "BOX", "BRIDGE", "BRUSH", "BUCK", "BUFFALO",
        "BUG", "BUGLE", "BUTTON", "CALF", "CANADA", "CAP", "CAPITAL", "CAR",
        "CARD", "CARROT", "CASINO", "CAST", "CAT", "CELL", "CENTAUR", "CENTER",
        "CHAIR", "CHANGE", "CHARGE", "CHECK", "CHEST", "CHICK", "CHINA", "CHOCOLATE",
        "CHURCH", "CIRCLE", "CLIFF", "CLOAK", "CLUB", "CODE", "COLD", "COMIC",
        "COMPOUND", "CONCERT", "CONDUCTOR", "CONTRACT", "COOK", "COPPER", "COTTON",
        "COURT", "COVER", "CRANE", "CRASH", "CRICKET", "CROSS", "CROWN", "CYCLE",
        "CZECH", "DANCE", "DATE", "DAY", "DEATH", "DECK", "DEGREE", "DIAMOND",
        "DICE", "DINOSAUR", "DISEASE", "DOCTOR", "DOG", "DRAFT", "DRAGON", "DRESS",
        "DRILL", "DROP", "DUCK", "DWARF", "EAGLE", "EGYPT", "EMBASSY", "ENGINE",
        "ENGLAND", "EUROPE", "EYE", "FACE", "FAIR", "FALL", "FAN", "FENCE"
    ]
