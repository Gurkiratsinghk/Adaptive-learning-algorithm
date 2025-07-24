import re
from bs4 import BeautifulSoup

def preprocess_text(text: str) -> list[str]:
    """
    Preprocess German text by removing HTML, tokenizing, and cleaning.
    Returns a list of cleaned words.
    """
    if not isinstance(text, str):
        return []
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    text = text.lower()
    text = re.sub(r'[^\w\säöüß]', ' ', text)
    words = [word.strip() for word in text.split() if len(word.strip()) > 1]
    return words

def extract_word_pairs(text: str) -> list[tuple[str, str]]:
    words = preprocess_text(text)
    return [(words[i], words[i+1]) for i in range(len(words)-1)]

def generate_sample_data() -> list[str]:
    return [
        "Der schnelle braune Fuchs springt über dem faulen Hund.",
        "Die deutsche Sprache ist sehr interessant und komplex.",
        "Berlin ist die Hauptstadt von Deutschland und sehr schön.",
        "Das Wetter heute ist sonnig und warm in München.",
        "Ich liebe deutsche Literatur und deutsche Musik sehr."
    ]
