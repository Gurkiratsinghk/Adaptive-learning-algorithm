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
    """
    Extract consecutive word pairs from text.
    """
    words = preprocess_text(text)
    return [(words[i], words[i+1]) for i in range(len(words)-1)]

def extract_sentence_starts(text: str) -> list[str]:
    """
    Extract words that typically start sentences.
    This helps identify potential anchor nodes.
    """
    if not isinstance(text, str):
        return []
    
    # Clean text but preserve sentence boundaries
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    
    # Split by sentence-ending punctuation
    sentences = re.split(r'[.!?]+\s+', text)
    
    sentence_starts = []
    for sentence in sentences:
        if sentence.strip():
            # Get first word of each sentence
            words = preprocess_text(sentence)
            if words:
                sentence_starts.append(words[0])
    
    return sentence_starts

def extract_word_pairs_with_sentence_info(text: str) -> tuple[list[tuple[str, str]], list[str]]:
    """
    Extract both word pairs and sentence start words.
    Returns: (word_pairs, sentence_starts)
    """
    word_pairs = extract_word_pairs(text)
    sentence_starts = extract_sentence_starts(text)
    return word_pairs, sentence_starts

def generate_sample_data() -> list[str]:
    """
    Generate sample German sentences for testing.
    """
    return [
        "Der schnelle braune Fuchs springt über dem faulen Hund.",
        "Die deutsche Sprache ist sehr interessant und komplex.",
        "Berlin ist die Hauptstadt von Deutschland und sehr schön.",
        "Das Wetter heute ist sonnig und warm in München.",
        "Ich liebe deutsche Literatur und deutsche Musik sehr.",
        "Ein gutes Buch kann das Leben verändern und inspirieren.",
        "Der Zug fährt pünktlich um acht Uhr am Morgen ab.",
        "Die Kinder spielen fröhlich im Park mit ihren Freunden.",
        "Eine schöne Reise durch Europa ist sehr bereichernd.",
        "Das Museum zeigt interessante Kunstwerke aus verschiedenen Epochen."
    ]