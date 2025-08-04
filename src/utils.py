import re
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple, Optional
import logging

# SpaCy integration with fallback
try:
    import spacy
    from spacy.lang.de import German
    SPACY_AVAILABLE = True
    
    # Try to load German model with fallback
    try:
        nlp = spacy.load("de_core_news_sm")
    except OSError:
        try:
            nlp = spacy.load("de_core_news_lg")
        except OSError:
            # Fallback to basic German tokenizer
            nlp = German()
            logging.warning("No German SpaCy model found. Using basic tokenizer.")
    
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None
    logging.warning("SpaCy not available. Using basic text processing.")

def preprocess_text(text: str, use_spacy: bool = True) -> List[str]:
    """
    Preprocess German text by removing HTML, tokenizing, and cleaning.
    Returns a list of cleaned words with optional SpaCy enhancement.
    """
    if not isinstance(text, str):
        return []
    
    # Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    
    if use_spacy and SPACY_AVAILABLE:
        return _preprocess_with_spacy(text)
    else:
        return _preprocess_basic(text)

def _preprocess_with_spacy(text: str) -> List[str]:
    """SpaCy-enhanced preprocessing"""
    doc = nlp(text.lower())
    words = []
    
    for token in doc:
        # Skip punctuation, spaces, and very short tokens
        if not token.is_punct and not token.is_space and len(token.text.strip()) > 1:
            # Use lemma if available, otherwise use token text
            word = token.lemma_ if token.lemma_ != "-PRON-" else token.text
            words.append(word.strip())
    
    return words

def _preprocess_basic(text: str) -> List[str]:
    """Basic preprocessing (fallback)"""
    text = text.lower()
    text = re.sub(r'[^\w\säöüß]', ' ', text)
    words = [word.strip() for word in text.split() if len(word.strip()) > 1]
    return words

def extract_enhanced_word_pairs(text: str) -> List[Dict]:
    """
    Extract word pairs with linguistic metadata using SpaCy.
    Returns enhanced word pair data for anchor node detection.
    """
    if not SPACY_AVAILABLE:
        # Fallback to basic pairs
        basic_pairs = extract_word_pairs(text)
        return [{'word1': w1, 'word2': w2, 'word1_pos': 'UNKNOWN', 'word2_pos': 'UNKNOWN',
                'is_sentence_start': False, 'dependency_relation': 'UNKNOWN'} 
                for w1, w2 in basic_pairs]
    
    if not isinstance(text, str):
        return []
    
    # Clean text
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    doc = nlp(text)
    
    enhanced_pairs = []
    
    for sent in doc.sents:
        tokens = [token for token in sent if not token.is_punct and not token.is_space and len(token.text.strip()) > 1]
        
        for i in range(len(tokens) - 1):
            token1, token2 = tokens[i], tokens[i + 1]
            
            pair_data = {
                'word1': token1.lemma_ if token1.lemma_ != "-PRON-" else token1.text.lower(),
                'word2': token2.lemma_ if token2.lemma_ != "-PRON-" else token2.text.lower(),
                'word1_pos': token1.pos_,
                'word2_pos': token2.pos_,
                'word1_lemma': token1.lemma_,
                'word2_lemma': token2.lemma_,
                'sentence_position': i,
                'dependency_relation': token2.dep_,
                'is_sentence_start': i == 0,
                'sentence_length': len(tokens)
            }
            enhanced_pairs.append(pair_data)
    
    return enhanced_pairs

def extract_word_pairs(text: str) -> List[Tuple[str, str]]:
    """
    Basic word pair extraction (backward compatibility).
    """
    words = preprocess_text(text, use_spacy=False)
    return [(words[i], words[i+1]) for i in range(len(words)-1)]

def identify_sentence_boundaries(text: str) -> List[str]:
    """
    Extract sentences using SpaCy sentence segmentation.
    """
    if not SPACY_AVAILABLE:
        # Basic fallback using punctuation
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    if not isinstance(text, str):
        return []
    
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def filter_content_words(tokens: List[str]) -> List[Dict]:
    """
    Filter tokens to content words (nouns, verbs, adjectives, adverbs)
    and preserve positional information.
    """
    if not SPACY_AVAILABLE:
        # Basic fallback - remove common function words
        function_words = {'der', 'die', 'das', 'und', 'oder', 'aber', 'wenn', 'dann', 
                         'ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr', 'in', 'auf', 'zu'}
        return [{'word': token, 'position': i, 'pos': 'UNKNOWN'} 
                for i, token in enumerate(tokens) if token.lower() not in function_words]
    
    # Join tokens back to process with SpaCy
    text = ' '.join(tokens)
    doc = nlp(text)
    
    content_words = []
    content_pos_tags = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}
    
    for i, token in enumerate(doc):
        if (token.pos_ in content_pos_tags and 
            not token.is_punct and 
            not token.is_space and 
            len(token.text.strip()) > 1):
            
            content_words.append({
                'word': token.lemma_ if token.lemma_ != "-PRON-" else token.text.lower(),
                'position': i,
                'pos': token.pos_,
                'is_content': True
            })
    
    return content_words

def calculate_anchor_score(word_data: Dict, frequency_data: Dict) -> float:
    """
    Calculate anchor node score based on linguistic features and frequency.
    Higher scores indicate better anchor candidates.
    """
    if not SPACY_AVAILABLE:
        # Basic scoring based on frequency only
        freq = frequency_data.get(word_data.get('word', ''), 0)
        return freq / max(sum(frequency_data.values()), 1)
    
    word = word_data.get('word', '')
    pos = word_data.get('pos', 'UNKNOWN')
    is_sentence_start = word_data.get('is_sentence_start', False)
    
    # Base frequency score (normalized)
    freq = frequency_data.get(word, 0)
    total_freq = max(sum(frequency_data.values()), 1)
    freq_score = freq / total_freq
    
    # POS-based scoring (German language specific)
    pos_weights = {
        'DET': 0.9,    # Articles (der, die, das) - strong anchors
        'PRON': 0.8,   # Pronouns (ich, du, er, sie) - good anchors
        'NOUN': 0.7,   # Nouns - moderate anchors
        'VERB': 0.6,   # Verbs - can be anchors
        'ADJ': 0.3,    # Adjectives - weak anchors
        'ADV': 0.2,    # Adverbs - weak anchors
        'ADP': 0.5,    # Prepositions - moderate anchors
        'UNKNOWN': 0.1
    }
    pos_score = pos_weights.get(pos, 0.1)
    
    # Sentence position bonus
    position_bonus = 0.2 if is_sentence_start else 0.0
    
    # Combined score
    anchor_score = (0.5 * freq_score + 0.3 * pos_score + 0.2 * position_bonus)
    
    return min(anchor_score, 1.0)  # Cap at 1.0

def generate_sample_data() -> List[str]:
    """Generate sample German sentences for testing"""
    return [
        "Der schnelle braune Fuchs springt über dem faulen Hund.",
        "Die deutsche Sprache ist sehr interessant und komplex.",
        "Berlin ist die Hauptstadt von Deutschland und sehr schön.",
        "Das Wetter heute ist sonnig und warm in München.",
        "Ich liebe deutsche Literatur und deutsche Musik sehr.",
        "Die Kinder spielen im Park und haben viel Spaß.",
        "Der Lehrer erklärt die Grammatik sehr deutlich und geduldig.",
        "In Deutschland gibt es viele schöne Städte und Landschaften."
    ]

def get_spacy_status() -> Dict:
    """Return SpaCy availability and model information"""
    if not SPACY_AVAILABLE:
        return {'available': False, 'model': None, 'reason': 'SpaCy not installed'}
    
    if nlp is None:
        return {'available': False, 'model': None, 'reason': 'No German model found'}
    
    model_name = nlp.meta.get('name', 'unknown')
    return {
        'available': True, 
        'model': model_name,
        'version': nlp.meta.get('version', 'unknown'),
        'language': 'German'
    }