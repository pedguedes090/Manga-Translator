"""
Context Memory Module for Manga Translation

Provides rolling context management with:
- Story summary generation (rolling, updated per batch)
- Terms dictionary (character names, skills, places)
- Token-efficient context formatting
- Term usage tracking for prioritization
"""
import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class ContextMemory:
    """
    Manages context memory for a translation session.
    Tracks story progression, character names, and important terms
    to ensure consistent translation across all pages.
    """
    
    # Patterns for detecting proper nouns (names, places, skills)
    # These are common patterns in manga/manhwa/manhua
    NAME_PATTERNS = [
        # Japanese honorifics
        r'(\w+)[-\s]?(さん|君|くん|ちゃん|様|さま|先生|先輩|後輩|殿)',
        # Korean honorifics  
        r'(\w+)[-\s]?(씨|님|선배|후배|형|누나|오빠|언니)',
        # Chinese titles
        r'(\w+)[-\s]?(先生|小姐|大人|师父|师兄|师姐)',
        # Capitalized words (often names in romaji/english)
        r'\b([A-Z][a-z]{2,})\b',
    ]
    
    # Max items to keep in rolling context
    MAX_RECENT_PAGES = 5
    MAX_TERMS = 50
    MAX_SUMMARY_LENGTH = 200
    
    def __init__(self):
        """Initialize empty context memory."""
        self.story_summary = ""
        self.terms: Dict[str, str] = {}  # {original: translated}
        self.term_usage: Dict[str, int] = defaultdict(int)  # {term: count}
        self.recent_translations: List[Dict] = []  # [{page: name, texts: [...]}]
        self.character_names: Dict[str, str] = {}  # {original_name: translated_name}
        
    def update_from_translation(
        self, 
        original_texts: Dict[str, List[str]], 
        translated_texts: Dict[str, List[str]],
    ):
        """
        Update context after a batch translation.
        
        Args:
            original_texts: Dict of {page_name: [original texts]}
            translated_texts: Dict of {page_name: [translated texts]}
        """
        # Extract and update terms
        for page_name in original_texts:
            orig_list = original_texts.get(page_name, [])
            trans_list = translated_texts.get(page_name, [])
            
            # Store recent translations (keep rolling window)
            self.recent_translations.append({
                'page': page_name,
                'original': orig_list,
                'translated': trans_list
            })
            
            # Keep only recent pages
            if len(self.recent_translations) > self.MAX_RECENT_PAGES:
                self.recent_translations = self.recent_translations[-self.MAX_RECENT_PAGES:]
            
            # Extract terms from this page
            self._extract_terms_from_page(orig_list, trans_list)
        
        # Update story summary based on recent content
        self._update_story_summary()
    
    def _extract_terms_from_page(
        self, 
        original_texts: List[str], 
        translated_texts: List[str]
    ):
        """
        Extract potential terms (names, skills, places) from translations.
        Uses heuristics to identify proper nouns and track their translations.
        """
        if len(original_texts) != len(translated_texts):
            return
        
        for orig, trans in zip(original_texts, translated_texts):
            if not orig or not trans:
                continue
            
            # Find potential names using patterns
            for pattern in self.NAME_PATTERNS:
                matches = re.findall(pattern, orig)
                for match in matches:
                    # Get the base name (first capture group if tuple)
                    name = match[0] if isinstance(match, tuple) else match
                    if len(name) >= 2:  # Avoid single characters
                        self.term_usage[name] += 1
            
            # Track repeated short phrases (likely names or terms)
            # Split into words and track short unique segments
            orig_words = orig.split()
            for word in orig_words:
                # Skip common particles and short words
                if len(word) >= 2 and not self._is_common_word(word):
                    self.term_usage[word] += 1
    
    def _is_common_word(self, word: str) -> bool:
        """Check if a word is a common particle/conjunction to ignore."""
        common_ja = {'は', 'が', 'を', 'に', 'の', 'で', 'と', 'も', 'や', 'か', 
                     'から', 'まで', 'より', 'など', 'だ', 'です', 'ます'}
        common_ko = {'은', '는', '이', '가', '을', '를', '의', '에', '와', '과',
                     '도', '만', '부터', '까지', '에서', '로', '으로'}
        common_zh = {'的', '了', '是', '在', '有', '和', '与', '或', '但', '而',
                     '就', '也', '都', '要', '会', '能', '可以'}
        return word in common_ja | common_ko | common_zh
    
    def _update_story_summary(self):
        """
        Generate a brief story summary from recent translations.
        This is a lightweight summarization without LLM calls.
        """
        if not self.recent_translations:
            self.story_summary = ""
            return
        
        # Collect all recent translated texts
        all_texts = []
        for entry in self.recent_translations[-3:]:  # Last 3 pages
            all_texts.extend(entry.get('translated', []))
        
        # Create a simple summary: first few significant lines
        significant_lines = []
        for text in all_texts:
            if text and len(text) > 10:  # Skip very short texts
                significant_lines.append(text)
                if len(significant_lines) >= 5:
                    break
        
        if significant_lines:
            self.story_summary = " | ".join(significant_lines[:3]) + "..."
        else:
            self.story_summary = ""
    
    def add_term(self, original: str, translated: str, is_name: bool = False):
        """
        Manually add a term to the dictionary.
        
        Args:
            original: Original text
            translated: Translated text
            is_name: Whether this is a character name
        """
        self.terms[original] = translated
        self.term_usage[original] += 5  # Boost priority for manual terms
        
        if is_name:
            self.character_names[original] = translated
    
    def get_priority_terms(self, max_count: int = 20) -> List[Tuple[str, int]]:
        """
        Get the most frequently used terms.
        
        Returns:
            List of (term, count) tuples sorted by frequency
        """
        sorted_terms = sorted(
            self.term_usage.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_terms[:max_count]
    
    def generate_context_prompt(self) -> str:
        """
        Generate a context prompt for the translator.
        
        Returns:
            Formatted context string for inclusion in translation prompt
        """
        sections = []
        
        # 1. Character names (highest priority)
        if self.character_names:
            names_text = "CHARACTER NAMES (use consistently):\n"
            for orig, trans in list(self.character_names.items())[:15]:
                names_text += f"  {orig} → {trans}\n"
            sections.append(names_text)
        
        # 2. Important terms (by frequency)
        priority_terms = self.get_priority_terms(20)
        if priority_terms:
            # Filter to only high-frequency terms (appeared 2+ times)
            frequent_terms = [(t, c) for t, c in priority_terms if c >= 2]
            if frequent_terms:
                terms_text = "KEY TERMS (translate consistently):\n"
                for term, count in frequent_terms[:15]:
                    if term in self.terms:
                        terms_text += f"  {term} → {self.terms[term]}\n"
                    else:
                        terms_text += f"  {term} (×{count})\n"
                sections.append(terms_text)
        
        # 3. Story summary
        if self.story_summary:
            summary_text = f"STORY SO FAR:\n  {self.story_summary}\n"
            sections.append(summary_text)
        
        if not sections:
            return ""
        
        return "---\nCONTEXT MEMORY:\n" + "".join(sections) + "---\n"
    
    def get_terms_for_extraction_prompt(self) -> str:
        """
        Generate a prompt section asking the LLM to extract new terms.
        This can be appended to translation requests.
        """
        return """
After translating, also identify any NEW character names, skill names, or important terms.
Add them at the end in this format:
[TERMS]
original1 → translated1
original2 → translated2
[/TERMS]
"""
    
    def parse_extracted_terms(self, response_text: str) -> Dict[str, str]:
        """
        Parse terms extracted by the LLM from the response.
        
        Args:
            response_text: Full LLM response that may contain [TERMS] section
            
        Returns:
            Dict of {original: translated} terms found
        """
        extracted = {}
        
        # Look for [TERMS]...[/TERMS] section
        match = re.search(r'\[TERMS\](.*?)\[/TERMS\]', response_text, re.DOTALL)
        if match:
            terms_section = match.group(1).strip()
            for line in terms_section.split('\n'):
                line = line.strip()
                if '→' in line:
                    parts = line.split('→', 1)
                    if len(parts) == 2:
                        orig = parts[0].strip()
                        trans = parts[1].strip()
                        if orig and trans:
                            extracted[orig] = trans
        
        # Add extracted terms to our dictionary
        for orig, trans in extracted.items():
            self.add_term(orig, trans, is_name=True)
        
        return extracted
    
    def clear(self):
        """Reset context memory to initial state."""
        self.story_summary = ""
        self.terms.clear()
        self.term_usage.clear()
        self.recent_translations.clear()
        self.character_names.clear()
    
    def get_stats(self) -> Dict:
        """Get statistics about the current context memory state."""
        return {
            'total_terms': len(self.terms),
            'character_names': len(self.character_names),
            'tracked_words': len(self.term_usage),
            'recent_pages': len(self.recent_translations),
            'has_summary': bool(self.story_summary)
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"ContextMemory(terms={stats['total_terms']}, "
                f"names={stats['character_names']}, "
                f"pages={stats['recent_pages']})")
