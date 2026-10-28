"""Query analysis and preprocessing"""

from typing import Dict, List
from dataclasses import dataclass
import re


@dataclass
class QueryAnalysis:
    """Result of query analysis"""
    original: str
    cleaned: str
    intent: str
    keywords: List[str]
    entities: List[str]
    length: int
    is_question: bool


class QueryAnalyzer:
    """Analyze queries for better retrieval"""

    def __init__(self):
        self.entity_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "url": r"https?://[^\s]+",
            "number": r"\b\d+(?:,\d{3})*(?:\.\d+)?\b",
            "date": r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
        }

    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze query and extract components.

        Returns:
            QueryAnalysis with detailed breakdown
        """
        # Clean query
        cleaned = self._clean_text(query)

        # Extract keywords
        keywords = self._extract_keywords(cleaned)

        # Detect entities
        entities = self._extract_entities(query)

        # Detect intent
        intent = self._detect_intent(cleaned)

        # Check if question
        is_question = query.rstrip().endswith("?")

        analysis = QueryAnalysis(
            original=query,
            cleaned=cleaned,
            intent=intent,
            keywords=keywords,
            entities=entities,
            length=len(cleaned.split()),
            is_question=is_question
        )

        return analysis

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove punctuation except important markers
        text = re.sub(r'[^\w\s\-\?]', '', text)

        # Convert to lowercase
        text = text.lower()

        return text

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords"""
        # Simple approach: words longer than 3 chars
        tokens = text.split()
        keywords = [t for t in tokens if len(t) > 3 and not t.startswith('-')]
        return list(set(keywords))[:10]

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities (emails, URLs, numbers, dates)"""
        entities = []

        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text)
            entities.extend(matches)

        # Also extract capitalized sequences (proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(capitalized)

        return list(set(entities))

    def _detect_intent(self, query: str) -> str:
        """Detect query intent"""
        query_lower = query.lower()

        if any(q in query_lower for q in ["what", "definition", "meaning"]):
            return "definition"
        elif any(q in query_lower for q in ["how", "steps", "guide"]):
            return "how-to"
        elif any(q in query_lower for q in ["where", "location", "find"]):
            return "location"
        elif any(q in query_lower for q in ["when", "time", "date"]):
            return "temporal"
        elif any(q in query_lower for q in ["why", "reason", "cause"]):
            return "reason"
        else:
            return "informational"

    def is_complex_query(self, query: str) -> bool:
        """Determine if query is complex (might need optimization)"""
        analysis = self.analyze(query)

        # Complex if:
        # 1. Very short (likely needs expansion)
        if analysis.length < 3:
            return True

        # 2. Has multiple question words
        question_words = {"what", "who", "where", "when", "why", "how"}
        q_words_count = sum(1 for kw in analysis.keywords if kw in question_words)
        if q_words_count > 1:
            return True

        # 3. Has AND/OR operators
        if any(op in query.lower() for op in [" and ", " or ", " not "]):
            return True

        return False

    def suggest_strategy(self, query: str) -> str:
        """Suggest retrieval strategy for query"""
        analysis = self.analyze(query)

        if analysis.length < 3:
            return "expand"  # Short query needs expansion
        elif analysis.is_question and len(analysis.keywords) < 2:
            return "expand"  # Question with few keywords
        elif len(analysis.entities) > 0:
            return "hybrid"  # Entity-based, use both semantic + lexical
        else:
            return "semantic"  # Default to semantic for general queries
