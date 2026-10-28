"""Query optimization techniques"""

from typing import List, Set
import re


class QueryOptimizer:
    """
    Query optimization to improve retrieval quality.

    Techniques:
    1. Query rewriting - clarify ambiguous queries
    2. Query expansion - add synonyms/related terms
    3. Stop word removal - focus on important terms
    """

    def __init__(self):
        self.stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "but", "by",
            "for", "from", "had", "has", "have", "he", "her", "hers",
            "him", "his", "how", "i", "if", "in", "is", "it", "its",
            "me", "my", "of", "on", "or", "she", "so", "than", "that",
            "the", "to", "too", "was", "what", "which", "who", "will",
            "with", "you", "your"
        }

        # Simple synonym mapping
        self.synonyms = {
            "what": ["why", "how"],
            "where": ["location", "place", "site"],
            "when": ["time", "date", "period"],
            "who": ["person", "people", "someone"],
            "how": ["method", "way", "process"],
            "big": ["large", "huge", "size"],
            "small": ["tiny", "little", "mini"],
            "good": ["great", "excellent", "positive"],
            "bad": ["poor", "negative", "terrible"],
            "fast": ["quick", "rapid", "speed"],
            "slow": ["gradual", "leisurely", "delay"],
        }

    def remove_stop_words(self, query: str) -> str:
        """Remove common stop words"""
        tokens = query.lower().split()
        filtered = [t for t in tokens if t not in self.stop_words]
        return " ".join(filtered)

    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with synonyms and related terms.

        Returns list of expanded queries for ensemble retrieval.
        """
        expanded = [query]  # Original query

        tokens = query.lower().split()

        # Add expansions with synonyms
        for token in tokens:
            if token in self.synonyms:
                for synonym in self.synonyms[token]:
                    expanded_query = query.replace(token, synonym, 1)
                    if expanded_query not in expanded:
                        expanded.append(expanded_query)

        return expanded[:5]  # Limit to 5 variants

    def rewrite_query(self, query: str) -> str:
        """
        Rewrite query for clarity.

        Simple heuristics to improve search quality.
        """
        # Remove extra spaces
        query = re.sub(r'\s+', ' ', query).strip()

        # Convert to lowercase for consistency
        query_lower = query.lower()

        # Add implicit wildcards for better matching
        # "machine learning" -> search for variations
        tokens = query_lower.split()

        # Handle question marks
        if query.endswith("?"):
            query = query.rstrip("?").strip()

        return query

    def detect_intent(self, query: str) -> str:
        """
        Detect query intent for routing.

        Returns: "factual", "how-to", "comparison", "opinion"
        """
        query_lower = query.lower()

        # Factual questions
        if any(q in query_lower for q in ["what is", "which", "who", "where", "when"]):
            return "factual"

        # How-to questions
        if any(q in query_lower for q in ["how", "how to", "steps to", "guide"]):
            return "how-to"

        # Comparison questions
        if any(q in query_lower for q in ["vs", "versus", "difference", "compare"]):
            return "comparison"

        # Opinion/discussion
        if any(q in query_lower for q in ["opinion", "think", "discuss", "debate"]):
            return "opinion"

        # Default
        return "factual"

    def optimize(self, query: str, strategy: str = "expand") -> List[str]:
        """
        Apply optimization strategy.

        Args:
            query: Original query
            strategy: "rewrite", "expand", "remove_stopwords", or "all"

        Returns:
            List of queries (for ensemble retrieval)
        """
        if strategy == "rewrite":
            return [self.rewrite_query(query)]

        elif strategy == "expand":
            return self.expand_query(query)

        elif strategy == "remove_stopwords":
            cleaned = self.remove_stop_words(query)
            return [cleaned] if cleaned else [query]

        elif strategy == "all":
            rewritten = self.rewrite_query(query)
            expanded = self.expand_query(rewritten)
            cleaned = [self.remove_stop_words(q) for q in expanded]
            return list(set(expanded + cleaned))

        return [query]
