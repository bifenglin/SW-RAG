from __future__ import annotations

import re
from typing import Any, List, Optional

from langchain_text_splitters.base import Language, TextSplitter

class TextSplitter:
    def __init__(self, keep_separator: bool = True, **kwargs: Any):
        self._keep_separator = keep_separator
        self._chunk_size = 100  # Assuming some default chunk size for example
        self._length_function = len  # Default length function

def _split_text_with_regex(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]

class FiexedSizeFixedStepSplitter(TextSplitter):
    """Splitting text by recursively looking at characters and dynamically choosing separators."""

    def __init__(
            self, separators: Optional[List[str]] = None, keep_separator: bool = True,
            is_separator_regex: bool = False, step_window: int = 50, **kwargs: Any):
        """Initialize the splitter with a list of separators and options."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or ["\n\n", "\n", " ", "", ".", "。", "!"]
        self._is_separator_regex = is_separator_regex
        self._step_window = step_window

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks by dynamically choosing the best separator."""
        final_chunks = []
        splits = []

        # Try to split text using the given separators
        for separator in separators:
            regex_separator = separator if self._is_separator_regex else re.escape(separator)
            if re.search(regex_separator, text):
                splits = _split_text_with_regex(text, regex_separator, self._keep_separator)
                break

        # If no appropriate separator is found, use the whole text
        if not splits:
            splits = [text]

        # Create chunks with fixed step size
        temp = 0
        temp_splits = []
        for start in range(0, len(text), self._step_window):
            end = start + self._chunk_size
            chunk = text[start:end]
            final_chunks.append(chunk)

        return final_chunks

    def split_text(self, text: str) -> List[str]:
        """Public method to split text using configured separators."""
        return self._split_text(text, self._separators)
