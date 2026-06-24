"""Evidence-based review log data structures for the simplified review agent."""

import logging
from typing import Any, List, Optional, Dict, Literal, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator

logger = logging.getLogger(__name__)


class Claim(BaseModel):
    """A claim extracted from the paper to be verified."""

    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(description="Unique identifier (C1, C2, ...)")
    text: str = Field(description="The claim statement")
    section: str = Field(description="Section where claim originates")
    type: str = Field(
        description="Type of claim: e.g. empirical, theoretical, novelty, etc."
    )
    status: Literal["to_be_verified", "supported", "weak", "invalid"] = Field(
        default="to_be_verified",
        description="Verification status: to_be_verified (not checked), supported (verified as valid), weak (partially supported), invalid (contradicted by evidence)"
    )
    issues: List[str] = Field(
        default_factory=list,
        description="List of identified issues or weaknesses with this claim"
    )
    cross_references: List[str] = Field(
        default_factory=list,
        description="Sections that provide supporting evidence or contradictions"
    )
    verifier_reason: Optional[str] = Field(
        default=None,
        description="Reasoning from the verifier about the status, including cross-section analysis"
    )
    step: Optional[int] = Field(
        default=None,
        description="Step number when this claim was created"
    )
    status_updated_step: Optional[int] = Field(
        default=None,
        description="Step number when this claim's status was last updated"
    )

    def to_prompt_str(self, detailed: bool = False) -> str:
        """Format claim for LLM context.

        Args:
            detailed: If True, include verifier reasoning; if False, brief format

        Returns:
            Formatted string representation
        """
        status_emoji = {
            "to_be_verified": "?",
            "supported": "✓",
            "weak": "~",
            "invalid": "✗"
        }

        parts = [f"[{status_emoji[self.status]}] {self.id}: {self.text}"]
        parts.append(f"(from {self.section}, {self.type})")

        if self.issues:
            issues_str = "; ".join(self.issues)
            parts.append(f"Issues: {issues_str}")

        if self.cross_references:
            xrefs = ", ".join(self.cross_references)
            parts.append(f"Cross-refs: {xrefs}")

        if detailed and self.verifier_reason:
            parts.append(f"Verification: {self.verifier_reason}")

        return " ".join(parts)


class Question(BaseModel):
    """A question or suspicion that arose during reading."""

    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(description="Unique identifier (Q1, Q2, ...)")
    question: str = Field(description="The question or concern")
    source_section: str = Field(description="Section where question arose")
    status: Literal["open", "partially_answered", "resolved"] = Field(
        default="open",
        description="Status: open (unanswered), partially_answered (some info found), resolved (fully answered)"
    )
    type: str = Field(
        description="Type of questions: e.g. clarification, methodology, novelty, presentation,reproducibility etc."
    )
    answer: Optional[str] = Field(
        default=None,
        description="The answer if found"
    )
    answer_sections: List[str] = Field(
        default_factory=list,
        description="Sections that provide answers or relevant information"
    )
    related_claims: List[str] = Field(
        default_factory=list,
        description="Claim IDs that this question relates to"
    )
    step: Optional[int] = Field(
        default=None,
        description="Step number when this question was created"
    )

    def to_prompt_str(self, detailed: bool = False) -> str:
        """Format question for LLM context.

        Args:
            detailed: If True, include answer details; if False, brief format

        Returns:
            Formatted string representation
        """
        status_emoji = {
            "open": "?",
            "partially_answered": "~",
            "resolved": "✓"
        }

        parts = [f"[{status_emoji[self.status]}] {self.id}: {self.question}"]
        parts.append(f"(from {self.source_section})")

        if self.related_claims:
            claims_str = ", ".join(self.related_claims)
            parts.append(f"Related to: {claims_str}")

        if self.answer_sections:
            sections_str = ", ".join(self.answer_sections)
            parts.append(f"Info in: {sections_str}")

        if detailed and self.answer:
            parts.append(f"Answer: {self.answer}")

        return " ".join(parts)


class Note(BaseModel):
    """A note or thought triggered during reading."""

    id: str = Field(description="Unique identifier (N1, N2, ...)")
    text: str = Field(description="The note text")
    section: str = Field(description="Section where note was triggered")
    tag: List[str] = Field(
        default_factory=list,
        description="Flexible tags (e.g., presentation, clarity, novelty, methodology)"
    )
    step: Optional[int] = Field(
        default=None,
        description="Step number when this note was created"
    )

    def to_prompt_str(self) -> str:
        """Format note for LLM context.

        Returns:
            Formatted string representation
        """
        tags_str = ", ".join(self.tag) if self.tag else "untagged"
        return f"{self.id}: {self.text} (from {self.section}, tags: {tags_str})"


class OutlineItem(BaseModel):
    """A point in the review outline with evidence references."""

    text: str = Field(description="The strength/weakness/question text")
    related_claims: List[str] = Field(
        default_factory=list,
        description="Claim IDs this point is based on (e.g., ['C1', 'C3'])"
    )
    related_questions: List[str] = Field(
        default_factory=list,
        description="Question IDs this point relates to (e.g., ['Q2'])"
    )
    related_notes: List[str] = Field(
        default_factory=list,
        description="Note IDs this point references (e.g., ['N5'])"
    )
    step: Optional[int] = Field(
        default=None,
        description="Step number when this outline item was created"
    )
    human_point: Optional[dict] = Field(default=None, description="the information whether this outline matches a human-generated point")

    def to_prompt_str(self, show_tags: bool = True) -> str:
        """Format outline item for LLM context.

        Args:
            show_tags: If True, show evidence tags; if False, just the text

        Returns:
            Formatted string representation
        """
        if not show_tags:
            return self.text

        # Collect all evidence references
        evidence_refs = []
        if self.related_claims:
            evidence_refs.extend(self.related_claims)
        if self.related_questions:
            evidence_refs.extend(self.related_questions)
        if self.related_notes:
            evidence_refs.extend(self.related_notes)

        if evidence_refs:
            refs_str = ", ".join(evidence_refs)
            return f"{self.text} [{refs_str}]"
        return self.text


class ReviewOutline(BaseModel):
    """Outline of the review being constructed - the final verdict."""

    model_config = ConfigDict(validate_assignment=True)

    summary: str = Field(default="", description="Summary of the paper")
    strengths: List[OutlineItem] = Field(default_factory=list, description="List of paper strengths with evidence tags")
    weaknesses: List[OutlineItem] = Field(default_factory=list, description="List of paper weaknesses with evidence tags")
    questions: List[OutlineItem] = Field(default_factory=list, description="Questions for the authors with evidence tags")
    overall_score: Optional[int] = Field(default=None, description="Single overall score")
    summary_step: Optional[int] = Field(default=None, description="Step number when summary was first set")
    overall_score_step: Optional[int] = Field(default=None, description="Step number when overall_score was first set")

    @staticmethod
    def _convert_to_outline_items(items: Union[List[str], List[OutlineItem], List[Dict]]) -> List[OutlineItem]:
        """Convert various formats to List[OutlineItem] for backward compatibility.

        Args:
            items: Can be List[str], List[OutlineItem], or List[Dict]

        Returns:
            List of OutlineItem objects
        """
        result = []
        for item in items:
            if isinstance(item, OutlineItem):
                result.append(item)
            elif isinstance(item, str):
                result.append(OutlineItem(text=item))
            elif isinstance(item, dict):
                # Handle dict with either 'text' key or as a full OutlineItem dict
                if 'text' in item:
                    result.append(OutlineItem(**item))
                else:
                    # Assume it's a plain string in a dict wrapper
                    result.append(OutlineItem(text=str(item)))
            else:
                # Fallback: convert to string
                result.append(OutlineItem(text=str(item)))
        return result

    def set_strengths(self, items: Union[List[str], List[OutlineItem], List[Dict]]):
        """Set strengths with automatic conversion from List[str] to List[OutlineItem]."""
        self.strengths = self._convert_to_outline_items(items)

    def set_weaknesses(self, items: Union[List[str], List[OutlineItem], List[Dict]]):
        """Set weaknesses with automatic conversion from List[str] to List[OutlineItem]."""
        self.weaknesses = self._convert_to_outline_items(items)

    def set_questions(self, items: Union[List[str], List[OutlineItem], List[Dict]]):
        """Set questions with automatic conversion from List[str] to List[OutlineItem]."""
        self.questions = self._convert_to_outline_items(items)

    def get_strengths_text(self) -> List[str]:
        """Get strengths as plain text list (backward compatibility)."""
        return [item.text for item in self.strengths]

    def get_weaknesses_text(self) -> List[str]:
        """Get weaknesses as plain text list (backward compatibility)."""
        return [item.text for item in self.weaknesses]

    def get_questions_text(self) -> List[str]:
        """Get questions as plain text list (backward compatibility)."""
        return [item.text for item in self.questions]

    def to_dict_with_text(self) -> Dict:
        """Convert to dict with text-only lists for backward compatibility.

        Returns:
            Dict with strengths/weaknesses/questions as List[str]
        """
        return {
            "summary": self.summary,
            "strengths": self.get_strengths_text(),
            "weaknesses": self.get_weaknesses_text(),
            "questions": self.get_questions_text(),
            "overall_score": self.overall_score
        }

    def to_dict_with_evidence(self) -> Dict:
        """Convert to dict with full OutlineItem data including evidence tags.

        Returns:
            Dict with strengths/weaknesses/questions as List[Dict] containing evidence
        """
        return {
            "summary": self.summary,
            "strengths": [item.model_dump() for item in self.strengths],
            "weaknesses": [item.model_dump() for item in self.weaknesses],
            "questions": [item.model_dump() for item in self.questions],
            "overall_score": self.overall_score
        }

    @field_validator('strengths', 'weaknesses', 'questions', mode='before')
    @classmethod
    def convert_to_outline_items(cls, v):
        """Automatically convert List[str] or List[Dict] to List[OutlineItem]."""
        if not isinstance(v, list):
            return v

        result = []
        for item in v:
            if isinstance(item, OutlineItem):
                result.append(item)
            elif isinstance(item, str):
                result.append(OutlineItem(text=item))
            elif isinstance(item, dict):
                # Handle dict format
                if 'text' in item:
                    result.append(OutlineItem(**item))
                else:
                    # Assume the whole dict is meant to be converted to string
                    result.append(OutlineItem(text=str(item)))
            else:
                # Fallback for other types
                result.append(OutlineItem(text=str(item)))
        return result

    def to_prompt_str(self, brief: bool = True, show_evidence_tags: bool = True) -> str:
        """Format outline for LLM context.

        Args:
            brief: If True, show only counts and first few items; if False, show all
            show_evidence_tags: If True, show evidence tags (claim/question/note IDs)

        Returns:
            Formatted string representation
        """
        parts = []

        if self.summary:
            summary_preview = self.summary if not brief or len(self.summary) < 100 else self.summary[:97] + "..."
            parts.append(f"Summary: {summary_preview}")

        if self.strengths:
            if brief and len(self.strengths) > 3:
                strengths_str = "; ".join([item.to_prompt_str(show_tags=show_evidence_tags) for item in self.strengths[:3]]) + f" ... [{len(self.strengths)} total]"
            else:
                strengths_str = "; ".join([item.to_prompt_str(show_tags=show_evidence_tags) for item in self.strengths])
            parts.append(f"Strengths: {strengths_str}")

        if self.weaknesses:
            if brief and len(self.weaknesses) > 3:
                weaknesses_str = "; ".join([item.to_prompt_str(show_tags=show_evidence_tags) for item in self.weaknesses[:3]]) + f" ... [{len(self.weaknesses)} total]"
            else:
                weaknesses_str = "; ".join([item.to_prompt_str(show_tags=show_evidence_tags) for item in self.weaknesses])
            parts.append(f"Weaknesses: {weaknesses_str}")

        if self.questions:
            if brief and len(self.questions) > 3:
                questions_str = "; ".join([item.to_prompt_str(show_tags=show_evidence_tags) for item in self.questions[:3]]) + f" ... [{len(self.questions)} total]"
            else:
                questions_str = "; ".join([item.to_prompt_str(show_tags=show_evidence_tags) for item in self.questions])
            parts.append(f"Questions: {questions_str}")

        if self.overall_score is not None:
            parts.append(f"Overall Score: {self.overall_score}")

        return "\n".join(parts) if parts else "No outline yet"


class ReviewLog(BaseModel):
    """Evidence-based review log for the ReviewerR1 agent.

    Collects evidence (claims, questions, notes) during paper review
    to support the final verdict in review_outline.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    claims: List[Claim] = Field(
        default_factory=list,
        description="List of extracted and verified claims"
    )
    questions: List[Question] = Field(
        default_factory=list,
        description="List of questions and suspicions that arose during reading"
    )
    notes: List[Note] = Field(
        default_factory=list,
        description="List of notes and thoughts triggered during reading"
    )
    review_outline: ReviewOutline = Field(
        default_factory=ReviewOutline,
        description="Review outline being constructed - the final verdict"
    )
    section_visits: Dict[str, int] = Field(
        default_factory=dict,
        description="Track number of visits to each section"
    )
    search_history: List[str] = Field(
        default_factory=list,
        description="Track search queries performed (in order)"
    )
    current_iteration: int = Field(
        default=0,
        description="Current iteration number"
    )

    # Duplicate checker (excluded from serialization)
    duplicate_checker_: Optional[Any] = Field(default=None, exclude=True, repr=False)

    # When True, duplicates are silently dropped (no error, no penalty).
    # When False (default), duplicates raise ValueError → mem_error penalty.
    silent_duplicates_: bool = Field(default=False, exclude=True, repr=False)

    def _check_duplicate(self, text: str, entry_type: str) -> bool:
        """Check for duplicate using embedding similarity.

        When silent_duplicates_ is True, returns True on duplicate without
        raising — callers should skip the add. When False, raises ValueError.

        Args:
            text: Text to check
            entry_type: Type of entry (claim, question, note, strength, weakness, question_outline)

        Returns:
            True if duplicate detected (and silently skipped), False otherwise.

        Raises:
            ValueError: If duplicate detected and silent_duplicates_ is False.
        """
        if self.duplicate_checker_ is None:
            return False  # No checker configured, skip

        try:
            result = self.duplicate_checker_.check_and_register(text, entry_type)
        except Exception as e:
            # Embedding service failure -- log warning but don't block the operation
            logger.warning(f"Duplicate check embedding call failed: {e}")
            return False

        if result is not None:
            idx, sim, preview = result
            if self.silent_duplicates_:
                logger.debug(
                    f"Silently skipping duplicate {entry_type} "
                    f"(sim={sim:.2f} to #{idx+1}: '{preview}...')"
                )
                return True
            raise ValueError(
                f"This {entry_type} is too similar to existing {entry_type} "
                f"#{idx+1}: '{preview}...' (similarity: {sim:.2f}). "
                f"Each {entry_type} must be a distinct point."
            )
        return False

    def add_claim(
        self,
        text: str,
        section: str,
        claim_type: str,
        issues: Optional[List[str]] = None,
        step: Optional[int] = None
    ) -> str:
        """Add a new claim to the log.

        Args:
            text: The claim text
            section: Section where claim originates
            claim_type: Type of claim (e.g. empirical, theoretical, novelty)
            issues: Optional list of issues with the claim
            step: Optional step number when this claim was created

        Returns:
            The assigned claim ID (e.g., "C1", "C2")

        Raises:
            ValueError: If claim is too similar to an existing claim
        """
        # Check for duplicates before adding
        if self._check_duplicate(text, "claim"):
            return None  # Silently skipped duplicate

        claim_id = f"C{len(self.claims) + 1}"
        claim = Claim(
            id=claim_id,
            text=text,
            section=section,
            type=claim_type,
            issues=issues or [],
            step=step
        )
        self.claims.append(claim)
        return claim_id

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Retrieve a claim by ID.

        Args:
            claim_id: The claim ID to retrieve

        Returns:
            The Claim object, or None if not found
        """
        for claim in self.claims:
            if claim.id == claim_id:
                return claim
        return None

    def update_claim_status(
        self,
        claim_id: str,
        status: Literal["to_be_verified", "supported", "weak", "invalid"],
        reason: Optional[str] = None,
        cross_refs: Optional[List[str]] = None,
        step: Optional[int] = None
    ) -> bool:
        """Update the verification status of a claim.

        Args:
            claim_id: The claim ID to update
            status: New status
            reason: Optional reasoning for the status
            cross_refs: Optional list of sections that provide evidence
            step: Step number when the status update occurred

        Returns:
            True if claim was found and updated, False otherwise
        """
        claim = self.get_claim(claim_id)
        if claim:
            claim.status = status
            if reason:
                claim.verifier_reason = reason
            if cross_refs:
                claim.cross_references = cross_refs
            if step is not None:
                claim.status_updated_step = step
            return True
        return False

    def add_question(
        self,
        question: str,
        source_section: str,
        question_type: str = "clarification",
        related_claims: Optional[List[str]] = None,
        step: Optional[int] = None
    ) -> str:
        """Add a new question to the log.

        Args:
            question: The question text
            source_section: Section where question arose
            question_type: Type/category of question
            related_claims: Optional list of related claim IDs
            step: Optional step number when this question was created

        Returns:
            The assigned question ID (e.g., "Q1", "Q2")

        Raises:
            ValueError: If question is too similar to an existing question
        """
        # Check for duplicates before adding
        if self._check_duplicate(question, "question"):
            return None  # Silently skipped duplicate

        question_id = f"Q{len(self.questions) + 1}"
        q = Question(
            id=question_id,
            question=question,
            source_section=source_section,
            type=question_type,
            related_claims=related_claims or [],
            step=step
        )
        self.questions.append(q)
        return question_id

    def get_question(self, question_id: str) -> Optional[Question]:
        """Retrieve a question by ID.

        Args:
            question_id: The question ID to retrieve

        Returns:
            The Question object, or None if not found
        """
        for q in self.questions:
            if q.id == question_id:
                return q
        return None

    def resolve_question(
        self,
        question_id: str,
        answer: str,
        answer_sections: List[str],
        status: Literal["partially_answered", "resolved"] = "resolved"
    ) -> bool:
        """Resolve a question with an answer.

        Args:
            question_id: The question ID to resolve
            answer: The answer text
            answer_sections: Sections that provided the answer
            status: New status (partially_answered or resolved)

        Returns:
            True if question was found and updated, False otherwise
        """
        question = self.get_question(question_id)
        if question:
            question.answer = answer
            question.answer_sections = answer_sections
            question.status = status
            return True
        return False

    def get_open_questions(self) -> List[Question]:
        """Get all open or partially answered questions.

        Returns:
            List of questions that are not fully resolved
        """
        return [q for q in self.questions if q.status != "resolved"]

    def add_note(
        self,
        text: str,
        section: str,
        tag: Optional[List[str]] = None,
        step: Optional[int] = None
    ) -> str:
        """Add a new note to the log.

        Args:
            text: The note text
            section: Section where note was triggered
            tag: Optional list of tags
            step: Optional step number when this note was created

        Returns:
            The assigned note ID (e.g., "N1", "N2")

        Raises:
            ValueError: If note is too similar to an existing note
        """
        # Check for duplicates before adding
        if self._check_duplicate(text, "note"):
            return None  # Silently skipped duplicate

        note_id = f"N{len(self.notes) + 1}"
        note = Note(
            id=note_id,
            text=text,
            section=section,
            tag=tag or [],
            step=step
        )
        self.notes.append(note)
        return note_id

    def get_note(self, note_id: str) -> Optional[Note]:
        """Retrieve a note by ID.

        Args:
            note_id: The note ID to retrieve

        Returns:
            The Note object, or None if not found
        """
        for note in self.notes:
            if note.id == note_id:
                return note
        return None

    def _check_outline_duplicate(
        self,
        existing_items: List["OutlineItem"],
        new_item: "OutlineItem",
        section_label: str,
        threshold: float = 0.7,
    ) -> bool:
        """Check if new_item is too similar to any existing item.

        Returns True if duplicate (and silently skipped when silent_duplicates_ is True).
        Raises ValueError if duplicate and silent_duplicates_ is False.

        Uses embedding-based duplicate detection if available, otherwise disabled.
        """
        # Use embedding-based checker if available (threshold param ignored -- uses checker's threshold)
        return self._check_duplicate(new_item.text, section_label)

    def _validate_evidence_tags(
        self,
        related_claims: Optional[List[str]],
        related_questions: Optional[List[str]],
        related_notes: Optional[List[str]],
    ) -> None:
        """Validate that all referenced evidence IDs exist in the current log.

        Raises:
            ValueError: listing the hallucinated IDs so the caller can route
                        the error to mem_error (tool penalty) rather than
                        validation_errors (format penalty).
        """
        valid_claim_ids = {c.id for c in self.claims}
        valid_question_ids = {q.id for q in self.questions}
        valid_note_ids = {n.id for n in self.notes}
        # C1, Q2 etc.
        hallucinated = (
            [c for c in (related_claims) if c not in valid_claim_ids] +
            [q for q in (related_questions) if q not in valid_question_ids] +
            [n for n in (related_notes) if n not in valid_note_ids]
        )
        if hallucinated:
            raise ValueError(
                f"Hallucinated evidence references: {hallucinated}. "
                "Only reference IDs that were confirmed in previous steps."
            )

    def update_outline(
        self,
        section: Literal["summary", "strengths", "weaknesses", "questions", "overall_score"],
        content: Union[str, int, OutlineItem],
        append: bool = True,
        related_claims: Optional[List[str]] = None,
        related_questions: Optional[List[str]] = None,
        related_notes: Optional[List[str]] = None,
        step: Optional[int] = None
    ):
        """Update the review outline.

        Args:
            section: Which section to update
            content: Content to add or set (str for summary/text, int for score, or OutlineItem)
            append: If True, append to list sections; if False, replace
            related_claims: List of claim IDs to associate (required for strengths/weaknesses/questions)
            related_questions: List of question IDs to associate (required for strengths/weaknesses/questions)
            related_notes: List of note IDs to associate (required for strengths/weaknesses/questions)
            step: Optional step number when this outline item was created

        Raises:
            ValueError: If no evidence tags provided for sections that require them
        """
        if section == "summary":
            self.review_outline.summary_step = step
            self.review_outline.summary = str(content)

        elif section == "strengths":
            # Create OutlineItem if content is str
            if isinstance(content, str):
                # Validate that at least one evidence reference is provided
                has_evidence = (
                    (related_claims and len(related_claims) > 0) or
                    (related_questions and len(related_questions) > 0) or
                    (related_notes and len(related_notes) > 0)
                )
                if not has_evidence:
                    raise ValueError(
                        "At least one evidence reference (related_claims, related_questions, or related_notes) "
                        "is required for outline items. Provide tags to link this point to supporting evidence."
                    )
                self._validate_evidence_tags(related_claims, related_questions, related_notes)
                
                item = OutlineItem(
                    text=content,
                    related_claims=related_claims,
                    related_questions=related_questions,
                    related_notes=related_notes,
                    step=step
                )
            else:
                item = content

            if append:
                if self._check_outline_duplicate(self.review_outline.strengths, item, "strength"):
                    return "duplicate_skipped"
                self.review_outline.strengths.append(item)
            else:
                self.review_outline.strengths = [item]

        elif section == "weaknesses":
            # Create OutlineItem if content is str
            if isinstance(content, str):
                # Validate that at least one evidence reference is provided
                has_evidence = (
                    (related_claims and len(related_claims) > 0) or
                    (related_questions and len(related_questions) > 0) or
                    (related_notes and len(related_notes) > 0)
                )
                if not has_evidence:
                    raise ValueError(
                        "At least one evidence reference (related_claims, related_questions, or related_notes) "
                        "is required for outline items. Provide tags to link this point to supporting evidence."
                    )
                self._validate_evidence_tags(related_claims, related_questions, related_notes)

                item = OutlineItem(
                    text=content,
                    related_claims=related_claims or [],
                    related_questions=related_questions or [],
                    related_notes=related_notes or [],
                    step=step
                )
            else:
                item = content

            if append:
                if self._check_outline_duplicate(self.review_outline.weaknesses, item, "weakness"):
                    return "duplicate_skipped"
                self.review_outline.weaknesses.append(item)
            else:
                self.review_outline.weaknesses = [item]

        elif section == "questions":
            # Create OutlineItem if content is str
            if isinstance(content, str):
                # Validate that at least one evidence reference is provided
                has_evidence = (
                    (related_claims and len(related_claims) > 0) or
                    (related_questions and len(related_questions) > 0) or
                    (related_notes and len(related_notes) > 0)
                )
                if not has_evidence:
                    raise ValueError(
                        "At least one evidence reference (related_claims, related_questions, or related_notes) "
                        "is required for outline items. Provide tags to link this point to supporting evidence."
                    )
                self._validate_evidence_tags(related_claims, related_questions, related_notes)

                item = OutlineItem(
                    text=content,
                    related_claims=related_claims or [],
                    related_questions=related_questions or [],
                    related_notes=related_notes or [],
                    step=step
                )
            else:
                item = content

            if append:
                if self._check_outline_duplicate(self.review_outline.questions, item, "question"):
                    return "duplicate_skipped"
                self.review_outline.questions.append(item)
            else:
                self.review_outline.questions = [item]

        elif section == "overall_score":
            self.review_outline.overall_score = int(content) if content is not None else None
            self.review_outline.overall_score_step = step

    def add_outline_evidence(
        self,
        section: Literal["strengths", "weaknesses", "questions"],
        item_index: int,
        related_claims: Optional[List[str]] = None,
        related_questions: Optional[List[str]] = None,
        related_notes: Optional[List[str]] = None
    ) -> bool:
        """Add evidence tags to an existing outline item.

        Args:
            section: Which section the item belongs to
            item_index: Index of the item in the list (0-based)
            related_claims: Claim IDs to add
            related_questions: Question IDs to add
            related_notes: Note IDs to add

        Returns:
            True if successful, False if item_index is out of range
        """
        items_list = None
        if section == "strengths":
            items_list = self.review_outline.strengths
        elif section == "weaknesses":
            items_list = self.review_outline.weaknesses
        elif section == "questions":
            items_list = self.review_outline.questions

        if items_list is None or item_index < 0 or item_index >= len(items_list):
            return False

        item = items_list[item_index]
        if related_claims:
            item.related_claims.extend(related_claims)
        if related_questions:
            item.related_questions.extend(related_questions)
        if related_notes:
            item.related_notes.extend(related_notes)

        return True

    def record_section_visit(self, section_name: str):
        """Record a visit to a section.

        Args:
            section_name: Name of the section visited
        """
        if section_name in self.section_visits:
            self.section_visits[section_name] += 1
        else:
            self.section_visits[section_name] = 1

    def record_search_query(self, query: str):
        """Record a search query.

        Args:
            query: The search query string
        """
        self.search_history.append(query)

    def build_context(self, detailed: bool = False, max_claims: int = 10) -> str:
        """Format log for LLM prompts.

        Args:
            detailed: If True, include full details; if False, brief summary
            max_claims: Maximum number of claims to show in brief mode

        Returns:
            Formatted log context string
        """
        if detailed:
            return self._build_detailed_context()
        else:
            return self._build_brief_context(max_claims)

    def _build_brief_context(self, max_claims: int = 10) -> str:
        """Build brief context for agent decision-making."""
        parts = [f"=== Review Log (Iteration {self.current_iteration}) ===\n"]

        # Claims (show recent ones)
        if self.claims:
            total_claims = len(self.claims)
            claims_to_show = self.claims[-max_claims:] if total_claims > max_claims else self.claims

            # Count by status
            status_counts = {"to_be_verified": 0, "supported": 0, "weak": 0, "invalid": 0}
            for claim in self.claims:
                status_counts[claim.status] += 1

            parts.append(f"Claims ({total_claims} total: {status_counts['supported']}✓, {status_counts['weak']}~, {status_counts['invalid']}✗, {status_counts['to_be_verified']}?):")
            for claim in claims_to_show:
                parts.append(f"  {claim.to_prompt_str(detailed=False)}")

            if total_claims > max_claims:
                parts.append(f"  ... [{total_claims - max_claims} more claims not shown]")
        else:
            parts.append("Claims: None yet")

        # Questions (show open and recent ones)
        if self.questions:
            open_questions = self.get_open_questions()
            resolved_count = len(self.questions) - len(open_questions)

            parts.append(f"\nQuestions ({len(self.questions)} total: {len(open_questions)} open, {resolved_count} resolved):")

            # Show open questions
            for q in open_questions[:6]:  # Show up to 6 open questions
                parts.append(f"  {q.to_prompt_str(detailed=False)}")

            if len(open_questions) > 6:
                parts.append(f"  ... [{len(open_questions) - 6} more open questions]")
        else:
            parts.append("\nQuestions: None yet")

        # Notes (show recent ones)
        if self.notes:
            parts.append(f"\nNotes ({len(self.notes)} total):")
            notes_to_show = self.notes[-5:] if len(self.notes) > 5 else self.notes
            for note in notes_to_show:
                parts.append(f"  {note.to_prompt_str()}")
            if len(self.notes) > 5:
                parts.append(f"  ... [{len(self.notes) - 5} more notes not shown]")
        else:
            parts.append("\nNotes: None yet")

        # Outline
        outline_str = self.review_outline.to_prompt_str(brief=True)
        if outline_str != "No outline yet":
            parts.append(f"\nOutline Progress:\n{outline_str}")

        # Section visits
        if self.section_visits:
            visited = ", ".join(f"{s}({n}x)" for s, n in sorted(self.section_visits.items()))
            parts.append(f"\nSections Read: {visited}")

        # Search history
        if self.search_history:
            parts.append(f"\nSearches Done: {', '.join(repr(q) for q in self.search_history)}")

        return "\n".join(parts)

    def _build_detailed_context(self) -> str:
        """Build detailed context for review generation."""
        parts = [f"=== Complete Review Log ===\n"]

        # All claims with full details
        if self.claims:
            parts.append(f"All Claims ({len(self.claims)} total):")
            for claim in self.claims:
                parts.append(f"\n{claim.to_prompt_str(detailed=True)}")
        else:
            parts.append("Claims: None")

        # All questions with answers
        if self.questions:
            parts.append(f"\n\nAll Questions ({len(self.questions)} total):")
            for q in self.questions:
                parts.append(f"\n{q.to_prompt_str(detailed=True)}")
        else:
            parts.append("\n\nQuestions: None")

        # All notes
        if self.notes:
            parts.append(f"\n\nAll Notes ({len(self.notes)} total):")
            for note in self.notes:
                parts.append(f"\n{note.to_prompt_str()}")
        else:
            parts.append("\n\nNotes: None")

        # Full outline
        outline_str = self.review_outline.to_prompt_str(brief=False)
        if outline_str != "No outline yet":
            parts.append(f"\n\nReview Outline:\n{outline_str}")

        # Section coverage
        if self.section_visits:
            res = ""
            for section in self.section_visits:
                res += f"{section}({self.section_visits[section]}x), "
            parts.append(f"\n\nSections Read: {res.strip(', ')}")

        # Search history
        if self.search_history:
            parts.append(f"\n\nSearches Done: {', '.join(repr(q) for q in self.search_history)}")

        return "\n".join(parts)


# Backward compatibility alias
ReviewMemory = ReviewLog
