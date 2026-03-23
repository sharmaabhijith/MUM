from enum import Enum


class RelationshipType(str, Enum):
    SYMMETRIC = "symmetric"
    HIERARCHICAL = "hierarchical"
    CROSS_FUNCTIONAL = "cross_functional"
    ADVERSARIAL = "adversarial"
    SEQUENTIAL = "sequential"


class AuthorityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EQUAL = "equal"


class ConflictType(str, Enum):
    FACTUAL = "factual"
    INTERPRETIVE = "interpretive"
    STRATEGIC = "strategic"
    AUTHORITY = "authority"


class ConflictResolution(str, Enum):
    PRESERVE_BOTH = "preserve_both"
    SUPERSEDE = "supersede"
    AUTHORITY_WINS = "authority_wins"
    TEMPORAL_SUPERSESSION = "temporal_supersession"


class MemoryType(str, Enum):
    FACT = "fact"
    OPINION = "opinion"
    STRATEGY = "strategy"
    DIRECTIVE = "directive"
    RISK_FLAG = "risk_flag"
    ASSESSMENT = "assessment"
    POSITION = "position"
    DIAGNOSIS = "diagnosis"
    CUSTOMER_COMM = "customer_communication"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    CORRECTED = "corrected"


class EvalQuestionCategory(str, Enum):
    USER_ATTRIBUTION = "user_attribution"
    CROSS_USER_SYNTHESIS = "cross_user_synthesis"
    CONFLICT_RESOLUTION = "conflict_resolution"
    INFORMATION_GAP = "information_gap"
    ROLE_APPROPRIATE_BRIEFING = "role_appropriate_briefing"
    ADVERSARIAL_CONFUSION = "adversarial_confusion"
    DOCUMENT_COVERAGE = "document_coverage"
    CROSS_USER_PROVENANCE = "cross_user_provenance"
    AUTHORITY_HIERARCHY = "authority_hierarchy"
    TEMPORAL_CORRECTION = "temporal_correction"
    ADVERSARIAL_SIDE_ISOLATION = "adversarial_side_isolation"
    SEQUENTIAL_HANDOFF = "sequential_handoff"
