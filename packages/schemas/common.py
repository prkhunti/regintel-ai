from enum import StrEnum


class DocumentType(StrEnum):
    CLINICAL_EVALUATION_REPORT = "clinical_evaluation_report"
    RISK_MANAGEMENT_FILE = "risk_management_file"
    SOFTWARE_REQUIREMENTS = "software_requirements"
    IFU = "ifu"
    CYBERSECURITY = "cybersecurity"
    STANDARD_GUIDANCE = "standard_guidance"
    OTHER = "other"


class QueryType(StrEnum):
    EVIDENCE_EXTRACTION = "evidence_extraction"
    SUMMARY = "summary"
    GAP_ANALYSIS = "gap_analysis"
    CONTRADICTION_CHECK = "contradiction_check"
    TRACEABILITY_MAPPING = "traceability_mapping"
    COMPLIANCE_MAPPING = "compliance_mapping"


class ProcessingStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
