import math
import re
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
from typing import List, Optional, Dict, Any, Literal, Set
from enum import Enum
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# Define PackageProfile, Behavior, DetectionResult Pydantic models

class BehaviorCategory(str, Enum):
    """Canonical behavior categories matching consolidated pattern set, focused on observable actions in the npm ecosystem."""
    CREDENTIAL_THEFT = "credential_theft"  
    SENSITIVE_DATA_COLLECTION = "sensitive_data_collection"   
    CRYPTO_HIJACKING = "crypto_hijacking"              
    NETWORK_EXFILTRATION = "network_exfiltration"      
    CODE_EXECUTION = "code_execution"
    DEPENDENCY_INJECTION = "dependency_injection" 
    SUPPLY_CHAIN_PROPAGATION = "supply_chain_propagation"
    REPOSITORY_MANIPULATION = "repository_manipulation"              
    PRIVILEGE_ESCALATION = "privilege_escalation"
    PERSISTENCE = "persistence"
    SENSITIVE_FILE_ACCESS = "sensitive_file_access"     
    OBFUSCATION = "obfuscation"
    ANTI_DEBUGGING = "anti_debugging"                   # checks for debugger (debugger keyword, inspector)
    ANTI_ANALYSIS = "anti_analysis"                     # broader (checks for sandboxes, VMs, honeypots)
    RESOURCE_ABUSE = "resource_abuse"
    TYPOSQUATTING = "typosquatting"
    UNKNOWN = "unknown"                                 # Fallback for unclassified behaviors
    
    
class PackageProfile(BaseModel):
    """NPM package representation for security analysis pipeline."""
    package_id:  str = Field(default_factory=lambda: f"PKG{uuid.uuid4().hex[:8].upper()}")
    package_name: str
    version: str
    dependencies: Dict[str, str] = Field(default_factory=dict)
    dev_dependencies: Dict[str, str] = Field(default_factory=dict)
    peer_dependencies: Dict[str, str] = Field(default_factory=dict)
    scripts: Dict[str, str] = Field(default_factory=dict) 
    readme_content: Optional[str] = None
    entry_point_code: Optional[str] = None
    install_script_content: Optional[str] = None
    file_structure: List[str] = Field(default_factory=list)
    has_native_code: bool = False
    package_json_raw: Dict[str, Any] = Field(default_factory=dict)
    label: Optional[Literal["malicious", "benign"]] = None  

    
    @field_validator('package_name')
    def validate_package_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Package name cannot be empty")
        if not re.match(r'^@[a-z0-9-~][a-z0-9-._~]*/[a-z0-9-~][a-z0-9-._~]*$|^[a-z0-9-~][a-z0-9-._~]*$', v):
            raise ValueError(f"Invalid npm package name format: {v}")
        return v.strip()


class Behavior(BaseModel):
    """Individual malicious behavior extracted by LLM
    - Key: This represents ONE semantic unit of suspicious activity.
    - Multiple behaviors per package form a "risk vector".
    """
    behavior_id: str = Field(default_factory=lambda: f"BR{uuid.uuid4().hex[:8].upper()}")
    category: BehaviorCategory   
    package_name: str
    summary: str = Field(..., min_length=10, max_length=500)
    details: str = Field(..., min_length=20, max_length=2000)
    
    # Evidence anchoring (prevents hallucinations)
    evidence_apis: List[str] = Field(
        default_factory=list,
        description="API calls, functions, or methods (e.g., ['child_process.exec', 'fs.readFileSync'])"
    )
    evidence_files: List[str] = Field(
        default_factory=list,
        description="File paths accessed (e.g., ['.npmrc', '/etc/passwd'])"
    )
    evidence_domains: List[str] = Field(
        default_factory=list,
        description="External domains contacted (e.g., ['evil.com', '192.168.1.1'])"
    )
    evidence_commands: List[str] = Field(
        default_factory=list,
        description="Shell commands executed (curl | sh, wget, etc.)"
    )
    evidence_env_vars: List[str] = Field(
        default_factory=list,
        description="Environment variables accessed (e.g., ['NPM_TOKEN', 'AWS_SECRET_ACCESS_KEY', 'GITHUB_TOKEN'])"
    )
    file_path: Optional[str] = Field(
        None,
        description="Relative path within package (e.g., 'lib/index.js', 'scripts/postinstall.sh')"
    )
    source_code_snippet: Optional[str] = Field(
        None,
        max_length=1000,
        description="Minimal code excerpt showing the behavior"
    ) 
    line_numbers: Optional[List[int]] = None
    extraction_stage: Literal["structure", "semantic", "chain", "legitimacy"] = Field(
        default="semantic"
    )
    related_behavior_ids: List[str] = Field(
        default_factory=list,
        description="Behavior IDs that form an attack chain with this behavior"
    )
    context_legitimacy_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Legitimacy check score: 0=malicious, 1=contextually legitimate"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in behavior extraction (used for weighted scoring)"
    )
    severity: Literal["critical", "high", "medium", "low", "info"] = Field(
        default="medium",
        description="Risk severity for prioritization"
    )
    semantic_embedding: Optional[List[float]] = Field(
        None,
        description="384-dim vector from all-MiniLM-L6-v2",
        max_length=384
    )
    model_metadata: Dict[str, Any] = Field(        # Model provenance (debugging & reproducibility)
        default_factory=dict,
        description="Model version, prompt template, temperature, GPU ID, inference time, etc."
    )
    
    created_at: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        'use_enum_values': False
    }                      # Keep enum objects, not strings
        
    @field_validator('package_name')
    def validate_package_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Package name cannot be empty")
        if not re.match(r'^@[a-z0-9-~][a-z0-9-._~]*/[a-z0-9-~][a-z0-9-._~]*$|^[a-z0-9-~][a-z0-9-._~]*$', v):
            raise ValueError(f"Invalid npm package name format: {v}")
        return v.strip()
    
    @field_validator('evidence_domains')
    def validate_domains(cls, v):
        """Check domain are valid, catch LLM hallucinations."""
        if not v:
            return v
           
        domain_pattern = re.compile(r'^(?=.{1,253}$)(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.[A-Za-z]{2,})+$')
        
        validated = []
        for domain in v:
            clean = domain.replace('http://', '').replace('https://', '').split('/')[0]
            
            if domain_pattern.match(clean) or re.match(r'^\d{1,3}(\.\d{1,3}){3}$', clean):  # IP address
                validated.append(clean)
            else:
                logger.warning(f"Invalid domain format: {domain}")
        
        return validated
    
    @field_validator('evidence_apis')
    def validate_apis(cls, v):
        """Ensure APIs valid, catch LLM hallucinations."""
        if not v:
            return v
        
        validated = []
        for api in v:
            api = api.strip()
            
            if len(api) < 3:
                continue
            
            #Trade-off - Simple filter
            bad_words = ['malicious', 'evil', 'hack', 'steal', 'virus', 'trojan']
            if any(bad_word in api.lower() for bad_word in bad_words):
                logger.warning(f"Removing hallucinated API: {api}")
                continue
            
            validated.append(api)
        
        return validated
            
    
    @model_validator(mode='after')
    def validate_evidence_exists(self):
        """Enforce evidence anchoring - at least one evidence type required."""
        has_evidence = any([
            self.evidence_apis,
            self.evidence_files,
            self.evidence_domains,
            self.source_code_snippet
        ])
        
        if not has_evidence:
            raise ValueError(
                f"Behavior {self.behavior_id} has no evidence! "
                f"Category: {self.category.value}, Summary: {self.summary[:50]}... "
                "This likely indicates LLM hallucination or prompt engineering issue."
            )
        return self
    
    @model_validator(mode='after')
    def auto_assign_severity(self) -> 'Behavior':
        """Auto-assign severity based on category, confidence, and evidence.
        Uses a weighted approach considering:
        - Behavior category intrinsic risk
        - Model confidence
        - Evidence quality and quantity
        - Context legitimacy
        - Extraction stage reliability
        """
        base_severity = self._get_base_severity_from_category()                     # Base severity from category
        
        adjusted_severity = self._adjust_severity_with_confidence(base_severity)    # Adjust based on confidence and evidence
        
        final_severity = self._adjust_severity_with_context(adjusted_severity)      # Final adjustment based on context
        
        return self.model_copy(update={'severity': final_severity})
    
    def _get_base_severity_from_category(self) -> str:
        """Get base severity level from behavior category."""
        CRITICAL_CATEGORIES: Set[BehaviorCategory] = {
            BehaviorCategory.CREDENTIAL_THEFT,
            BehaviorCategory.CODE_EXECUTION, 
            BehaviorCategory.PRIVILEGE_ESCALATION,
            BehaviorCategory.SUPPLY_CHAIN_PROPAGATION,
            BehaviorCategory.REPOSITORY_MANIPULATION
        }
        
        HIGH_RISK_CATEGORIES: Set[BehaviorCategory] = {
            BehaviorCategory.NETWORK_EXFILTRATION,
            BehaviorCategory.SENSITIVE_DATA_COLLECTION,
            BehaviorCategory.CRYPTO_HIJACKING,
            BehaviorCategory.DEPENDENCY_INJECTION,
            BehaviorCategory.SENSITIVE_FILE_ACCESS
        }
        
        MEDIUM_RISK_CATEGORIES: Set[BehaviorCategory] = {
            BehaviorCategory.PERSISTENCE,
            BehaviorCategory.RESOURCE_ABUSE,
            BehaviorCategory.OBFUSCATION
        }
        
        if self.category in CRITICAL_CATEGORIES:
            return "critical"
        elif self.category in HIGH_RISK_CATEGORIES:
            return "high" 
        elif self.category in MEDIUM_RISK_CATEGORIES:
            return "medium"
        else:
            return "low"
    
    def _adjust_severity_with_confidence(self, base_severity: str) -> str:
        """Adjust severity based on model confidence and evidence quality."""
        # Confidence thresholds
        if self.confidence < 0.3:
            severity_map = {"critical": "high", "high": "medium", "medium": "low", "low": "info"}       # Low confidence - downgrade severity
            return severity_map.get(base_severity, "info")
        
        elif self.confidence < 0.6:
            if base_severity == "critical":                                                             # Medium - keep or slightly downgrade
                return "high"
            return base_severity
        
        elif self.confidence >= 0.8:
            if base_severity == "high" and self._has_strong_evidence():                                 # High - consider upgrading
                return "critical"
            elif base_severity == "medium" and self._has_strong_evidence():
                return "high"
        
        return base_severity
    
    def _adjust_severity_with_context(self, current_severity: str) -> str:
        """Adjust severity based on contextual factors."""
        if self.context_legitimacy_score and self.context_legitimacy_score > 0.7:                      # Override Context legitimacy if behavior is legitimate in context
            severity_map = {"critical": "low", "high": "info", "medium": "info", "low": "info"}
            return severity_map.get(current_severity, "info")
        
        if self.extraction_stage == "structure" and current_severity != "critical":
            return "high" 
        
        if self.category in [BehaviorCategory.ANTI_DEBUGGING, BehaviorCategory.ANTI_ANALYSIS]:         # Anti-analysis behaviors tend to be more severe
            if current_severity == "medium":
                return "high"
        
        return current_severity
    
    def _has_strong_evidence(self) -> bool:
        """Check if behavior has strong supporting evidence."""
        evidence_score = 0
        
        if len(self.evidence_apis) >= 2:
            evidence_score += 1
        if len(self.evidence_files) >= 1:
            evidence_score += 1  
        if len(self.evidence_domains) >= 1:
            evidence_score += 1
        if len(self.evidence_commands) >= 1:
            evidence_score += 1
        if self.source_code_snippet and len(self.source_code_snippet) > 50:
            evidence_score += 1
        
        return evidence_score >= 2
        

class DetectionResult(BaseModel):
    """Final classification output with confidence calibration and explainability."""
    package_name: str
    package_version: str
    package_id: str = Field(default_factory=lambda: f"PKG{uuid.uuid4().hex[:8].upper()}")
    classification: Literal["malicious", "benign", "suspicious", "unknown"]
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_scores: Dict[str, float] = Field(                                   # Confidence breakdown         
        default_factory=dict,
        description="""
        pattern_similarity: Semantic similarity to known patterns (0-1)
        chain_completeness: Behavior chain coherence (0-1) 
        context_violation: Legitimacy violation score (0-1)
        cross_perspective_consistency: Stage agreement (0-1)
        evidence_strength: Quality/quantity of supporting evidence (0-1)
        """
    )
    risk_vector: List[BehaviorCategory] = Field(default_factory=list)
    risk_level: Literal["critical", "high", "medium", "low", "info"] = Field(
        default="medium",
        description="Overall risk level considering confidence and impact"
    )
    explanation: str = Field(..., min_length=50, max_length=2000)
    risk_factors: List[str] = Field(
        default_factory=list,
        description="Specific risk indicators found during analysis"
    )
    mitigation_recommendations: List[str] = Field(
        default_factory=list,
        description="Suggested actions (block, review, monitor, etc.)"
    )
    threshold_used: float                                  # The θ (theta) threshold applied
    is_boundary_case: bool = Field(
        default=False, 
        description="Flag for human-in-the-loop review when near threshold"
    )
    triggered_verification_stages: List[str] = Field(
        default_factory=list,
        description="Which verification stages were activated (chain, context)"
    )
    detected_behaviors: List[str] = Field(
        default_factory=list,
        description="List of behavior IDs found in this package"
    )
    top_matching_patterns: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top N matching malicious patterns with similarity scores"
    )
    detection_timestamp: datetime = Field(default_factory=datetime.now)
    analysis_duration: float = Field(
        default=0.0,
        description="Total analysis time in seconds"
    )
    pipeline_version: str = Field(default="1.0")
    model_metadata: Dict[str, Any] = Field(           # Model provenance (debugging & reproducibility)
        default_factory=dict,
        description="Model version, prompt template, temperature, GPU ID, inference time, etc."
    )
    
    @field_validator('package_name')
    def validate_package_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Package name cannot be empty")
        if not re.match(r'^@[a-z0-9-~][a-z0-9-._~]*/[a-z0-9-~][a-z0-9-._~]*$|^[a-z0-9-~][a-z0-9-._~]*$', v):
            raise ValueError(f"Invalid package name format: {v}")
        return v.strip()
    
    @field_validator('confidence_scores')
    def validate_confidence_breakdown(cls, v):
        """Ensure all confidence factors are properly weighted."""
        required_factors = {
            'pattern_similarity', 
            'chain_completeness', 
            'context_violation', 
            'cross_perspective_consistency'
        }
        
        missing = required_factors - set(v.keys())
        if missing:
            raise ValueError(f"Missing required confidence factors: {missing}")
            
        for key, score in v.items():
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"Confidence factor {key} must be in [0, 1], got {score}")
        return v
    
    @field_validator('overall_confidence')
    def validate_overall_confidence(cls, v, values):
        """Ensure overall confidence aligns with individual factors."""
        if 'confidence_scores' in values.data:
            factors = values.data['confidence_scores']
            # Supposed weights for each factor: pattern_similarity(40%), chain_completeness(30%), context_violation(20%), cross_perspective_consistency(10%)
            weights = {
                'pattern_similarity': 0.4,
                'chain_completeness': 0.3, 
                'context_violation': 0.2,
                'cross_perspective_consistency': 0.1
            }
            expected = sum(factors.get(k, 0) * w for k, w in weights.items())
            
            if abs(v - expected) > 0.3:
                raise ValueError(f"Overall confidence {v} significantly differs from weighted factors {expected}")
        return v

    
    # --- Helper methods for analysis ---
    @computed_field
    @property
    def boundary_distance(self) -> float:
        """Distance from classification threshold."""
        return abs(self.overall_confidence - self.threshold_used)
    
    @computed_field
    @property 
    def is_high_confidence_malicious(self) -> bool:
        """Quick check for high-confidence malicious detections."""
        return (self.classification == "malicious" and 
                self.overall_confidence >= 0.8)
    
    @computed_field
    @property
    def behavior_count(self) -> int:
        """Number of detected behaviors."""
        return len(self.detected_behaviors)
    
    def get_primary_risk_factors(self, n: int = 3) -> List[str]:
        """Top-n most significant risk factors."""
        if not self.risk_factors:
            return []
        return self.risk_factors[:n]

    def should_escalate_for_review(self) -> bool:
        """If this result needs human review."""
        return (self.is_boundary_case or 
                self.classification == "suspicious" or
                self.overall_confidence > 0.7 and self.classification == "benign")
    
    def to_analyst_summary(self) -> Dict[str, Any]:
        """Generate summary for security analysts."""
        return {
            "package": f"{self.package_name}@{self.package_version}",
            "classification": self.classification,
            "confidence": round(self.overall_confidence, 3),
            "risk_level": self.risk_level,
            "primary_risks": self.get_primary_risk_factors(),
            "needs_review": self.should_escalate_for_review(),
            "analysis_time": f"{self.analysis_duration:.2f}s"
        }
    
    
# class PatternStatus(str, Enum):
#     """Lifecycle status of the consolidated malicious pattern."""
#     ACTIVE = "active"                    # Pattern is in detection use
#     DEPRECATED = "deprecated"            # Superseded by newer pattern
#     ARCHIVED = "archived"                # Historical, not used in current detection
#     UNDER_REVIEW = "under_review"        # High false positive rate, needs investigation
#     REVALIDATING = "revalidating"        # Being tested on new dataset

class PatternDetectionMetrics(BaseModel):
    """Performance metrics on validation set (training phase)."""
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0              
    false_negatives: int = 0             
    precision: Optional[float] = None    # TP / (TP + FP)
    recall: Optional[float] = None       # TP / (TP + FN)
    f1_score: Optional[float] = None     # Harmonic mean of precision & recall
    
    @model_validator(mode='after')
    def compute_metrics(self) -> 'PatternDetectionMetrics':
        """Auto-compute metrics from confusion matrix."""
        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)
        if self.true_positives + self.false_negatives > 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        if self.precision and self.recall and self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        return self



class ConsolidatedPattern(BaseModel):
    """Canonical malicious pattern from training phase (Pattern Recognition)"""
    pattern_id: str = Field(default_factory=lambda: f"PTN{uuid.uuid4().hex[:8].upper()}")
    category: BehaviorCategory
    canonical_behavior: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Human-readable description of the malicious pattern"
    )
    
    frequency_in_training: int = Field(
        ...,
        ge=3,
        description="Number of training packages exhibiting this pattern"
    )
    confidence_in_training: float = Field(
        ...,
        ge=0,
        le=1,
        description="Average confidence from source behaviors"
    )
    cluster_size: int = Field(
        ...,
        ge=1,
        description="Number of behaviors consolidated into this pattern"
    )
    semantic_embedding: List[float] = Field(
        ...,
        description="384-dim vector from all-MiniLM-L6-v2"
    )
    source_behavior_ids: List[str] = Field(
        default_factory=list,
        description="Original behavior IDs that formed this pattern"
    )
    # Performance tracking 
    match_count: int = Field(
        default=0,
        description="Times this pattern matched during detection"
    )
    true_positive_count: int = Field(
        default=0,
        description="Confirmed malicious matches"
    )
    false_positive_count: int = Field(
        default=0,
        description="False alarm matches"
    )
    created_at: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        'use_enum_values': False
    }
        
    @field_validator('semantic_embedding')
    def validate_embedding_dimensions(cls, v):
        """Ensure correct dimensionality for all-MiniLM-L6-v2."""
        if v and len(v) != 384:
            raise ValueError(f"Embedding must be 384-dim, got {len(v)}")
        return v
    
    def calculate_weighted_score(self, similarity: float) -> float:
        """Calculate weighted pattern match score.
        
        Combines:
        - Raw similarity (60%)
        - Training frequency (25%) — normalized using log1p for soft capping
        - Training confidence (15%)
        """
        frequency_weight = math.log1p(self.frequency_in_training) / math.log1p(10)
        frequency_weight = min(frequency_weight, 1.0)
        
        weighted_score = (
            0.60 * similarity +
            0.25 * frequency_weight +
            0.15 * self.confidence_in_training
        )
        
        return min(weighted_score, 1.0)
        
    def update_statistics(self, is_true_positive: bool):
        """Update pattern usage statistics."""
        self.match_count += 1
        if is_true_positive:
            self.true_positive_count += 1
        else:
            self.false_positive_count += 1
    
    def get_precision(self) -> Optional[float]:
        """Calculate pattern precision (TP / (TP + FP))."""
        total = self.true_positive_count + self.false_positive_count
        if total == 0:
            return None
        return self.true_positive_count / total
    
    
# Helper for portfolio analysis
def analyze_pattern_database(patterns: List[ConsolidatedPattern]) -> Dict[str, Any]:
    """Quick overview of pattern database quality."""
    precisions = [p.get_precision() for p in patterns]
    valid_precisions = [p for p in precisions if p is not None]
    
    return {
        "total_patterns": len(patterns),
        "patterns_with_data": len(valid_precisions),
        "avg_precision": sum(valid_precisions) / len(valid_precisions) if valid_precisions else None,
        "category_distribution": {
            cat.value: len([p for p in patterns if p.category == cat])
            for cat in BehaviorCategory
        }
    }