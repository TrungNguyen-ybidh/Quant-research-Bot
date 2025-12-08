"""
Run Signal Validation
=====================
Comprehensive validation of the 24-hour signal.

Usage:
    python scripts/validate_signal.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.signal_validator import SignalValidator, ValidationConfig


def main():
    # Configure for 24-hour lookahead
    config = ValidationConfig(
        lookahead=24,
        n_splits=5,
        pairs=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
    )
    
    # Run validation
    validator = SignalValidator(config)
    results = validator.run_full_validation("EURUSD")
    
    # Save results
    validator.save_results(results)
    
    print("\n" + "=" * 80)
    print("NEXT STEPS BASED ON RESULTS")
    print("=" * 80)
    
    verdict = results.get("verdict", {})
    
    if verdict.get("confidence") == "HIGH":
        print("""
    ✓ Signal is validated! Recommended next steps:
    
    1. Paper trade for 1-3 months
    2. Start with small position sizes
    3. Monitor for regime changes
    4. Build automated execution system
        """)
    elif verdict.get("confidence") == "MEDIUM":
        print("""
    ⚠ Signal shows promise but has weaknesses:
    
    1. Investigate failing validations
    2. Consider regime filtering
    3. Test with longer out-of-sample period
    4. Combine with other confirming signals
        """)
    else:
        print("""
    ✗ Signal does not validate reliably:
    
    1. Reconsider the hypothesis
    2. Try different holding periods
    3. Focus on feature engineering
    4. Consider the signal may be spurious
        """)
    
    return results


if __name__ == "__main__":
    main()