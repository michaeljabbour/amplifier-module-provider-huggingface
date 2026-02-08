"""Structural validation tests for huggingface provider.

Inherits authoritative tests from amplifier-core.
"""

from amplifier_core.validation.structural import ProviderStructuralTests


class TestHuggingFaceProviderStructural(ProviderStructuralTests):
    """Run standard provider structural tests for huggingface.

    All tests from ProviderStructuralTests run automatically.
    Add module-specific structural tests below if needed.
    """
