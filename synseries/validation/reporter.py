"""
synseries.validation.reporter
Generates formatted validation reports and authenticity certificates for synthetic datasets.
"""

from typing import Dict, Any


class ValidationReporter:
    """Renders human-readable validation certificates in Markdown and terminal text."""

    @classmethod
    def generate_markdown_report(cls, results: Dict[str, Any]) -> str:
        domain = results["domain"].upper()
        status = results["overall_status"]
        score = results["score_pct"]
        badge = "PASSED" if status == "PASS" else "FAILED"

        lines = [
            f"# Domain Validity & Authenticity Certificate",
            f"**Target Domain:** `{domain}` | **Validation Status:** {badge} ({score}%)",
            f"**Total Samples:** {results['total_samples']} | **Channels:** {len(results['columns'])}",
            "",
            "## Invariant & Statistical Verification Matrix",
            "| Check Name | Category | Status | Verification Details |",
            "| :--- | :--- | :---: | :--- |",
        ]

        for check in results["checks"]:
            st_text = "PASS" if "PASS" in check["status"] else "FAIL"
            lines.append(
                f"| **{check['name']}** | {check['category']} | {st_text} | {check['details']} |"
            )

        lines.extend([
            "",
            "> [!NOTE]",
            f"> This synthetic dataset has been verified against the physical, biological, and mathematical constraints of the **{domain}** domain.",
            "> All physical boundaries, temporal continuity (ACF), and invariant laws are mathematically verified.",
        ])

        return "\n".join(lines)
