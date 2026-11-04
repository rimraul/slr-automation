"""Self-contained HTML reporting for reproduction evaluations."""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Union

from .adjudication_journal import MATCHER_VERSION
from .evaluator import NotEvaluable, PipelineEvaluation
from .gold_standard import GoldStandard

if TYPE_CHECKING:
    from .berg2023_gates import GateReport


def _metric(value: Union[float, NotEvaluable, None]) -> str:
    if value is None:
        return "&mdash;"
    if isinstance(value, NotEvaluable):
        return (
            "<span class='not-evaluable'>Not evaluable: "
            + html.escape(value.reason)
            + "</span>"
        )
    color = "#2e7d32" if value >= 0.8 else "#ef6c00" if value >= 0.5 else "#c62828"
    return f"<span class='metric' style='background:{color}'>{value:.1%}</span>"


def _cell(value: object) -> str:
    if isinstance(value, NotEvaluable):
        return _metric(value)
    return html.escape(str(value))


def _row(*cells: str, header: bool = False) -> str:
    tag = "th" if header else "td"
    return "<tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>"


def _section(title: str, body: str) -> str:
    return (
        "<section><h2>"
        + html.escape(title)
        + "</h2>"
        + body
        + "</section>"
    )


def _overview(evaluation: PipelineEvaluation, gold: GoldStandard) -> str:
    values = (
        ("Review", gold.review_citation or gold.review_id),
        ("Review type", gold.review_type),
        ("Reproduction mode", evaluation.reproduction_mode),
        ("Source mode", evaluation.source_mode),
        ("Gold included papers", str(evaluation.gold_n_included)),
        ("Search recall", _metric(evaluation.search_recall)),
        ("Final recall", _metric(evaluation.final_recall)),
        ("Final precision", _metric(evaluation.final_precision)),
        ("Partial reproduction", "yes" if evaluation.partial_reproduction else "no"),
        ("Missing databases", ", ".join(evaluation.missing_databases) or "none"),
        (
            "Additional databases",
            ", ".join(evaluation.unsupported_databases) or "none",
        ),
        ("Specification hash", f"<code>{evaluation.spec_hash}</code>"),
        ("Gold-standard hash", f"<code>{evaluation.gold_standard_hash}</code>"),
    )
    rows = "".join(
        _row(html.escape(label), value if value.startswith("<") else html.escape(value))
        for label, value in values
    )
    return f"<table>{rows}</table>"


def _stage_table(evaluation: PipelineEvaluation) -> str:
    rows = _row(
        "Stage",
        "Pipeline n",
        "Gold n",
        "Definite matches",
        "Recall / sensitivity",
        "Precision",
        "Specificity",
        "Pending adjudications",
        header=True,
    )
    for stage in evaluation.stages:
        rows += _row(
            html.escape(stage.stage),
            str(stage.pipeline_count),
            str(stage.gold_count),
            _cell(stage.found_in_gold),
            _metric(stage.recall),
            _metric(stage.precision),
            _metric(stage.specificity),
            str(len(stage.pending_adjudication_ids)),
        )
    return f"<table>{rows}</table>"


def _prisma_table(evaluation: PipelineEvaluation) -> str:
    rows = _row("Stage", "Original", "Reproduced", "Difference", header=True)
    for stage in evaluation.stages:
        for name, comparison in stage.prisma_count_comparison.items():
            if isinstance(comparison, NotEvaluable):
                rows += _row(html.escape(name), _metric(comparison), str(stage.pipeline_count), "&mdash;")
            else:
                rows += _row(
                    html.escape(name),
                    str(comparison.original),
                    str(comparison.reproduced),
                    str(comparison.difference),
                )
    return f"<table>{rows}</table>"


def _adjudications(evaluation: PipelineEvaluation) -> str:
    if not evaluation.pending_adjudications:
        return "<p>No pending match adjudications.</p>"
    items = "".join(
        "<li><code>"
        + html.escape(item.adjudication_id)
        + "</code> &mdash; "
        + html.escape(item.tier)
        + f" ({len(item.scores)} candidate pair(s))</li>"
        for item in evaluation.pending_adjudications
    )
    return f"<ul>{items}</ul>"


def _temporal_findings(evaluation: PipelineEvaluation) -> str:
    if not evaluation.newly_discovered_post_cutoff:
        return "<p>No definite post-cutoff additions were identified.</p>"
    items = "".join(
        "<li>"
        + html.escape(match.title or match.doi or match.pipeline_id)
        + (f" ({match.year})" if match.year is not None else "")
        + "</li>"
        for match in evaluation.newly_discovered_post_cutoff
    )
    return (
        "<p>These records are reported separately as post-cutoff discoveries. "
        "They are not counted as reproduced inclusions.</p><ul>"
        + items
        + "</ul>"
    )


def _gates(gate_report: "GateReport" | None) -> str:
    if gate_report is None:
        return "<p>No executable gate report was supplied.</p>"
    rows = _row("Gate", "Status", "Reason", header=True)
    for gate, status in gate_report.gate_status.items():
        rows += _row(
            html.escape(gate),
            html.escape(status),
            html.escape(gate_report.gate_reasons[gate]),
        )
    if isinstance(gate_report.loss_stage_distribution, NotEvaluable):
        distribution = _metric(gate_report.loss_stage_distribution)
    else:
        distribution = ", ".join(
            f"{html.escape(stage)}: {count}"
            for stage, count in gate_report.loss_stage_distribution.items()
        )
    bounds = (
        _metric(gate_report.final_recall_lower_bound)
        + " to "
        + _metric(gate_report.final_recall_upper_bound)
    )
    return (
        f"<table>{rows}</table>"
        f"<p><strong>First-loss distribution:</strong> {distribution}</p>"
        f"<p><strong>Final recall:</strong> {_metric(gate_report.final_recall)} "
        f"(basis: {html.escape(str(gate_report.final_recall_basis))})</p>"
        f"<p><strong>Conservative recall bounds:</strong> {bounds}</p>"
    )



def _missed(evaluation: PipelineEvaluation) -> str:
    parts: list[str] = []
    for stage in evaluation.stages:
        identities = [*stage.missed_dois, *stage.missed_titles]
        if not identities:
            continue
        items = "".join(f"<li>{html.escape(value)}</li>" for value in identities)
        parts.append(f"<h3>{html.escape(stage.stage)}</h3><ul>{items}</ul>")
    return "".join(parts) or "<p>No definite missed gold papers were identified.</p>"


def generate_html_report(
    evaluation: PipelineEvaluation,
    gold_standard: GoldStandard,
    output_path: Path,
    *,
    gate_report: "GateReport" | None = None,
) -> None:
    """Render already-classified evaluation results without reinterpreting them."""

    if evaluation.review_id != gold_standard.review_id:
        raise ValueError("evaluation review_id does not match gold standard")
    if evaluation.gold_standard_hash != gold_standard.content_hash():
        raise ValueError("evaluation gold_standard_hash does not match supplied content")
    if gate_report is not None:
        if gate_report.review_id != evaluation.review_id:
            raise ValueError("gate report review_id does not match evaluation")
        if gate_report.spec_hash != evaluation.spec_hash:
            raise ValueError("gate report spec_hash does not match evaluation")
        if gate_report.gold_standard_hash != evaluation.gold_standard_hash:
            raise ValueError("gate report gold_standard_hash does not match evaluation")
        evaluation_stage_hashes = {
            "raw_records": evaluation.raw_records_hash,
            "deduplicated_records": evaluation.deduplicated_records_hash,
            "screened_records": evaluation.screened_records_hash,
        }
        if evaluation.full_text_records_hash is not None:
            evaluation_stage_hashes["full_text_records"] = evaluation.full_text_records_hash
        if evaluation.included_records_hash is not None:
            evaluation_stage_hashes["included_records"] = evaluation.included_records_hash
        if evaluation_stage_hashes != gate_report.stage_corpus_hashes:
            raise ValueError("gate report stage-corpus hashes do not match evaluation")
        if gate_report.matcher_version != MATCHER_VERSION:
            raise ValueError(
                f"gate report matcher_version {gate_report.matcher_version!r} does not "
                f"match this reporter's own matcher_version {MATCHER_VERSION!r}"
            )

    notes = "".join(f"<li>{html.escape(note)}</li>" for note in evaluation.summary_notes)
    body = "".join(
        (
            "<h1>Pipeline Reproduction Report</h1>",
            f"<p>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>",
            _section("Overview and input binding", _overview(evaluation, gold_standard)),
            _section("Stage-by-stage metrics", _stage_table(evaluation)),
            _section("PRISMA count comparison", _prisma_table(evaluation)),
            _section("Pending human adjudications", _adjudications(evaluation)),
            _section("Executable acceptance gates", _gates(gate_report)),
            _section("Post-cutoff discoveries", _temporal_findings(evaluation)),
            _section("Definite losses", _missed(evaluation)),
            _section("Notes", f"<ul>{notes}</ul>" if notes else "<p>None.</p>"),
        )
    )
    css = """
body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;line-height:1.5}
table{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem}th,td{border:1px solid #ddd;padding:.55rem;text-align:left;vertical-align:top}th{background:#f3f5f7}.metric{color:white;border-radius:1rem;padding:.15rem .55rem;white-space:nowrap}.not-evaluable{color:#7a3e00;background:#fff3e0;padding:.15rem .4rem}code{font-size:.85em;word-break:break-all}section{margin:1.6rem 0}
"""
    document = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width'><title>Reproduction report - {html.escape(evaluation.review_id)}</title>"
        f"<style>{css}</style></head><body>{body}</body></html>"
    )
    output_path.write_text(document, encoding="utf-8")


__all__ = ["generate_html_report"]
