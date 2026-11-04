"""The Track 2 reference run's synthetic fixture corpus -- ONE structured
answer key (`track2_step12_revision_proposal_2026-09-02.md`, Step 12A).

This module is the **sole authoritative source** for every fixture-corpus
fact the reference run's other artifacts must agree with: the 30 raw
records' bibliographic content, which 8 are title-duplicates of which
canonical record, the title/abstract screening decision for each of the 22
post-dedup canonical records, the full-text eligibility determination for
each of the 6 records whose full text is sought, and the 9-field extraction
values for each of the 5 full-text-eligible records. `CORPUS_ORACLE.json`,
the scenario table, the native `fx_<db>.json` files, and (later) the
replay-cassette responses are all **generated from, or acceptance-checked
against, this one module** -- never independently authored a second or
third time (the revision proposal's explicit anti-drift requirement).

All content is invented for pipeline-conformance testing. No record here
describes a real study; see `PROVENANCE.md` for the corpus-wide synthetic-
data warning this module's own `dump_native_fixture_files` also stamps
into `known_real_doi_denylist.txt`.

The corpus topology is fixed by `CORPUS_ORACLE.json` (frozen design
literals, not derived here): 30 raw records across the 7 canonical Berg2023
databases, 8 exact-title duplicates, 22 canonical survivors, 6 title/
abstract inclusions, 6 full-text-sought records, 1 full-text exclusion, 5
full-text-eligible/extracted records. This module's own `verify_topology()`
cross-checks its authored data against those literals so the two can never
silently drift apart.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ANSWER_KEY_VERSION = "1"

# The 7 canonical Berg2023 databases, in `ReferenceRunSpec.databases` order
# (`slr_pipeline/reference_spec.py::_CANONICAL_DATABASES`) -- this is also
# the `raw_records` ordering `nodes/search.py` produces (configured
# database order, then native-file list order within each database), which
# is what `_dedup_doi_only`'s first-seen-wins semantics depend on.
CANONICAL_DATABASES: Tuple[str, ...] = (
    "pubmed", "web_of_science", "cinahl", "scopus", "acm",
    "engineering_village", "business_source_complete",
)

# ---------------------------------------------------------------------------
# The 30 raw records, authored in `raw_records` order: grouped by database
# in `CANONICAL_DATABASES` order, and within each database in the exact
# list order its `native/fx_<db>.json` file will hold. Every duplicate's
# canonical record is authored strictly earlier in this same order (the
# `_dedup_doi_only` first-seen-by-normalised-title-wins precondition) --
# `verify_topology()` checks this explicitly, not just by construction.
#
# Each entry: id, source_db, native_key, title, abstract, authors, year,
# journal, cited_by, url, duplicate_of (canonical fx-id, or None),
# duplicate_basis ("title", or None for canonical records).
# ---------------------------------------------------------------------------

_RAW_RECORDS: Tuple[Dict[str, Any], ...] = (
    # --- pubmed (6) ---
    {
        "id": "fx-0009", "source_db": "pubmed", "native_key": "NK-fx-0009",
        "title": "A model-based systems engineering framework for rural emergency medical response",
        "abstract": (
            "We present an MBSE framework, implemented in SysML, that models the "
            "emergency medical response system for a network of rural clinics "
            "coordinating during a regional flood disaster."
        ),
        "authors": ["A. Okafor", "B. Lindqvist"], "year": 2021,
        "journal": "Journal of Rural Health Systems Engineering", "cited_by": 4,
        "url": "https://example-fixture.invalid/fx-0009", "duplicate_of": None, "duplicate_basis": None,
    },
    {
        "id": "fx-0001", "source_db": "pubmed", "native_key": "NK-fx-0001",
        "title": "Machine learning for predictive maintenance in industrial manufacturing plants",
        "abstract": (
            "This study applies gradient-boosted machine learning models to sensor "
            "data from manufacturing equipment to predict mechanical failures before "
            "they occur, reducing unplanned downtime in urban factories."
        ),
        "authors": ["C. Reyes"], "year": 2022,
        "journal": "International Journal of Industrial Analytics", "cited_by": 11,
        "url": "https://example-fixture.invalid/fx-0001", "duplicate_of": None, "duplicate_basis": None,
    },
    {
        "id": "fx-0003", "source_db": "pubmed", "native_key": "NK-fx-0003",
        "title": "Urban emergency department overcrowding during seasonal influenza outbreaks",
        "abstract": (
            "A retrospective analysis of emergency department overcrowding metrics "
            "at metropolitan hospitals during three influenza seasons, examining "
            "triage delays and resource allocation in a large urban health system."
        ),
        "authors": ["D. Marchetti", "E. Huang"], "year": 2019,
        "journal": "Urban Health Services Quarterly", "cited_by": 7,
        "url": "https://example-fixture.invalid/fx-0003", "duplicate_of": None, "duplicate_basis": None,
    },
    {
        "id": "fx-0004", "source_db": "pubmed", "native_key": "NK-fx-0004",
        "title": "Qualitative interviews with disaster relief coordinators in coastal megacities",
        "abstract": (
            "Semi-structured interviews with 24 disaster relief coordinators in "
            "three coastal megacities explore organisational barriers to inter-"
            "agency coordination during flood response, without any computational "
            "modelling component."
        ),
        "authors": ["F. Adeyemi"], "year": 2020,
        "journal": "Disaster Coordination Studies", "cited_by": 3,
        "url": "https://example-fixture.invalid/fx-0004", "duplicate_of": None, "duplicate_basis": None,
    },
    {
        "id": "fx-0006", "source_db": "pubmed", "native_key": "NK-fx-0006",
        "title": "Cost-effectiveness of telemedicine triage in remote mining communities",
        "abstract": (
            "An economic evaluation compares telemedicine-based triage against "
            "in-person triage for non-emergency complaints in remote mining "
            "communities, focusing on cost per consultation rather than disaster "
            "response."
        ),
        "authors": ["G. Novak", "H. Petrov"], "year": 2018,
        "journal": "Journal of Remote Health Economics", "cited_by": 9,
        "url": "https://example-fixture.invalid/fx-0006", "duplicate_of": None, "duplicate_basis": None,
    },
    {
        "id": "fx-0007", "source_db": "pubmed", "native_key": "NK-fx-0007",
        "title": "urban emergency department overcrowding during  seasonal influenza  outbreaks",
        "abstract": "Duplicate listing of fx-0003 as re-indexed by the source database; content identical.",
        "authors": ["D. Marchetti", "E. Huang"], "year": 2019,
        "journal": "Urban Health Services Quarterly", "cited_by": 7,
        "url": "https://example-fixture.invalid/fx-0007", "duplicate_of": "fx-0003", "duplicate_basis": "title",
    },
    # --- web_of_science (5) ---
    {
        "id": "fx-0002", "source_db": "web_of_science", "native_key": "NK-fx-0002",
        "title": "System dynamics simulation of surge capacity in rural disaster response",
        "abstract": (
            "We build a system-dynamics model of hospital surge capacity in three "
            "rural US counties during simulated hurricane evacuation scenarios, "
            "testing staffing and bed-allocation policies under demand shocks."
        ),
        "authors": ["I. Sorensen", "J. Whitfield"], "year": 2022,
        "journal": "Simulation in Rural Health Planning", "cited_by": 6,
        "url": "https://example-fixture.invalid/fx-0002", "duplicate_of": None, "duplicate_basis": None,
    },
    {
        "id": "fx-0008", "source_db": "web_of_science", "native_key": "NK-fx-0008",
        "title": "Blockchain-based supply chain traceability for disaster relief donations",
        "abstract": (
            "This paper proposes a blockchain ledger architecture to trace in-kind "
            "donations through disaster relief supply chains, with a proof-of-"
            "concept deployment following an urban earthquake response."
        ),
        "authors": ["K. Yilmaz"], "year": 2021,
        "journal": "Journal of Humanitarian Logistics Technology", "cited_by": 5,
        "url": "https://example-fixture.invalid/fx-0008", "duplicate_of": None, "duplicate_basis": None,
    },
    {
        "id": "fx-0010", "source_db": "web_of_science", "native_key": "NK-fx-0010",
        "title": "Wearable sensor networks for continuous glucose monitoring in diabetic patients",
        "abstract": (
            "A clinical evaluation of a wearable continuous glucose monitoring "
            "sensor network in an outpatient diabetes clinic, with no relation to "
            "disaster planning or rural health systems."
        ),
        "authors": ["L. Fontaine", "M. Osei"], "year": 2020,
        "journal": "Wearable Clinical Sensors", "cited_by": 14,
        "url": "https://example-fixture.invalid/fx-0010", "duplicate_of": None, "duplicate_basis": None,
    },
    {
        "id": "fx-0011", "source_db": "web_of_science", "native_key": "NK-fx-0011",
        "title": "QUALITATIVE INTERVIEWS WITH DISASTER RELIEF COORDINATORS IN COASTAL MEGACITIES",
        "abstract": "Duplicate listing of fx-0004, indexed by this database under all-caps formatting.",
        "authors": ["F. Adeyemi"], "year": 2020,
        "journal": "Disaster Coordination Studies", "cited_by": 3,
        "url": "https://example-fixture.invalid/fx-0011", "duplicate_of": "fx-0004", "duplicate_basis": "title",
    },
    {
        "id": "fx-0012", "source_db": "web_of_science", "native_key": "NK-fx-0012",
        "title": "A discrete-event simulation of airport security queuing during holiday travel peaks",
        "abstract": (
            "This study uses discrete-event simulation to optimise airport "
            "security-checkpoint staffing during holiday travel peaks, an "
            "application of simulation methods entirely outside the healthcare or "
            "disaster domain."
        ),
        "authors": ["N. Kowalski"], "year": 2019,
        "journal": "Transportation Systems Simulation", "cited_by": 8,
        "url": "https://example-fixture.invalid/fx-0012", "duplicate_of": None, "duplicate_basis": None,
    },
    # --- cinahl (4) ---
    {
        "id": "fx-0013", "source_db": "cinahl", "native_key": "NK-fx-0013",
        "title": "Discrete-event simulation of a rural community's mass-casualty triage pathway",
        "abstract": (
            "A discrete-event simulation of the mass-casualty triage pathway at a "
            "single rural community hospital is used to test staffing "
            "configurations against a simulated multi-vehicle accident and a "
            "simulated tornado strike."
        ),
        "authors": ["O. Dubois", "P. Nakamura"], "year": 2023,
        "journal": "Rural Emergency Care Simulation", "cited_by": 2,
        "url": "https://example-fixture.invalid/fx-0013", "duplicate_of": None, "duplicate_basis": None,
    },
    {
        "id": "fx-0014", "source_db": "cinahl", "native_key": "NK-fx-0014",
        "title": "Nutrition education programmes for elderly residents of rural care homes",
        "abstract": (
            "A pre-post evaluation of a nutrition education programme delivered to "
            "elderly residents across six rural care homes, unrelated to simulation "
            "methods or disaster response."
        ),
        "authors": ["Q. Ibrahim"], "year": 2018,
        "journal": "Rural Gerontology Nutrition", "cited_by": 6,
        "url": "https://example-fixture.invalid/fx-0014", "duplicate_of": None, "duplicate_basis": None,
    },
    {
        "id": "fx-0015", "source_db": "cinahl", "native_key": "NK-fx-0015",
        "title": "Nurse staffing ratios and patient outcomes in rural critical access hospitals",
        "abstract": (
            "An observational study of nurse staffing ratios and patient outcomes "
            "across a network of rural critical-access hospitals, with no "
            "simulation or disaster-response component."
        ),
        "authors": ["R. Svensson", "S. Delgado"], "year": 2021,
        "journal": "Rural Nursing Outcomes", "cited_by": 10,
        "url": "https://example-fixture.invalid/fx-0015", "duplicate_of": None, "duplicate_basis": None,
    },
    {
        "id": "fx-0016", "source_db": "cinahl", "native_key": "NK-fx-0016",
        "title": "a model-based systems engineering  framework for rural emergency medical response",
        "abstract": "Duplicate listing of fx-0009, indexed by this database in lower case with a double space.",
        "authors": ["A. Okafor", "B. Lindqvist"], "year": 2021,
        "journal": "Journal of Rural Health Systems Engineering", "cited_by": 4,
        "url": "https://example-fixture.invalid/fx-0016", "duplicate_of": "fx-0009", "duplicate_basis": "title",
    },
    # --- scopus (5) ---
    {
        "id": "fx-0005", "source_db": "scopus", "native_key": "NK-fx-0005",
        "title": "Agent-based modelling of evacuation logistics for rural hospitals during hurricanes",
        "abstract": (
            "An agent-based simulation models patient and staff evacuation "
            "logistics for a network of rural hospitals during hurricane landfall, "
            "comparing three staged-evacuation protocols under varying road-closure "
            "scenarios."
        ),
        "authors": ["T. Andersen", "U. Chowdhury"], "year": 2022,
        "journal": "Agent-Based Disaster Logistics", "cited_by": 5,
        "url": "https://example-fixture.invalid/fx-0005", "duplicate_of": None, "duplicate_basis": None,
    },
    {
        "id": "fx-0017", "source_db": "scopus", "native_key": "NK-fx-0017",
        "title": "Public perception of autonomous vehicles in suburban transportation planning",
        "abstract": (
            "A survey of public perception of autonomous vehicle adoption in "
            "suburban transportation planning committees, with no connection to "
            "disaster response, rural healthcare, or simulation-based methods."
        ),
        "authors": ["V. Moreau"], "year": 2020,
        "journal": "Transportation Planning Perceptions", "cited_by": 4,
        "url": "https://example-fixture.invalid/fx-0017", "duplicate_of": None, "duplicate_basis": None,
    },
    {
        "id": "fx-0019", "source_db": "scopus", "native_key": "NK-fx-0019",
        "title": "Cybersecurity vulnerabilities in hospital SCADA systems",
        "abstract": (
            "A technical audit of cybersecurity vulnerabilities in SCADA-"
            "controlled building systems at a sample of metropolitan hospitals, "
            "unrelated to disaster planning, rural health, or simulation methods."
        ),
        "authors": ["W. Bakker", "X. Papadopoulos"], "year": 2021,
        "journal": "Healthcare Cybersecurity Review", "cited_by": 12,
        "url": "https://example-fixture.invalid/fx-0019", "duplicate_of": None, "duplicate_basis": None,
    },
    {
        "id": "fx-0020", "source_db": "scopus", "native_key": "NK-fx-0020",
        "title": "Comparative analysis of ambulance response times across three national health systems",
        "abstract": (
            "A comparative statistical analysis of ambulance response times "
            "across three national health systems, using historical dispatch "
            "records rather than any simulation or systems-engineering method."
        ),
        "authors": ["Y. Kristiansen"], "year": 2019,
        "journal": "Ambulance Systems Comparative Studies", "cited_by": 7,
        "url": "https://example-fixture.invalid/fx-0020", "duplicate_of": None, "duplicate_basis": None,
    },
    {
        "id": "fx-0021", "source_db": "scopus", "native_key": "NK-fx-0021",
        "title": "Nutrition   Education Programmes For Elderly Residents Of Rural Care Homes",
        "abstract": "Duplicate listing of fx-0014, indexed by this database in title case with triple spacing.",
        "authors": ["Q. Ibrahim"], "year": 2018,
        "journal": "Rural Gerontology Nutrition", "cited_by": 6,
        "url": "https://example-fixture.invalid/fx-0021", "duplicate_of": "fx-0014", "duplicate_basis": "title",
    },
    # --- acm (4) ---
    {
        "id": "fx-0022", "source_db": "acm", "native_key": "NK-fx-0022",
        "title": "A serious game for teaching Boolean logic to undergraduate computer science students",
        "abstract": (
            "This paper describes the design and classroom evaluation of a "
            "serious game intended to teach Boolean logic concepts to "
            "undergraduate computer science students, unconnected to healthcare "
            "or disaster response."
        ),
        "authors": ["Z. Herrera"], "year": 2022,
        "journal": "Proceedings of Serious Games in CS Education", "cited_by": 3,
        "url": "https://example-fixture.invalid/fx-0022", "duplicate_of": None, "duplicate_basis": None,
    },
    {
        "id": "fx-0023", "source_db": "acm", "native_key": "NK-fx-0023",
        "title": "BLOCKCHAIN-BASED SUPPLY CHAIN TRACEABILITY FOR DISASTER RELIEF DONATIONS",
        "abstract": "Duplicate listing of fx-0008, indexed by this database in all-caps formatting.",
        "authors": ["K. Yilmaz"], "year": 2021,
        "journal": "Journal of Humanitarian Logistics Technology", "cited_by": 5,
        "url": "https://example-fixture.invalid/fx-0023", "duplicate_of": "fx-0008", "duplicate_basis": "title",
    },
    {
        "id": "fx-0024", "source_db": "acm", "native_key": "NK-fx-0024",
        "title": "A viewpoint on model-based systems engineering for disaster-resilient rural health systems",
        "abstract": (
            "This viewpoint article argues for wider adoption of model-based "
            "systems engineering to design disaster-resilient rural health "
            "systems, drawing on the authors' consulting experience rather than a "
            "specific empirical case or named simulation method."
        ),
        "authors": ["AA. Larsson"], "year": 2023,
        "journal": "Viewpoints in Health Systems Engineering", "cited_by": 1,
        "url": "https://example-fixture.invalid/fx-0024", "duplicate_of": None, "duplicate_basis": None,
    },
    {
        "id": "fx-0025", "source_db": "acm", "native_key": "NK-fx-0025",
        "title": "Procedurally generated terrain for open-world video games",
        "abstract": (
            "A technical paper on procedural terrain-generation algorithms for "
            "open-world video game engines, with no connection whatsoever to "
            "healthcare, disaster response, or systems engineering."
        ),
        "authors": ["AB. Costa"], "year": 2020,
        "journal": "Proceedings of Procedural Content Generation", "cited_by": 9,
        "url": "https://example-fixture.invalid/fx-0025", "duplicate_of": None, "duplicate_basis": None,
    },
    # --- engineering_village (3) ---
    {
        "id": "fx-0018", "source_db": "engineering_village", "native_key": "NK-fx-0018",
        "title": "Computer simulation of ambulance dispatch during epidemic outbreaks in remote regions",
        "abstract": (
            "We simulate ambulance dispatch and routing policies during a "
            "simulated epidemic outbreak affecting a network of remote-region "
            "hospitals, comparing static and dynamic dispatch heuristics under a "
            "discrete-event model."
        ),
        "authors": ["AC. Nilsen", "AD. Bello"], "year": 2022,
        "journal": "Computer Simulation in Emergency Logistics", "cited_by": 6,
        "url": "https://example-fixture.invalid/fx-0018", "duplicate_of": None, "duplicate_basis": None,
    },
    {
        "id": "fx-0026", "source_db": "engineering_village", "native_key": "NK-fx-0026",
        "title": "cybersecurity vulnerabilities in hospital scada systems",
        "abstract": "Duplicate listing of fx-0019, indexed by this database in lower case.",
        "authors": ["W. Bakker", "X. Papadopoulos"], "year": 2021,
        "journal": "Healthcare Cybersecurity Review", "cited_by": 12,
        "url": "https://example-fixture.invalid/fx-0026", "duplicate_of": "fx-0019", "duplicate_basis": "title",
    },
    {
        "id": "fx-0027", "source_db": "engineering_village", "native_key": "NK-fx-0027",
        "title": "Finite element analysis of seismic retrofitting for reinforced-concrete bridges",
        "abstract": (
            "A finite-element structural analysis of seismic retrofitting "
            "techniques for reinforced-concrete highway bridges, an engineering-"
            "simulation application unrelated to healthcare systems or emergency "
            "medical response."
        ),
        "authors": ["AE. Tanaka"], "year": 2019,
        "journal": "Structural Simulation Engineering", "cited_by": 15,
        "url": "https://example-fixture.invalid/fx-0027", "duplicate_of": None, "duplicate_basis": None,
    },
    # --- business_source_complete (3) ---
    {
        "id": "fx-0028", "source_db": "business_source_complete", "native_key": "NK-fx-0028",
        "title": "A Discrete-Event Simulation Of Airport Security Queuing During Holiday Travel Peaks",
        "abstract": "Duplicate listing of fx-0012, indexed by this database in title case.",
        "authors": ["N. Kowalski"], "year": 2019,
        "journal": "Transportation Systems Simulation", "cited_by": 8,
        "url": "https://example-fixture.invalid/fx-0028", "duplicate_of": "fx-0012", "duplicate_basis": "title",
    },
    {
        "id": "fx-0029", "source_db": "business_source_complete", "native_key": "NK-fx-0029",
        "title": "Supply chain resilience strategies for multinational retail corporations",
        "abstract": (
            "A business-strategy analysis of supply-chain resilience practices "
            "adopted by multinational retail corporations following global "
            "shipping disruptions, unrelated to rural healthcare or disaster "
            "response planning."
        ),
        "authors": ["AF. Lindberg"], "year": 2020,
        "journal": "Retail Supply Chain Strategy", "cited_by": 5,
        "url": "https://example-fixture.invalid/fx-0029", "duplicate_of": None, "duplicate_basis": None,
    },
    {
        "id": "fx-0030", "source_db": "business_source_complete", "native_key": "NK-fx-0030",
        "title": "cost-effectiveness of telemedicine triage in remote  mining communities",
        "abstract": "Duplicate listing of fx-0006, indexed by this database in lower case with a double space.",
        "authors": ["G. Novak", "H. Petrov"], "year": 2018,
        "journal": "Journal of Remote Health Economics", "cited_by": 9,
        "url": "https://example-fixture.invalid/fx-0030", "duplicate_of": "fx-0006", "duplicate_basis": "title",
    },
)

_BY_ID: Dict[str, Dict[str, Any]] = {r["id"]: r for r in _RAW_RECORDS}


def all_ids() -> List[str]:
    return [r["id"] for r in _RAW_RECORDS]


def canonical_ids() -> List[str]:
    return [r["id"] for r in _RAW_RECORDS if r["duplicate_of"] is None]


def duplicate_ids() -> List[str]:
    return [r["id"] for r in _RAW_RECORDS if r["duplicate_of"] is not None]


def record(record_id: str) -> Dict[str, Any]:
    return _BY_ID[record_id]


def records_by_db() -> Dict[str, List[Dict[str, Any]]]:
    """Groups `_RAW_RECORDS` by `source_db`, preserving authored order --
    exactly the per-database list each `native/fx_<db>.json` file holds."""
    grouped: Dict[str, List[Dict[str, Any]]] = {db: [] for db in CANONICAL_DATABASES}
    for r in _RAW_RECORDS:
        grouped[r["source_db"]].append(r)
    return grouped


def native_payload(db: str) -> List[Dict[str, Any]]:
    """The exact JSON-object shape `fixture_search_adapter.py::
    _record_from_native` expects, for one database's records, in file
    order. `doi` is always `None` -- every fixture record is DOI-less by
    design (`CORPUS_ORACLE.json`'s `all_fixture_records_doi_less`), so all
    8 duplicates are caught by the dedup title stage, never the DOI
    stage."""
    return [
        {
            "id": r["id"],
            "native_key": r["native_key"],
            "doi": None,
            "title": r["title"],
            "abstract": r["abstract"],
            "authors": r["authors"],
            "year": r["year"],
            "journal": r["journal"],
            "cited_by": r["cited_by"],
            "url": r["url"],
        }
        for r in records_by_db()[db]
    ]


def duplicate_mappings() -> List[Dict[str, str]]:
    """`CORPUS_ORACLE.json`'s `duplicate_mappings` shape, derived here
    rather than independently retyped."""
    return [
        {"dup_id": r["id"], "canonical_id": r["duplicate_of"], "basis": r["duplicate_basis"]}
        for r in _RAW_RECORDS
        if r["duplicate_of"] is not None
    ]


# ---------------------------------------------------------------------------
# Title/abstract screening decisions -- one entry per canonical (non-
# duplicate) record, `ScreeningResponse`-shaped
# (`nodes/screening.py::ScreeningResponse`: decision/reason/
# relevant_topics/confidence).
# ---------------------------------------------------------------------------

SCREENING_DECISIONS: Dict[str, Dict[str, Any]] = {
    "fx-0001": {"decision": "exclude", "confidence": 0.94,
                "reason": "This study concerns predictive maintenance in industrial manufacturing, not rural healthcare disaster response.",
                "relevant_topics": []},
    "fx-0002": {"decision": "include", "confidence": 0.91,
                "reason": "System-dynamics simulation of rural hospital surge capacity during a hurricane evacuation directly matches the review's scope.",
                "relevant_topics": ["system dynamics simulation", "rural hospitals", "hurricane evacuation"]},
    "fx-0003": {"decision": "exclude", "confidence": 0.88,
                "reason": "Concerns urban emergency department overcrowding, not a rural healthcare setting or a disaster-response method.",
                "relevant_topics": []},
    "fx-0004": {"decision": "exclude", "confidence": 0.80,
                "reason": "Qualitative interviews on inter-agency coordination in coastal megacities; no MBSE or simulation method and not a rural setting.",
                "relevant_topics": []},
    "fx-0005": {"decision": "include", "confidence": 0.92,
                "reason": "Agent-based simulation of rural hospital evacuation logistics during a hurricane directly matches the review's scope.",
                "relevant_topics": ["agent-based simulation", "rural hospitals", "hurricane evacuation"]},
    "fx-0006": {"decision": "exclude", "confidence": 0.86,
                "reason": "A cost-effectiveness evaluation of telemedicine triage, not a disaster-response or simulation/MBSE study.",
                "relevant_topics": []},
    "fx-0008": {"decision": "exclude", "confidence": 0.83,
                "reason": "Proposes a blockchain traceability architecture, not an MBSE or computer-simulation method.",
                "relevant_topics": []},
    "fx-0009": {"decision": "include", "confidence": 0.93,
                "reason": "An MBSE (SysML) framework for rural clinic coordination during a regional flood is a direct match to the review's scope.",
                "relevant_topics": ["model-based systems engineering", "rural clinics", "flood response"]},
    "fx-0010": {"decision": "exclude", "confidence": 0.95,
                "reason": "Concerns wearable glucose-monitoring sensors in an outpatient diabetes clinic; unrelated to disaster planning.",
                "relevant_topics": []},
    "fx-0012": {"decision": "exclude", "confidence": 0.87,
                "reason": "Discrete-event simulation of airport security queuing; the simulation method is present but the domain is unrelated to healthcare or disaster response.",
                "relevant_topics": []},
    "fx-0013": {"decision": "include", "confidence": 0.90,
                "reason": "Discrete-event simulation of a rural community hospital's mass-casualty triage pathway directly matches the review's scope.",
                "relevant_topics": ["discrete-event simulation", "rural community hospital", "mass-casualty triage"]},
    "fx-0014": {"decision": "exclude", "confidence": 0.89,
                "reason": "A nutrition education programme evaluation in rural care homes; no simulation/MBSE method and no disaster context.",
                "relevant_topics": []},
    "fx-0015": {"decision": "exclude", "confidence": 0.62,
                "reason": "Rural and healthcare-related, but an observational staffing-ratio study with neither a simulation/MBSE method nor a disaster-response context.",
                "relevant_topics": []},
    "fx-0017": {"decision": "exclude", "confidence": 0.96,
                "reason": "A survey of public perception of autonomous vehicles in suburban transportation planning; unrelated to the review's scope.",
                "relevant_topics": []},
    "fx-0018": {"decision": "include", "confidence": 0.84,
                "reason": "Discrete-event simulation of ambulance dispatch during an epidemic outbreak in remote-region hospitals matches the review's scope at title/abstract level.",
                "relevant_topics": ["discrete-event simulation", "remote-region hospitals", "epidemic outbreak"]},
    "fx-0019": {"decision": "exclude", "confidence": 0.90,
                "reason": "A cybersecurity audit of hospital SCADA systems; no simulation/MBSE method and no disaster-response context.",
                "relevant_topics": []},
    "fx-0020": {"decision": "exclude", "confidence": 0.85,
                "reason": "A comparative statistical analysis of historical ambulance dispatch records, not a simulation or systems-engineering method.",
                "relevant_topics": []},
    "fx-0022": {"decision": "exclude", "confidence": 0.97,
                "reason": "A serious game for teaching Boolean logic to computer science students; entirely unrelated to healthcare or disaster response.",
                "relevant_topics": []},
    "fx-0024": {"decision": "include", "confidence": 0.72,
                "reason": "A viewpoint article advocating MBSE adoption for disaster-resilient rural health systems; conceptually within scope though it reports no specific empirical case.",
                "relevant_topics": ["model-based systems engineering", "rural health systems", "disaster resilience"]},
    "fx-0025": {"decision": "exclude", "confidence": 0.98,
                "reason": "Procedural terrain generation for video games; entirely unrelated to healthcare or disaster response.",
                "relevant_topics": []},
    "fx-0027": {"decision": "exclude", "confidence": 0.91,
                "reason": "A finite-element structural simulation of bridge retrofitting; the simulation method is present but the domain is structural engineering, not healthcare.",
                "relevant_topics": []},
    "fx-0029": {"decision": "exclude", "confidence": 0.93,
                "reason": "A business-strategy analysis of retail supply-chain resilience; unrelated to rural healthcare or disaster response.",
                "relevant_topics": []},
}

TITLE_ABSTRACT_INCLUSIONS: Tuple[str, ...] = (
    "fx-0002", "fx-0005", "fx-0009", "fx-0013", "fx-0018", "fx-0024",
)

# ---------------------------------------------------------------------------
# Full-text eligibility -- one entry per record in `TITLE_ABSTRACT_
# INCLUSIONS` (the 6 full-text-sought records), keyed by the 3 frozen
# criterion ids `slr_pipeline/reference_spec.py::FULL_TEXT_CRITERIA_SET`
# defines (`mbse_or_simulation`, `rural_healthcare`, `disaster_context`).
# `fx-0018` is the corpus's ONE full-text exclusion: its title/abstract
# reads as rural, but the full text reveals a peri-urban, metropolitan-
# adjacent hospital consortium -- the `rural_healthcare` criterion fails
# only once the full text is actually read, exercising the "passed title/
# abstract screening, excluded at full-text" pipeline branch.
# ---------------------------------------------------------------------------

FULL_TEXT_ELIGIBILITY: Dict[str, Dict[str, Any]] = {
    "fx-0002": {
        "eligible": True, "exclusion_reason_code": None,
        "criteria": {
            "mbse_or_simulation": {"met": True, "evidence": "A system-dynamics simulation model of hospital surge capacity is built and run under three evacuation scenarios."},
            "rural_healthcare": {"met": True, "evidence": "The simulated hospitals are explicitly three rural US county hospital systems."},
            "disaster_context": {"met": True, "evidence": "The scenarios simulate hurricane evacuation demand shocks."},
        },
    },
    "fx-0005": {
        "eligible": True, "exclusion_reason_code": None,
        "criteria": {
            "mbse_or_simulation": {"met": True, "evidence": "An agent-based simulation models patient and staff evacuation logistics under three staged-evacuation protocols."},
            "rural_healthcare": {"met": True, "evidence": "The simulated network consists of rural hospitals."},
            "disaster_context": {"met": True, "evidence": "The simulation is run under hurricane landfall and road-closure scenarios."},
        },
    },
    "fx-0009": {
        "eligible": True, "exclusion_reason_code": None,
        "criteria": {
            "mbse_or_simulation": {"met": True, "evidence": "The full text presents a SysML-based MBSE framework and a worked case model."},
            "rural_healthcare": {"met": True, "evidence": "The framework coordinates a network of rural primary-care clinics."},
            "disaster_context": {"met": True, "evidence": "The coordination scenario is a regional flood disaster."},
        },
    },
    "fx-0013": {
        "eligible": True, "exclusion_reason_code": None,
        "criteria": {
            "mbse_or_simulation": {"met": True, "evidence": "A discrete-event simulation of the triage pathway is built and run under two accident scenarios."},
            "rural_healthcare": {"met": True, "evidence": "The simulated hospital is explicitly a single rural community hospital."},
            "disaster_context": {"met": True, "evidence": "The simulated scenarios are a multi-vehicle accident and a tornado strike."},
        },
    },
    "fx-0018": {
        "eligible": False, "exclusion_reason_code": "not_rural_healthcare",
        "criteria": {
            "mbse_or_simulation": {"met": True, "evidence": "A discrete-event simulation model compares static and dynamic ambulance-dispatch heuristics."},
            "rural_healthcare": {"met": False, "evidence": "The full text describes the hospital network as a well-resourced peri-urban consortium serving a metropolitan-adjacent catchment, not a genuinely rural population."},
            "disaster_context": {"met": True, "evidence": "The scenario is an epidemic outbreak affecting the hospital network."},
        },
    },
    "fx-0024": {
        "eligible": True, "exclusion_reason_code": None,
        "criteria": {
            "mbse_or_simulation": {"met": True, "evidence": "The full text advocates for, and describes at a conceptual level, MBSE adoption as a design approach."},
            "rural_healthcare": {"met": True, "evidence": "The full text explicitly frames its argument around rural health systems."},
            "disaster_context": {"met": True, "evidence": "The full text explicitly frames its argument around disaster resilience."},
        },
    },
}

FULL_TEXT_SOUGHT: Tuple[str, ...] = TITLE_ABSTRACT_INCLUSIONS
FULL_TEXT_EXCLUSIONS: Tuple[str, ...] = ("fx-0018",)
FULL_TEXT_ELIGIBLE: Tuple[str, ...] = ("fx-0002", "fx-0005", "fx-0009", "fx-0013", "fx-0024")

# `fx-0024` is a viewpoint article: eligible (it discusses MBSE/rural/
# disaster at a conceptual level) but names no specific empirical method,
# setting, or disaster type, so it does not contribute evidence to RQ1
# ("what MBSE methods, healthcare settings, disaster types, and contexts
# have been studied") even though it does answer RQ0 ("is there a body of
# evidence...").
SYNTHESIS_MAP_RELEVANCE: Dict[str, Tuple[str, ...]] = {
    "rq_0": FULL_TEXT_ELIGIBLE,
    "rq_1": tuple(rid for rid in FULL_TEXT_ELIGIBLE if rid != "fx-0024"),
}

# ---------------------------------------------------------------------------
# Extraction (charting) answers -- one entry per `FULL_TEXT_ELIGIBLE`
# record, keyed exactly to `reference_charting_adapter.py::
# reference_extraction_schema()`'s 9 fields. The 5 funding-related field
# pairs deliberately cover 5 distinct branches: present-with-role
# (fx-0002), the exact "No external funding" canned phrase (fx-0005), the
# exact "The funder had no role in the study." canned phrase (fx-0009), no
# funding statement at all in the paper (fx-0013), and a funder-mentioned-
# but-non-canned, ambiguous role statement (fx-0024).
# ---------------------------------------------------------------------------

EXTRACTION_ANSWERS: Dict[str, Dict[str, Any]] = {
    "fx-0002": {
        "country": "United States",
        "study_design": "Simulation study (system dynamics model)",
        "population": "Three rural county hospital systems and their catchment populations",
        "concept": "Surge-capacity system dynamics simulation for hospital bed and staffing allocation",
        "context": "Rural hurricane-evacuation disaster response",
        "limitations": "The authors state the model was calibrated against a single historical hurricane season and may not generalise to slower-onset disasters.",
        "key_findings": [
            "Dynamic bed-reallocation policies reduced simulated patient wait times by 18% compared to static allocation.",
            "Staffing shortfalls emerged earliest in the smallest of the three simulated county hospitals.",
        ],
        "funding_sources": ["National Institute for Rural Health Systems (fictional, synthetic fixture)"],
        "funder_role": "The funder participated in study design by specifying which hurricane scenarios to simulate, but had no role in data analysis or the decision to publish.",
    },
    "fx-0005": {
        "country": "United States",
        "study_design": "Simulation study (agent-based model)",
        "population": "A network of five rural hospitals and their nursing and physician staff",
        "concept": "Agent-based simulation of patient and staff evacuation logistics",
        "context": "Rural hospital hurricane evacuation planning",
        "limitations": "The authors note the agent behaviours were parameterised from a single prior evacuation exercise and not validated against a real hurricane evacuation.",
        "key_findings": [
            "A staged evacuation protocol completed evacuation 22% faster than a simultaneous-evacuation protocol in simulation.",
            "Road-closure scenarios most affected the two hospitals furthest from the regional trauma centre.",
        ],
        "funding_sources": ["No external funding"],
        "funder_role": "Not applicable; the study received no external funding.",
    },
    "fx-0009": {
        "country": "United States",
        "study_design": "Model-based systems engineering (MBSE) framework development, illustrated with a SysML case model",
        "population": "A network of rural primary-care clinics coordinating regional flood response",
        "concept": "MBSE framework for structuring emergency medical response coordination",
        "context": "Rural regional flood disaster response",
        "limitations": "The authors state the framework has not yet been validated against a real flood event and was demonstrated only on a synthetic coordination scenario.",
        "key_findings": [
            "The SysML-based framework surfaced two previously undocumented coordination gaps between clinics.",
            "Framework users reported the model improved shared situational awareness in a tabletop exercise.",
        ],
        "funding_sources": ["Regional Rural Health Consortium (fictional, synthetic fixture)"],
        "funder_role": "The funder had no role in the study.",
    },
    "fx-0013": {
        "country": "United States",
        "study_design": "Simulation study (discrete-event simulation)",
        "population": "Staff and patients at a single rural community hospital",
        "concept": "Discrete-event simulation of mass-casualty triage pathway staffing",
        "context": "Rural mass-casualty and tornado-strike disaster response",
        "limitations": "The authors state that the model did not account for staff absenteeism during the simulated disaster itself.",
        "key_findings": [
            "Adding one additional triage nurse reduced simulated time-to-triage by 31% under the tornado-strike scenario.",
            "The multi-vehicle-accident scenario did not saturate triage capacity at current staffing levels.",
        ],
        "funding_sources": [],
        "funder_role": None,
    },
    "fx-0024": {
        "country": "Not stated (viewpoint article; no empirical fieldwork site is named)",
        "study_design": "Viewpoint / discussion article",
        "population": "Not stated (no specific empirical population; the article draws on the authors' general consulting experience)",
        "concept": "Advocacy for model-based systems engineering in disaster-resilient rural health system design",
        "context": "Rural health system disaster resilience, discussed at a general/conceptual level",
        "limitations": "The authors state the viewpoint does not present new empirical data and call for future empirical validation.",
        "key_findings": [
            "The authors argue that MBSE adoption in rural health systems currently lags behind urban systems.",
            "The authors recommend a phased MBSE adoption roadmap for rural health authorities.",
        ],
        "funding_sources": ["Independent Rural Systems Institute (fictional, synthetic fixture)"],
        "funder_role": (
            "The paper states the funder reviewed the manuscript prior to submission, but does not describe "
            "this as a role in study design, analysis, or the decision to publish."
        ),
    },
}

# ---------------------------------------------------------------------------
# Scenario table -- one compact row per fixture record, rendered from the
# data above (never independently authored), naming the pipeline branch
# each record exercises.
# ---------------------------------------------------------------------------

_EXCLUDE_SCENARIO_NOTE: Dict[str, str] = {
    "fx-0001": "title/abstract exclude — off-topic (industrial predictive maintenance)",
    "fx-0003": "title/abstract exclude — wrong setting (urban ED); duplicate-of-title canonical (fx-0007 dup)",
    "fx-0004": "title/abstract exclude — qualitative only, no MBSE/simulation; duplicate-of-title canonical (fx-0011 dup)",
    "fx-0006": "title/abstract exclude — no disaster context; duplicate-of-title canonical (fx-0030 dup)",
    "fx-0008": "title/abstract exclude — wrong method (blockchain, not simulation/MBSE); duplicate-of-title canonical (fx-0023 dup)",
    "fx-0010": "title/abstract exclude — entirely off-topic (wearable glucose monitoring)",
    "fx-0012": "title/abstract exclude — simulation present, wrong domain (airport security); duplicate-of-title canonical (fx-0028 dup)",
    "fx-0014": "title/abstract exclude — no simulation/MBSE, no disaster context; duplicate-of-title canonical (fx-0021 dup)",
    "fx-0015": "title/abstract exclude — rural+healthcare present but BOTH other criteria fail (harder near-miss, lower confidence)",
    "fx-0017": "title/abstract exclude — entirely off-topic (autonomous vehicle perception)",
    "fx-0019": "title/abstract exclude — off-topic (SCADA cybersecurity); duplicate-of-title canonical (fx-0026 dup)",
    "fx-0020": "title/abstract exclude — descriptive/statistical, no simulation/MBSE method",
    "fx-0022": "title/abstract exclude — entirely off-topic (serious game for CS education)",
    "fx-0025": "title/abstract exclude — entirely off-topic (video-game terrain generation)",
    "fx-0027": "title/abstract exclude — simulation present, wrong domain (structural engineering); duplicate-of-title canonical (fx-0028's own db-pair, see fx-0012)",
    "fx-0029": "title/abstract exclude — entirely off-topic (retail supply-chain strategy)",
}


def scenario_table() -> List[Dict[str, str]]:
    """One row per of the 30 fixture records, rendered from
    `_RAW_RECORDS`/`SCREENING_DECISIONS`/`FULL_TEXT_ELIGIBILITY`/
    `EXTRACTION_ANSWERS` -- never independently authored a second time."""
    rows: List[Dict[str, str]] = []
    for r in _RAW_RECORDS:
        rid = r["id"]
        if r["duplicate_of"] is not None:
            branch = f"dedup exact-title duplicate of {r['duplicate_of']} (basis: {r['duplicate_basis']})"
        elif rid not in TITLE_ABSTRACT_INCLUSIONS:
            branch = _EXCLUDE_SCENARIO_NOTE.get(rid, "title/abstract exclude")
        elif rid in FULL_TEXT_EXCLUSIONS:
            ft = FULL_TEXT_ELIGIBILITY[rid]
            branch = f"title/abstract include -> FULL-TEXT EXCLUDED (reason: {ft['exclusion_reason_code']})"
        else:
            extra = ""
            if rid in EXTRACTION_ANSWERS:
                funding = EXTRACTION_ANSWERS[rid]["funding_sources"]
                role = EXTRACTION_ANSWERS[rid]["funder_role"]
                if not funding:
                    extra = "; funding statement ABSENT"
                elif funding == ["No external funding"]:
                    extra = "; funding: 'No external funding' canned phrase"
                elif role == "The funder had no role in the study.":
                    extra = "; funding: funder-had-no-role canned phrase"
                elif rid == "fx-0024":
                    extra = "; funding: present, ambiguous non-canned role text"
                else:
                    extra = "; funding: present with a substantive stated role"
            branch = f"title/abstract include -> full-text eligible -> extracted{extra}"
        rows.append({
            "id": rid, "source_db": r["source_db"],
            "title": r["title"], "branch": branch,
        })
    return rows


def scenario_table_markdown() -> str:
    rows = scenario_table()
    lines = [
        "| fx id | source db | branch |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['id']} | {row['source_db']} | {row['branch']} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Cross-checks against `CORPUS_ORACLE.json`'s frozen design literals --
# verified here so this module and the oracle can never silently drift
# apart (called by `scripts/author_fixture_corpus.py` before it writes
# anything, and by the dedicated pytest module).
# ---------------------------------------------------------------------------

_ORACLE_RECORDS_PER_DB: Dict[str, int] = {
    "fx_pubmed": 6, "fx_web_of_science": 5, "fx_cinahl": 4,
    "fx_scopus": 5, "fx_acm": 4, "fx_engineering_village": 3,
    "fx_business_source_complete": 3,
}


class AnswerKeyTopologyError(ValueError):
    """This module's own authored data contradicts `CORPUS_ORACLE.json`'s
    frozen design literals -- an authoring bug, never silently tolerated."""


def verify_topology() -> None:
    if len(_RAW_RECORDS) != 30:
        raise AnswerKeyTopologyError(f"expected 30 raw records, got {len(_RAW_RECORDS)}")
    if len(set(all_ids())) != 30:
        raise AnswerKeyTopologyError("raw record ids are not all unique")

    grouped = records_by_db()
    for db, expected_count in _ORACLE_RECORDS_PER_DB.items():
        short = db.removeprefix("fx_")
        actual = len(grouped.get(short, []))
        if actual != expected_count:
            raise AnswerKeyTopologyError(f"{db}: expected {expected_count} records, got {actual}")

    if len(duplicate_ids()) != 8:
        raise AnswerKeyTopologyError(f"expected 8 duplicates, got {len(duplicate_ids())}")
    if len(canonical_ids()) != 22:
        raise AnswerKeyTopologyError(f"expected 22 canonical records, got {len(canonical_ids())}")
    if set(TITLE_ABSTRACT_INCLUSIONS) - set(canonical_ids()):
        raise AnswerKeyTopologyError("a title/abstract inclusion is not a canonical record")
    if len(TITLE_ABSTRACT_INCLUSIONS) != 6:
        raise AnswerKeyTopologyError(f"expected 6 title/abstract inclusions, got {len(TITLE_ABSTRACT_INCLUSIONS)}")
    if set(SCREENING_DECISIONS) != set(canonical_ids()):
        raise AnswerKeyTopologyError("SCREENING_DECISIONS keys must be exactly the 22 canonical ids")
    included_by_decision = {rid for rid, d in SCREENING_DECISIONS.items() if d["decision"] == "include"}
    if included_by_decision != set(TITLE_ABSTRACT_INCLUSIONS):
        raise AnswerKeyTopologyError("SCREENING_DECISIONS' include set must equal TITLE_ABSTRACT_INCLUSIONS")

    if set(FULL_TEXT_SOUGHT) != set(TITLE_ABSTRACT_INCLUSIONS):
        raise AnswerKeyTopologyError("FULL_TEXT_SOUGHT must equal TITLE_ABSTRACT_INCLUSIONS")
    if set(FULL_TEXT_ELIGIBILITY) != set(FULL_TEXT_SOUGHT):
        raise AnswerKeyTopologyError("FULL_TEXT_ELIGIBILITY keys must be exactly FULL_TEXT_SOUGHT")
    eligible_by_criteria = {rid for rid, ft in FULL_TEXT_ELIGIBILITY.items() if ft["eligible"]}
    if eligible_by_criteria != set(FULL_TEXT_ELIGIBLE):
        raise AnswerKeyTopologyError("FULL_TEXT_ELIGIBILITY's eligible set must equal FULL_TEXT_ELIGIBLE")
    if set(FULL_TEXT_EXCLUSIONS) != set(FULL_TEXT_SOUGHT) - set(FULL_TEXT_ELIGIBLE):
        raise AnswerKeyTopologyError("FULL_TEXT_EXCLUSIONS must equal FULL_TEXT_SOUGHT minus FULL_TEXT_ELIGIBLE")
    if len(FULL_TEXT_ELIGIBLE) != 5:
        raise AnswerKeyTopologyError(f"expected 5 full-text-eligible records, got {len(FULL_TEXT_ELIGIBLE)}")

    if set(EXTRACTION_ANSWERS) != set(FULL_TEXT_ELIGIBLE):
        raise AnswerKeyTopologyError("EXTRACTION_ANSWERS keys must be exactly FULL_TEXT_ELIGIBLE")

    # Every duplicate's canonical must be authored strictly earlier in
    # `_RAW_RECORDS` order -- the `_dedup_doi_only` first-seen-wins
    # precondition this whole corpus depends on.
    position = {r["id"]: i for i, r in enumerate(_RAW_RECORDS)}
    for r in _RAW_RECORDS:
        if r["duplicate_of"] is not None and position[r["duplicate_of"]] >= position[r["id"]]:
            raise AnswerKeyTopologyError(
                f"{r['id']}'s canonical {r['duplicate_of']} must appear earlier in raw_records order"
            )

    # `all_fixture_records_doi_less`: every record's authored native
    # payload must have doi=None.
    for db in CANONICAL_DATABASES:
        for native in native_payload(db):
            if native["doi"] is not None:
                raise AnswerKeyTopologyError(f"{native['id']}: doi must be None (all_fixture_records_doi_less)")


def corpus_oracle_payload(pdf_sha256: Dict[str, str]) -> Dict[str, Any]:
    """Assembles `CORPUS_ORACLE.json`'s full payload (`slr_pipeline.
    corpus_oracle.CorpusOracle`'s schema) from this module's own data plus
    the 8-surface call-topology counts already frozen in
    `replay/cassette.py::REFERENCE_SURFACE_CALL_COUNTS` -- imported, never
    retyped. `pdf_sha256` is the one genuine build output this function
    cannot compute itself (the caller must have already generated the 6
    synthetic PDFs and hashed their actual bytes)."""
    from slr_pipeline.corpus_oracle import CORPUS_ORACLE_VERSION, ORACLE_KIND
    from slr_pipeline.replay.cassette import REFERENCE_SURFACE_CALL_COUNTS as C

    verify_topology()
    expected_pdf_ids = set(FULL_TEXT_SOUGHT)
    if set(pdf_sha256) != expected_pdf_ids:
        raise AnswerKeyTopologyError(f"pdf_sha256 keys must be exactly {sorted(expected_pdf_ids)}")

    total_replay_calls = sum(C.values())
    return {
        "corpus_oracle_version": CORPUS_ORACLE_VERSION,
        "oracle_kind": ORACLE_KIND,
        "raw_records": len(_RAW_RECORDS),
        "records_per_fixture_db": {f"fx_{db}": len(recs) for db, recs in records_by_db().items()},
        "duplicates_removed": len(duplicate_ids()),
        "_dedup_note": (
            "Every fixture record has doi=None, so all 8 duplicates are removed by the exact "
            "normalised-title match (casefold + NFKC + collapse-whitespace + strip). The DOI-exact "
            "path is unit-tested separately with in-memory synthetic Record objects, not in this corpus."
        ),
        "duplicate_mappings": duplicate_mappings(),
        "unique_after_dedup": len(canonical_ids()),
        "title_abstract_inclusions": list(TITLE_ABSTRACT_INCLUSIONS),
        "all_fixture_records_doi_less": True,
        "local_pdfs_present": len(FULL_TEXT_SOUGHT),
        "full_text_sought": len(FULL_TEXT_SOUGHT),
        "full_text_review_queue_entries": 0,
        "full_text_exclusions": list(FULL_TEXT_EXCLUSIONS),
        "full_text_eligible": list(FULL_TEXT_ELIGIBLE),
        "research_questions": 2,
        "map_relevant_by_rq": {rq: list(ids) for rq, ids in SYNTHESIS_MAP_RELEVANCE.items()},
        "screening_batches": C["title_abstract_screening"],
        "qc_sample_size": C["qc_second_rater"],
        "ft_adjudication_calls": C["ft_adjudication"],
        "extraction_batch_calls": C["extraction_batch"],
        "synthesis_map_calls": C["synthesis_map"],
        "synthesis_refine_calls": C["synthesis_refine"],
        "total_replay_calls": total_replay_calls,
        "pdf_sha256": dict(pdf_sha256),
    }


def dump_native_fixture_files(root: Path) -> List[Path]:
    """Writes `native/fx_<db>.json` for all 7 canonical databases.
    `verify_topology()` runs first -- this function never writes an
    internally-inconsistent corpus. `root` is required (Track-2 packaging-
    boundary fix, CI run 34155102694): this module is `slr_pipeline`-wheel-
    packaged and must not assume a checkout-relative `tests/` fixture
    default -- both real callers already pass `root=` explicitly, so this
    is a signature-only fix for consistency, not a behavior change."""
    verify_topology()
    native_dir = root / "native"
    native_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for db in CANONICAL_DATABASES:
        path = native_dir / f"fx_{db}.json"
        text = json.dumps(native_payload(db), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written
