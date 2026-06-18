"""
Country Configuration Abstraction
===================================
Defines the CountryConfig dataclass that encapsulates all country-specific
parameters for the debate system. This abstraction ensures the system can
be extended to a second country (Netherlands recommended, UK fallback)
with minimal refactoring.

Design principles:
  1. Each country provides a self-contained config: party list, personas,
     VAA data path, manifesto corpus path, debate topics, and language.
  2. The orchestrator, RAG pipeline, and trigger-check module are
     country-agnostic — they consume CountryConfig without knowing which
     country is active.
  3. Persona prompts are parameterised: a template takes party name,
     ideology summary, rhetorical style, and policy orientation as vars.
  4. The number of agents is NOT hard-coded — derived from the config's
     party list.
  5. Language-agnostic embedding model (paraphrase-multilingual-MiniLM-L12-v2)
     already supports Dutch/German/English.

Extension path for Netherlands:
  - StemWijzer data (30 theses × party) replaces Wahl-O-Mat
  - Dutch Partijprogramma's replace Bundestagswahlprogramme
  - 6-8 parties selected from 15+ (matching German party count)
  - Debates run in English with translated grounding, OR in Dutch
    using the multilingual embedding model

See also:
  - mds/second_country_memo.md — full research memo on country selection
  - expose.tex §4.X (Cross-Country Extensibility)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass
class PartyConfig:
    """Configuration for a single political party agent."""
    name: str                      # Display name (e.g. "CDU/CSU", "VVD")
    short_name: str                # Short identifier for file stems (e.g. "cdu_csu")
    ideology_summary: str          # 1-sentence ideology description
    rhetorical_style: str          # 1-sentence rhetorical style description
    policy_orientation: str        # Key policy orientation keywords
    persona_prompt: Optional[str] = None  # Full persona prompt (if pre-written)


@dataclass
class DebateTopic:
    """A debate topic with its neutral framing prompt."""
    id: str                        # Short identifier (e.g. "mindestlohn")
    title: str                     # Human-readable title
    framing_prompt: str            # Neutral framing prompt for debate
    topic_type: str = "empirical"  # "empirical" or "values-driven"


@dataclass
class CountryConfig:
    """Complete configuration for one country's debate setup.

    The orchestrator, RAG pipeline, trigger module, and judge module
    all consume this config without knowing which country is active.
    """
    # --- Identity ---
    country_code: str              # ISO 3166-1 alpha-2 (e.g. "DE", "NL", "GB")
    country_name: str              # Full name (e.g. "Germany", "Netherlands")
    language: str                  # Primary debate language (e.g. "German", "English")
    debate_language: str = "German"  # Language debates are conducted in

    # --- Parties ---
    parties: list[PartyConfig] = field(default_factory=list)
    turn_order: list[str] = field(default_factory=list)  # Party names in turn order

    # --- Data paths ---
    data_dir: Optional[Path] = None           # Root data directory for this country
    vaa_data_path: Optional[Path] = None      # Structured VAA data (Wahl-O-Mat / StemWijzer)
    manifesto_chunks_dir: Optional[Path] = None  # Pre-chunked manifesto passages
    embeddings_dir: Optional[Path] = None     # FAISS indices directory

    # --- VAA metadata ---
    vaa_name: str = ""             # e.g. "Wahl-O-Mat", "StemWijzer"
    vaa_thesis_count: int = 0      # Number of VAA theses
    election_name: str = ""        # e.g. "Bundestagswahl 2025"

    # --- Debate topics ---
    topics: list[DebateTopic] = field(default_factory=list)

    # --- Model config ---
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    agent_model: str = "llama-3.1-8b-instant"
    judge_model: str = "llama-3.3-70b-versatile"

    # --- Pipeline parameters ---
    initial_top_k: int = 15
    rerank_top_n: int = 3
    max_interventions: int = 3
    silent_control_rate: float = 0.20
    rounds_per_debate: int = 10
    iterations_per_condition: int = 5

    @property
    def num_agents(self) -> int:
        """Number of agents derived from party list."""
        return len(self.parties)

    @property
    def party_names(self) -> list[str]:
        """List of party display names."""
        return [p.name for p in self.parties]

    @property
    def party_stems(self) -> dict[str, str]:
        """Map party display name → file stem for RAG indices."""
        return {p.name: p.short_name for p in self.parties}

    def validate(self) -> list[str]:
        """Check config completeness. Returns list of issues."""
        issues = []
        if not self.parties:
            issues.append("No parties configured")
        if not self.turn_order:
            issues.append("No turn order specified")
        if self.turn_order and set(self.turn_order) != set(self.party_names):
            issues.append("Turn order does not match party names")
        if self.vaa_data_path and not self.vaa_data_path.exists():
            issues.append(f"VAA data path does not exist: {self.vaa_data_path}")
        if self.manifesto_chunks_dir and not self.manifesto_chunks_dir.exists():
            issues.append(f"Chunks dir does not exist: {self.manifesto_chunks_dir}")
        return issues


# ---------------------------------------------------------------------------
# Germany (production config)
# ---------------------------------------------------------------------------

def get_germany_config(data_dir: Optional[Path] = None) -> CountryConfig:
    """Return the production CountryConfig for Germany (BTW 2025).

    This is the primary configuration used in the thesis experiment.
    Party personas are loaded from src/prompts/agent_prompts.py.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "data"

    parties = [
        PartyConfig(
            name="CDU/CSU",
            short_name="cdu_csu_2025",
            ideology_summary="Centre-right Christian democratic; social market economy",
            rhetorical_style="Measured, institutional; appeals to stability and experience",
            policy_orientation="fiscal conservatism, law and order, EU integration",
        ),
        PartyConfig(
            name="SPD",
            short_name="spd_2025",
            ideology_summary="Centre-left social democratic; workers' rights and welfare state",
            rhetorical_style="Pragmatic, empathetic; emphasises fairness and social justice",
            policy_orientation="minimum wage, pension security, affordable housing",
        ),
        PartyConfig(
            name="Bündnis 90/Die Grünen",
            short_name="gruene_2025",
            ideology_summary="Green progressive; ecological sustainability and social equity",
            rhetorical_style="Morally urgent, policy-dense; frames issues as systemic crises",
            policy_orientation="climate action, renewable energy, diversity",
        ),
        PartyConfig(
            name="FDP",
            short_name="fdp_2025",
            ideology_summary="Liberal; individual freedom, market economy, deregulation",
            rhetorical_style="Confident, market-oriented; appeals to entrepreneurship and innovation",
            policy_orientation="tax cuts, deregulation, digital transformation",
        ),
        PartyConfig(
            name="Die Linke",
            short_name="linke_2025",
            ideology_summary="Democratic socialist; redistribution, anti-militarism, social rights",
            rhetorical_style="Combative, structural critique; frames issues as class conflict",
            policy_orientation="wealth tax, public ownership, peace policy",
        ),
        PartyConfig(
            name="AfD",
            short_name="afd_2025",
            ideology_summary="National-conservative; anti-immigration, Eurosceptic, traditional values",
            rhetorical_style="Direct, populist; anti-establishment framing, appeals to common sense",
            policy_orientation="migration restriction, EU scepticism, national sovereignty",
        ),
    ]

    # Import canonical framing prompts from experiment config
    from src.experiment_config import FRAMING_PROMPTS

    topics = [
        DebateTopic(
            id="mindestlohn",
            title="Mindestlohn",
            framing_prompt=FRAMING_PROMPTS["mindestlohn"],
            topic_type="empirical",
        ),
        DebateTopic(
            id="rentenpolitik",
            title="Rentenpolitik",
            framing_prompt=FRAMING_PROMPTS["rentenpolitik"],
            topic_type="empirical",
        ),
        DebateTopic(
            id="migrationspolitik",
            title="Migrationspolitik",
            framing_prompt=FRAMING_PROMPTS["migrationspolitik"],
            topic_type="values-driven",
        ),
        DebateTopic(
            id="sozialpolitik",
            title="Sozialpolitik / Vermögensungleichheit",
            framing_prompt=FRAMING_PROMPTS["sozialpolitik"],
            topic_type="values-driven",
        ),
    ]

    return CountryConfig(
        country_code="DE",
        country_name="Germany",
        language="German",
        debate_language="German",
        parties=parties,
        turn_order=[p.name for p in parties],
        data_dir=data_dir,
        vaa_data_path=data_dir / "wahlomat" / "wahlomat_2025.json",
        manifesto_chunks_dir=data_dir / "chunks",
        embeddings_dir=data_dir / "embeddings",
        vaa_name="Wahl-O-Mat",
        vaa_thesis_count=38,
        election_name="Bundestagswahl 2025",
        topics=topics,
    )


# ---------------------------------------------------------------------------
# Netherlands (placeholder for future implementation)
# ---------------------------------------------------------------------------

def get_netherlands_config(data_dir: Optional[Path] = None) -> CountryConfig:
    """Return a placeholder CountryConfig for the Netherlands.

    This config outlines the expected structure for the NL extension.
    Actual party personas and data paths will be populated when
    StemWijzer data and Dutch manifesto chunks are prepared.

    Recommended parties (6 selected from 15+ parliamentary parties):
      VVD, PVV, GL-PvdA, NSC, D66, SP
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "data_nl"

    parties = [
        PartyConfig(
            name="VVD",
            short_name="vvd_2023",
            ideology_summary="Conservative-liberal; free market, law and order",
            rhetorical_style="Business-oriented, pragmatic; appeals to fiscal responsibility",
            policy_orientation="tax reduction, entrepreneurship, tough immigration",
        ),
        PartyConfig(
            name="PVV",
            short_name="pvv_2023",
            ideology_summary="Right-wing populist; anti-Islam, Eurosceptic, welfare chauvinism",
            rhetorical_style="Direct, provocative; anti-establishment, plain language",
            policy_orientation="immigration halt, EU exit (Nexit), lower healthcare costs",
        ),
        PartyConfig(
            name="GL-PvdA",
            short_name="gl_pvda_2023",
            ideology_summary="Green-left / social democratic alliance; sustainability and equality",
            rhetorical_style="Progressive, morally urgent; emphasises solidarity and climate",
            policy_orientation="climate action, redistribution, workers' rights",
        ),
        PartyConfig(
            name="NSC",
            short_name="nsc_2023",
            ideology_summary="Christian-social centrist; rule of law, political reform",
            rhetorical_style="Principled, institutional; appeals to norms and trust",
            policy_orientation="constitutional reform, housing, migration control",
        ),
        PartyConfig(
            name="D66",
            short_name="d66_2023",
            ideology_summary="Social-liberal; progressive, pro-EU, education-focused",
            rhetorical_style="Optimistic, evidence-based; emphasises innovation and freedom",
            policy_orientation="education investment, EU integration, climate",
        ),
        PartyConfig(
            name="SP",
            short_name="sp_2023",
            ideology_summary="Socialist; anti-privatisation, healthcare, workers' protection",
            rhetorical_style="Combative, class-conscious; anti-corporate framing",
            policy_orientation="public healthcare, affordable housing, wealth tax",
        ),
    ]

    return CountryConfig(
        country_code="NL",
        country_name="Netherlands",
        language="Dutch",
        debate_language="English",  # Run in English with translated grounding
        parties=parties,
        turn_order=[p.name for p in parties],
        data_dir=data_dir,
        vaa_data_path=data_dir / "stemwijzer" / "stemwijzer_2023.json" if data_dir else None,
        manifesto_chunks_dir=data_dir / "chunks" if data_dir else None,
        embeddings_dir=data_dir / "embeddings" if data_dir else None,
        vaa_name="StemWijzer",
        vaa_thesis_count=30,
        election_name="Tweede Kamerverkiezingen 2023",
        topics=[],  # To be defined
    )
