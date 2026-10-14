"""
Vector diagrams for the thesis (Figures 3.1 and 3.2).

Drawn at printed size (6.3 in wide), so all text is at least 8 pt in print.

Usage:
    python experiments/generate_diagrams.py

Output:
    paper/figures/architecture.pdf, paper/figures/timeline.pdf (+ PNG previews)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans",
                    "font.size": 8, "pdf.fonttype": 42})
FS, FS_TITLE = 8, 9

PARTIES = [("CDU/CSU", "#1A1A1A", "white"), ("SPD", "#E3000F", "white"),
           ("Grüne", "#64A12D", "white"), ("FDP", "#FFED00", "black"),
           ("Die Linke", "#BE3075", "white"), ("AfD", "#009EE0", "white")]


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, body="", fill="#f3f4f6", edge="#374151", tcolor="black", lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=fill, edgecolor=edge, linewidth=lw))
    if body:
        ax.text(x + w / 2, y + h - 0.1, title, ha="center", va="top", fontsize=FS_TITLE,
                fontweight="bold", color=tcolor)
        ax.text(x + w / 2, y + h - 0.32, body, ha="center", va="top", fontsize=FS,
                color=tcolor, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=FS,
                fontweight="bold", color=tcolor)


def arrow(ax, p, q, label=None, dashed=False, lpos=0.5, loff=(0, 0.07), ha="center", color="#111827"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, linewidth=1.0,
                                 color=color, linestyle=(
                                     0, (4, 2)) if dashed else "-",
                                 shrinkA=0, shrinkB=0))
    if label:
        x = p[0] + (q[0] - p[0]) * lpos + loff[0]
        y = p[1] + (q[1] - p[1]) * lpos + loff[1]
        ax.text(x, y, label, ha=ha, va="bottom",
                fontsize=FS, color=color, linespacing=1.15)


# ─── Figure 3.1: architecture ────────────────────────────────────────────────

def architecture():
    W, H = 6.3, 4.6
    fig, ax = canvas(W, H)

    # Agents, horizontally, in party colours.
    ax.add_patch(FancyBboxPatch((0.1, 3.45), 6.1, 1.05, boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor="#eff6ff", edgecolor="#1d4ed8", linewidth=1.0))
    ax.text(3.15, 4.4, "Six political agents (llama-3.1-8b-instant): party persona + retrieved passages",
            ha="center", va="top", fontsize=FS_TITLE, fontweight="bold")
    bw, gap = 0.88, (6.1 - 0.2 - 6 * 0.88) / 5
    for i, (name, fill, tc) in enumerate(PARTIES):
        box(ax, 0.2 + i * (bw + gap), 3.55, bw, 0.4,
            name, fill=fill, edge=fill, tcolor=tc)

    # Middle row.
    box(ax, 0.1, 1.75, 1.6, 1.3, "RAG pipeline",
        "per-party FAISS index\n(programmes, Wahl-O-Mat)\nparaphrase-multilingual-\nMiniLM-L12-v2",
        fill="#e0f2fe", edge="#0369a1")
    box(ax, 2.2, 1.6, 1.9, 1.6, "Orchestrator",
        "turn order (round robin),\ndebate state,\ncontext assembly,\nevent logging",
        fill="#fce7f3", edge="#be185d", lw=1.6)
    box(ax, 4.6, 1.75, 1.6, 1.3, "LLM judge",
        "Llama 4 Scout\nscores every turn\non 7 dimensions (1–5)",
        fill="#ffedd5", edge="#c2410c")

    # Bottom row.
    box(ax, 0.1, 0.35, 1.6, 0.95, "JSON log", "turns, scores,\ninterventions",
        fill="#f3f4f6", edge="#4b5563")
    box(ax, 2.2, 0.35, 1.9, 0.95, "Moderator", "Llama 4 Scout,\nstrategy prompt (A–D)",
        fill="#dcfce7", edge="#15803d")
    box(ax, 4.6, 0.35, 1.6, 0.95, "Trigger check",
        "Stage 1: score < 4\nStage 2: judge confirms\n20% silent, max. 3",
        fill="#fef9c3", edge="#a16207")

    # Orchestrator <-> agents.
    arrow(ax, (2.85, 3.2), (2.85, 3.45), "context",
          lpos=0.0, loff=(-0.06, 0.02), ha="right")
    arrow(ax, (3.45, 3.45), (3.45, 3.2), "turn",
          lpos=1.0, loff=(0.06, 0.02), ha="left")
    # Orchestrator <-> RAG.
    arrow(ax, (2.2, 2.7), (1.7, 2.7), "query", loff=(0, 0.04))
    arrow(ax, (1.7, 2.1), (2.2, 2.1), "top-3", loff=(0, 0.04))
    # Orchestrator <-> judge.
    arrow(ax, (4.1, 2.7), (4.6, 2.7), "turn", loff=(0, 0.04))
    arrow(ax, (4.6, 2.1), (4.1, 2.1), "scores", loff=(0, 0.04))
    # Orchestrator -> trigger check; trigger check <-> judge (Stage 2).
    arrow(ax, (4.1, 1.75), (4.6, 1.15), "scores",
          lpos=0.6, loff=(-0.06, -0.06), ha="right")
    arrow(ax, (5.3, 1.3), (5.3, 1.75), "confirm?",
          lpos=0.5, loff=(-0.05, -0.07), ha="right")
    arrow(ax, (5.65, 1.75), (5.65, 1.3), "yes/no",
          lpos=0.5, loff=(0.05, -0.07), ha="left")
    # Conditional path: trigger -> moderator -> orchestrator.
    arrow(ax, (4.6, 0.7), (4.1, 0.7), "fire", dashed=True, loff=(0, 0.04))
    arrow(ax, (3.15, 1.3), (3.15, 1.6), "intervention", dashed=True, lpos=0.5,
          loff=(0.06, -0.07), ha="left")
    # Orchestrator -> log.
    arrow(ax, (2.2, 1.75), (1.7, 1.15), "events",
          lpos=0.45, loff=(-0.05, 0.0), ha="right")

    # Legend.
    ax.plot([0.15, 0.5], [0.12, 0.12], color="#111827", linewidth=1.0)
    ax.text(0.58, 0.12, "every turn", va="center", fontsize=FS)
    ax.plot([1.55, 1.9], [0.12, 0.12], color="#111827",
            linewidth=1.0, linestyle=(0, (4, 2)))
    ax.text(1.98, 0.12, "only when a trigger is confirmed and not assigned to silent control",
            va="center", fontsize=FS)

    fig.savefig(FIG_DIR / "architecture.pdf")
    fig.savefig(FIG_DIR / "architecture.png", dpi=200)
    plt.close(fig)
    print("  Saved: architecture.pdf/.png")


# ─── Figure 3.2: sequence of one agent turn ─────────────────────────────────

def timeline():
    W, H = 6.3, 7.0
    fig, ax = canvas(W, H)
    lanes = {"agent": 0.5, "orch": 1.45, "judge": 2.55,
             "trigger": 3.65, "mod": 4.75, "next": 5.75}
    names = {"agent": "Agent\n(speaking)", "orch": "Orchestrator", "judge": "LLM judge",
             "trigger": "Trigger\ncheck", "mod": "Moderator", "next": "Next\nagent"}
    top = H - 0.1
    for k, x in lanes.items():
        box(ax, x - 0.45, top - 0.45, 0.9, 0.45,
            names[k], fill="#eff6ff", edge="#1d4ed8")

    def msg(y, a, b, label, dashed=False, ha="left", lx=None):
        xa, xb = lanes[a], lanes[b]
        ax.add_patch(FancyArrowPatch((xa, y), (xb, y), arrowstyle="-|>", mutation_scale=9,
                                     linewidth=1.0, color="#111827",
                                     linestyle=(0, (4, 2)) if dashed else "-", shrinkA=0, shrinkB=0))
        x = min(xa, xb) + 0.05 if lx is None else lx
        ax.text(x, y + 0.04, label, ha=ha, va="bottom", fontsize=FS, linespacing=1.15,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.3))

    def frame(y_top, y_bot, x0, x1, guard):
        ax.add_patch(Rectangle((x0, y_bot), x1 - x0, y_top - y_bot, fill=False,
                               edgecolor="#1d4ed8", linewidth=0.9))
        ax.text(x0 + 0.05, y_top - 0.04, "alt", ha="left", va="top", fontsize=FS,
                fontweight="bold", color="#1d4ed8")
        ax.text(x0 + 0.3, y_top - 0.04, guard, ha="left",
                va="top", fontsize=FS, color="#1d4ed8")

    step = 0.36
    y = top - 1.0
    msg(y, "orch", "agent", "context: persona, framing, top-3 passages,\ntranscript, latest intervention",
        lx=0.55)
    y -= step + 0.05
    msg(y, "agent", "orch", "turn")
    y -= step
    msg(y, "orch", "judge", "turn + up to 12 preceding turns")
    y -= step
    msg(y, "judge", "orch", "7 scores + justifications", dashed=True)
    y -= step
    msg(y, "orch", "trigger", "scores (from round 2)")

    alt_top = y - 0.12
    y = alt_top - 0.42
    msg(y, "trigger", "judge", "Stage 2: confirm?")
    y -= step
    msg(y, "judge", "trigger", "score 1\u20133 = confirmed", dashed=True)

    inner_top = y - 0.12
    y = inner_top - 0.42
    msg(y, "trigger", "orch", "intervene", lx=2.6)
    y -= step
    msg(y, "orch", "mod", "strategy prompt + trigger context + transcript")
    y -= step
    msg(y, "mod", "orch", "intervention (JSON)", dashed=True, lx=2.6)
    sep = y - 0.12
    y = sep - 0.42
    msg(y, "trigger", "orch", "logged only, no message", lx=2.6)
    inner_bot = y - 0.12
    alt_bot = inner_bot - 0.1

    frame(alt_top, alt_bot, 1.15, 6.25,
          "[Stage-1 rule met and fewer than 3 interventions so far]")
    frame(inner_top, inner_bot, 1.25, 6.15,
          "[confirmed, not assigned to silent control (80%)]")
    ax.plot([1.25, 6.15], [sep, sep], color="#1d4ed8",
            linewidth=0.9, linestyle=(0, (4, 2)))
    ax.text(1.3, sep - 0.04, "[confirmed, silent control (20%)]", ha="left", va="top",
            fontsize=FS, color="#1d4ed8")

    y = alt_bot - 0.35
    msg(y, "orch", "next", "context incl. the latest intervention (kept until replaced)")
    bottom = y - 0.15
    for x in lanes.values():
        ax.plot([x, x], [bottom, top - 0.45], color="#94a3b8", linewidth=0.8,
                linestyle=(0, (3, 2)), zorder=0)

    ax.text(W / 2, bottom - 0.3, "Repeated for each of the 6 agents in each of the 10 rounds. Triggers are "
            "not checked in round 1,\nand at most one trigger fires per round.",
            ha="center", va="center", fontsize=FS)

    # Crop the unused canvas below the footnote while keeping 1 data unit = 1 inch.
    y_min = bottom - 0.55
    ax.set_ylim(y_min, H)
    fig.set_size_inches(W, H - y_min)
    fig.savefig(FIG_DIR / "timeline.pdf")
    fig.savefig(FIG_DIR / "timeline.png", dpi=200)
    plt.close(fig)
    print("  Saved: timeline.pdf/.png")


if __name__ == "__main__":
    architecture()
    timeline()
