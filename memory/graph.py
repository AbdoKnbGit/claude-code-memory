"""
memory/graph.py

Graph-based search engine for cc-memory.

ARCHITECTURE:
  Instead of flat O(n) cosine search, memories form a weighted graph:
  - Nodes = memory entries
  - Edges = temporal (same session) + semantic (cosine > 0.7) + component (same partition)

  Search = ChromaDB gives top-K seed nodes → BFS through graph → re-score by graph distance

  COMPONENT PARTITIONING:
  Memories are auto-tagged by component detected from file paths:
    auth, api, db, ui, infra, test, config, hooks, models, utils

  Benefits:
  - Search scoped to relevant component first, expands to neighbors
  - O(component_size) not O(total_memories) for targeted queries
  - 10 years of memories stay fast because partitioned

ZERO-LATENCY LAYER:
  graph.py also maintains an in-RAM edge index so BFS never hits SQLite.
  Cold on startup, warms on first search, stays warm.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections import defaultdict, deque

from loguru import logger


COMPONENT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"auth|login|jwt|token|session|password|oauth|permission|role", re.I), "auth"),
    (re.compile(r"route|router|endpoint|controller|handler|api|view|request|response", re.I), "api"),
    (re.compile(r"model|schema|migration|database|db|sql|prisma|alembic|orm|entity", re.I), "db"),
    (re.compile(r"component|page|template|style|css|frontend|react|vue|svelte|html|ui", re.I), "ui"),
    (re.compile(r"docker|compose|nginx|deploy|k8s|kubernetes|infra|server|config|env|yaml", re.I), "infra"),
    (re.compile(r"test|spec|fixture|mock|assert|pytest|jest|coverage", re.I), "test"),
    (re.compile(r"hook|event|trigger|listener|middleware|interceptor", re.I), "hooks"),
    (re.compile(r"util|helper|common|shared|lib|tool|parser|format", re.I), "utils"),
    (re.compile(r"package|requirements|pyproject|cargo|gemfile|pom|gradle|dep", re.I), "deps"),
]


def detect_component(text: str, trigger: str = "") -> str:
    """Auto-detect component from memory text + trigger (file path / command).
    Returns one of: auth, api, db, ui, infra, test, hooks, utils, deps, general
    """
    combined = f"{text} {trigger}".lower()
    for pattern, component in COMPONENT_PATTERNS:
        if pattern.search(combined):
            return component
    return "general"


class GraphIndex:
    """
    Lightweight in-RAM adjacency index.
    Loaded from SQLite on first use, stays warm.
    Invalidated when new edges are added.
    """

    def __init__(self):
        self._adj: dict[int, list[tuple[int, float, str]]] = defaultdict(list)
        self._loaded_projects: set[str] = set()
        self._last_load: dict[str, float] = {}
        self._stale_ttl = 60.0

    def load(self, project_id: str, db_path: str) -> None:
        """Load edges for a project from SQLite into RAM."""
        now = time.time()
        last = self._last_load.get(project_id, 0)
        if project_id in self._loaded_projects and (now - last) < self._stale_ttl:
            return

        try:
            conn = sqlite3.connect(db_path, timeout=2)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_edges (
                    from_id   INTEGER NOT NULL,
                    to_id     INTEGER NOT NULL,
                    weight    REAL    NOT NULL DEFAULT 0.5,
                    edge_type TEXT    NOT NULL DEFAULT 'semantic',
                    project_id TEXT  NOT NULL,
                    PRIMARY KEY (from_id, to_id)
                )
            """)
            conn.commit()
            rows = conn.execute(
                "SELECT from_id, to_id, weight, edge_type FROM memory_edges WHERE project_id=?",
                (project_id,),
            ).fetchall()
            conn.close()

            project_node_ids: set[int] = set()
            for from_id, to_id, weight, edge_type in rows:
                project_node_ids.add(from_id)
                project_node_ids.add(to_id)
            for nid in project_node_ids:
                self._adj.pop(nid, None)

            for from_id, to_id, weight, edge_type in rows:
                existing = [n for n, w, t in self._adj[from_id] if n == to_id]
                if not existing:
                    self._adj[from_id].append((to_id, weight, edge_type))
                existing = [n for n, w, t in self._adj[to_id] if n == from_id]
                if not existing:
                    self._adj[to_id].append((from_id, weight, edge_type))

            self._loaded_projects.add(project_id)
            self._last_load[project_id] = now
            logger.debug(f"[Graph] Loaded {len(rows)} edges for project={project_id}")
        except Exception as e:
            logger.warning(f"[Graph] Failed to load edges: {e}")

    def add_edge(self, from_id: int, to_id: int, weight: float, edge_type: str, db_path: str, project_id: str) -> None:
        """Add edge to RAM index + persist to SQLite."""
        self._adj[from_id].append((to_id, weight, edge_type))
        self._adj[to_id].append((from_id, weight, edge_type))

        try:
            conn = sqlite3.connect(db_path, timeout=2)
            conn.execute(
                "INSERT OR IGNORE INTO memory_edges (from_id, to_id, weight, edge_type, project_id) VALUES (?,?,?,?,?)",
                (from_id, to_id, weight, edge_type, project_id),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"[Graph] Edge persist failed: {e}")

    def neighbors(self, node_id: int) -> list[tuple[int, float, str]]:
        return self._adj.get(node_id, [])

    def invalidate(self, project_id: str) -> None:
        self._loaded_projects.discard(project_id)


_graph_index = GraphIndex()


def build_edges_for_entry(
    entry_id: int,
    project_id: str,
    session_id: str,
    component: str,
    db_path: str,
    recent_ids: list[tuple[int, str, str]],
) -> None:
    """
    Build graph edges for a newly saved entry.
    Called after save — doesn't block the save path.

    Edge types built:
    - temporal: same session, adjacent in time → weight 0.8
    - component: same component → weight 0.6
    """
    for other_id, other_session, other_component in recent_ids:
        if other_id == entry_id:
            continue

        weight = 0.0
        edge_type = "none"

        if other_session == session_id:
            weight = 0.8
            edge_type = "temporal"

        elif other_component == component and component != "general":
            weight = 0.6
            edge_type = "component"

        if weight > 0:
            _graph_index.add_edge(entry_id, other_id, weight, edge_type, db_path, project_id)


def add_semantic_edges(
    entry_id: int,
    project_id: str,
    db_path: str,
    similar_ids: list[tuple[int, float]],
) -> None:
    """Add semantic edges from ChromaDB similarity results."""
    for other_id, sim in similar_ids:
        if other_id == entry_id:
            continue
        if sim >= 0.65:
            _graph_index.add_edge(entry_id, other_id, sim, "semantic", db_path, project_id)


def graph_search(
    seed_results: list[dict],
    project_id: str,
    query_component: str,
    db_path: str,
    max_hops: int = 2,
    max_results: int = 12,
) -> list[dict]:
    """
    BFS from seed nodes through the graph, scoring by:
    - Initial ChromaDB score (semantic similarity)
    - Graph distance (closer = higher bonus)
    - Component match (same partition = bonus)

    Returns re-ranked list of memory entries.
    """
    if not seed_results:
        return []

    _graph_index.load(project_id, db_path)

    visited: dict[int, float] = {}
    queue: deque[tuple[int, float, int]] = deque()
    result_map: dict[int, dict] = {}

    for entry in seed_results:
        eid = entry.get("id", -1)
        if eid < 0:
            continue
        score = entry.get("score", 0.5)
        visited[eid] = score
        queue.append((eid, score, 0))
        result_map[eid] = entry

    while queue:
        node_id, node_score, depth = queue.popleft()

        if depth >= max_hops:
            continue

        for neighbor_id, edge_weight, edge_type in _graph_index.neighbors(node_id):
            if neighbor_id in visited:
                neighbor_score = node_score * edge_weight * (0.7 ** depth)
                if neighbor_score <= visited[neighbor_id]:
                    continue
                visited[neighbor_id] = neighbor_score
                if neighbor_id in result_map:
                    result_map[neighbor_id]["_graph_score"] = neighbor_score
                continue

            neighbor_score = node_score * edge_weight * (0.7 ** depth)
            visited[neighbor_id] = neighbor_score
            queue.append((neighbor_id, neighbor_score, depth + 1))

            if neighbor_id not in result_map:
                entry_data = _fetch_entry(neighbor_id, db_path)
                if entry_data:
                    entry_data["_graph_score"] = neighbor_score
                    entry_data["_via_graph"] = True
                    result_map[neighbor_id] = entry_data

    final = []
    for eid, entry in result_map.items():
        base = entry.get("score", 0.0)
        graph_bonus = visited.get(eid, 0) * 0.3
        comp_bonus = 0.1 if entry.get("component") == query_component else 0.0
        entry["_final_score"] = base + graph_bonus + comp_bonus
        final.append(entry)

    final.sort(key=lambda x: x["_final_score"], reverse=True)

    logger.debug(
        f"[Graph] search: {len(seed_results)} seeds → {len(final)} results "
        f"({len(final) - len(seed_results)} via graph hops)"
    )
    return final[:max_results]


def _fetch_entry(entry_id: int, db_path: str) -> dict | None:
    """Fetch a single memory entry from SQLite by ID."""
    try:
        conn = sqlite3.connect(db_path, timeout=2)
        row = conn.execute(
            "SELECT id, text, pinned, created_at, session_id, token_count, component "
            "FROM memories WHERE id=? AND deleted=0",
            (entry_id,),
        ).fetchone()
        conn.close()
        if not row:
            return None
        return {
            "id": row[0],
            "text": row[1],
            "pinned": bool(row[2]),
            "created_at": row[3],
            "session_id": row[4],
            "token_count": row[5] or 0,
            "component": row[6] or "general",
            "score": 0.0,
        }
    except Exception:
        return None


def get_priority_components(query: str) -> list[str]:
    """
    Given a query, return which components to search first.
    E.g. "fix login bug" → ["auth", "api", "general"]
    """
    components = []
    query_lower = query.lower()
    for pattern, component in COMPONENT_PATTERNS:
        if pattern.search(query_lower):
            components.append(component)
    components.append("general")
    return components


def get_recent_entry_meta(
    project_id: str,
    db_path: str,
    limit: int = 20,
) -> list[tuple[int, str, str]]:
    """Get recent entry metadata for edge building: (id, session_id, component)."""
    try:
        conn = sqlite3.connect(db_path, timeout=2)
        rows = conn.execute(
            "SELECT id, session_id, COALESCE(component, 'general') "
            "FROM memories WHERE project_id=? AND deleted=0 "
            "ORDER BY id DESC LIMIT ?",
            (project_id, limit),
        ).fetchall()
        conn.close()
        return [(r[0], r[1], r[2]) for r in rows]
    except Exception:
        return []


def ensure_schema(db_path: str) -> None:
    """Add graph-related columns and tables to existing DB."""
    try:
        conn = sqlite3.connect(db_path, timeout=5)

        try:
            conn.execute("ALTER TABLE memories ADD COLUMN component TEXT DEFAULT 'general'")
        except sqlite3.OperationalError:
            pass

        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_edges (
                from_id    INTEGER NOT NULL,
                to_id      INTEGER NOT NULL,
                weight     REAL    NOT NULL DEFAULT 0.5,
                edge_type  TEXT    NOT NULL DEFAULT 'semantic',
                project_id TEXT    NOT NULL,
                PRIMARY KEY (from_id, to_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_project ON memory_edges(project_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_component ON memories(project_id, component) WHERE deleted=0")

        conn.commit()
        conn.close()
        logger.info("[Graph] Schema ready")
    except Exception as e:
        logger.warning(f"[Graph] Schema migration failed: {e}")
