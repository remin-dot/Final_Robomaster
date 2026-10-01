"""Round-2 route planning on the map saved at the end of round 1.

Pure Python (no robot / OpenCV) so it can be tested offline:

    python3 src/route_planner.py data/raw/run1/round1_targets.json

For every designated target we look for a *firing cell*: the target's own block
or one of the eight adjacent blocks (diagonal included) with no known wall in
between - the robot never shoots from more than one block away.
Cells the target was actually seen from in round 1 are preferred, because the
camera already proved it can see the target from there.  Targets are visited
greedily (nearest first by BFS path length) using only edges that round 1
found open; unknown edges are used only when no known route exists.
"""

import json
import math
from collections import deque

MOVES = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}  # N E S W


def _edge_key(a, b):
    return frozenset((tuple(a), tuple(b)))


def parse_edges(items):
    """JSON edge list -> set of frozenset edges (outer-wall entries skipped)."""
    out = set()
    for e in items or []:
        if len(e) == 2 and isinstance(e[0], list) and isinstance(e[1], list):
            out.add(_edge_key(e[0], e[1]))
    return out


class GridGraph:
    def __init__(self, nx, ny, walls, open_edges=None):
        self.nx, self.ny = nx, ny
        self.walls = set(walls)
        self.open = set(open_edges or [])

    def inside(self, c):
        return 0 <= c[0] < self.nx and 0 <= c[1] < self.ny

    def neighbours(self, cell, allow_unknown=False):
        for d, (dx, dy) in MOVES.items():
            nb = (cell[0] + dx, cell[1] + dy)
            if not self.inside(nb):
                continue
            e = _edge_key(cell, nb)
            if e in self.walls:
                continue
            if e in self.open or allow_unknown:
                yield d, nb

    def bfs(self, start, goals, allow_unknown=False):
        """Shortest cell path from start to any goal cell (or None)."""
        start = tuple(start)
        goals = {tuple(g) for g in goals}
        if start in goals:
            return [start]
        prev = {start: None}
        q = deque([start])
        while q:
            c = q.popleft()
            for _, nb in self.neighbours(c, allow_unknown):
                if nb in prev:
                    continue
                prev[nb] = c
                if nb in goals:
                    path = [nb]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    return path[::-1]
                q.append(nb)
        return None

    def line_of_sight(self, cell, target_xy_m, tile):
        """True when the segment cell-centre -> target crosses no known wall."""
        x0, y0 = (cell[0] + 0.5) * tile, (cell[1] + 0.5) * tile
        x1, y1 = target_xy_m
        steps = max(2, int(math.hypot(x1 - x0, y1 - y0) / (tile * 0.1)))
        cur = tuple(cell)
        for i in range(1, steps + 1):
            t = i / steps
            c = (int((x0 + (x1 - x0) * t) // tile), int((y0 + (y1 - y0) * t) // tile))
            if c == cur or not self.inside(c):
                continue
            # diagonal jump: require one of the two L-shaped routes to be clear
            if abs(c[0] - cur[0]) + abs(c[1] - cur[1]) == 2:
                mid_a, mid_b = (c[0], cur[1]), (cur[0], c[1])
                ok_a = _edge_key(cur, mid_a) not in self.walls and _edge_key(mid_a, c) not in self.walls
                ok_b = _edge_key(cur, mid_b) not in self.walls and _edge_key(mid_b, c) not in self.walls
                if not (ok_a or ok_b):
                    return False
            elif _edge_key(cur, c) in self.walls:
                return False
            cur = c
        return True


def aim_deg(cell, target_xy_m, tile):
    """Absolute map bearing (0 = north, 90 = east) from a cell centre to a point."""
    x0, y0 = (cell[0] + 0.5) * tile, (cell[1] + 0.5) * tile
    return math.degrees(math.atan2(target_xy_m[0] - x0, target_xy_m[1] - y0)) % 360


# shooting rule (settings.yaml shooting.reach_pattern):
#   "3x3"   - the card is in the robot's block or any of the 8 blocks around it (diagonals
#             too): the robot stands in the middle of a 3 x 3 square, the card inside it
#   "cross" - the old rule: own block or the next block straight left / right / ahead /
#             behind, on the robot's row / column line
# Always: no known wall between, and not seen too edge-on (SHOT_MAX_VIEW_DEG).
REACH_CELLS = 1
REACH_PATTERN = "3x3"


def target_cell(graph, target_xy_m, tile):
    """Block a card position (m) is in, clamped to the maze."""
    return (min(max(int(target_xy_m[0] // tile), 0), graph.nx - 1),
            min(max(int(target_xy_m[1] // tile), 0), graph.ny - 1))


# a card in the next block is shot from here only when it sits straight ahead in that
# block (left / right / front / back): within this far of the robot's row / column line,
# and seen at most this slanted. Anything else is a hard angle -> from inside its block.
STRAIGHT_BAND_M = 0.2
SHOT_MAX_VIEW_DEG = 55.0


def straight_shot(graph, cell, target_xy_m, tile, step):
    """The card in the neighbour block `step` away is straight in line with this block and
    not seen too slanted (last run shot cards 0.25 m off the line and 71 deg edge-on)."""
    cx, cy = (cell[0] + 0.5) * tile, (cell[1] + 0.5) * tile
    off = abs(target_xy_m[1] - cy) if step[0] else abs(target_xy_m[0] - cx)
    if off > STRAIGHT_BAND_M:
        return False
    view = cell_view_deg(cell, target_xy_m, tile, graph.nx, graph.ny, open_test(graph.open))
    return view <= SHOT_MAX_VIEW_DEG


def within_reach(graph, cell, target_xy_m, tile, reach=None):
    """True when a card at target_xy_m may be shot from `cell` (see REACH_PATTERN): its block
    is within `reach` blocks in both directions (3x3 around the robot for reach 1) - or, with
    the old "cross" pattern, straight in line - with no known wall between and not seen too
    edge-on."""
    reach = REACH_CELLS if reach is None else reach
    cell = tuple(cell)
    tc = target_cell(graph, target_xy_m, tile)
    dx, dy = tc[0] - cell[0], tc[1] - cell[1]
    if REACH_PATTERN == "cross":
        if dx and dy:                  # diagonal: that is across a block
            return False
        if abs(dx) + abs(dy) > reach:
            return False
        if (dx or dy) and not straight_shot(graph, cell, target_xy_m, tile, (dx, dy)):
            return False               # off to the side / too slanted: shoot it from inside its own block
    else:
        if max(abs(dx), abs(dy)) > reach:
            return False               # outside the 3 x 3 square around the robot
        if (dx or dy) and cell_view_deg(cell, target_xy_m, tile, graph.nx, graph.ny,
                                        open_test(graph.open)) > SHOT_MAX_VIEW_DEG:
            return False               # nearly edge-on from here: a better spot will be found
    if graph.line_of_sight(cell, target_xy_m, tile):
        return True
    # a card on the face of the wall between the blocks is often mapped a few cm
    # beyond it: pulled 8 cm towards the robot it must then be in plain view
    cx, cy = (cell[0] + 0.5) * tile, (cell[1] + 0.5) * tile
    d = math.hypot(target_xy_m[0] - cx, target_xy_m[1] - cy)
    if d < 1e-6:
        return True
    k = max(0.0, d - 0.08) / d
    return graph.line_of_sight(cell, (cx + (target_xy_m[0] - cx) * k, cy + (target_xy_m[1] - cy) * k), tile)


RECT_FAMILY = ("square", "rect_wide", "rect_tall")
FRONTAL_DEG = 35.0      # a view closer than this to face-on shows the true shape
MAX_VIEW_DEG = 65.0     # beyond this the card is almost edge-on: no firing spot


def card_normal(target_xy_m, tile, nx=None, ny=None, max_off=None, is_open=None):
    """Unit normal of the wall a card hangs on: the nearest edge of its block that is
    not a known-open passage (cards hang on walls). None when no such edge is close,
    or when two walls are about as close (a card near a corner: cannot tell which).
    is_open(cell, dir) -> True for a passage the robot knows is open."""
    x, y = target_xy_m
    cx, cy = int(x // tile), int(y // tile)
    if nx is not None:
        cx = min(max(cx, 0), nx - 1)
        cy = min(max(cy, 0), ny - 1)
    # dir: 0 N, 1 E, 2 S, 3 W (as MOVES); normal points into the block
    edges = [(abs((cy + 1) * tile - y), 0, (0.0, -1.0)), (abs((cx + 1) * tile - x), 1, (-1.0, 0.0)),
             (abs(y - cy * tile), 2, (0.0, 1.0)), (abs(x - cx * tile), 3, (1.0, 0.0))]
    if is_open is not None:
        edges = [e for e in edges if not is_open((cx, cy), e[1])]
    if not edges:
        return None
    edges.sort()
    lim = tile * 0.3 if max_off is None else max_off
    if edges[0][0] > lim:
        return None
    if len(edges) > 1 and edges[1][0] <= lim and edges[1][0] - edges[0][0] < 0.08:
        return None
    return edges[0][2]


def view_angle_deg(from_xy_m, target_xy_m, normal):
    """Angle between the line of sight and the card's face normal (0 = face-on).
    The normal's sign does not matter (a card near a block edge may be mapped on
    either side of it)."""
    if normal is None:
        return 0.0
    vx, vy = from_xy_m[0] - target_xy_m[0], from_xy_m[1] - target_xy_m[1]
    d = math.hypot(vx, vy)
    if d < 1e-6:
        return 0.0
    c = abs(vx * normal[0] + vy * normal[1]) / d
    return math.degrees(math.acos(min(1.0, c)))


def cell_view_deg(cell, target_xy_m, tile, nx=None, ny=None, is_open=None):
    """View angle of a card from a block centre (0 when the card's wall is unknown)."""
    return view_angle_deg(((cell[0] + 0.5) * tile, (cell[1] + 0.5) * tile), target_xy_m,
                          card_normal(target_xy_m, tile, nx, ny, is_open=is_open))


def open_test(open_edges):
    """is_open(cell, dir) for card_normal from a set of frozenset edges."""
    def is_open(cell, d):
        dx, dy = MOVES[d]
        return frozenset((tuple(cell), (cell[0] + dx, cell[1] + dy))) in open_edges
    return is_open


def firing_cells(graph, target, tile, max_shoot_m, exclude=(), margin=0.9, reach=None):
    """Candidate cells to shoot `target` from, best first: [(cell, dist, seen)].

    Only cells within the shooting rule (see within_reach: this block or one of
    the eight adjacent blocks in 3x3 mode, no wall between). The margin keeps the cell inside the range
    limit: the saved target position has some error."""
    txy = (target["x_m"], target["y_m"])
    seen = {tuple(v["cell"]): v["dist_m"] for v in target.get("views", [])}
    if target.get("seen_from"):
        seen.setdefault(tuple(target["seen_from"]), target.get("best_dist_m", 0))
    target_cell = tuple(target.get("cell") or
                        (min(max(int(txy[0] / tile), 0), graph.nx - 1),
                         min(max(int(txy[1] / tile), 0), graph.ny - 1)))
    out, edge_on = [], []
    for x in range(graph.nx):
        for y in range(graph.ny):
            c = (x, y)
            if c in exclude:
                continue
            d = math.hypot((x + 0.5) * tile - txy[0], (y + 0.5) * tile - txy[1])
            # A neighbouring firing cell that projects implausibly close is bad map
            # geometry. The target's OWN cell is different: same-cell shooting is legal,
            # and the close-look pitch/search exists specifically for a card on that cell's
            # wall. Excluding it made Round 2 give up without entering the target cell.
            if d > max_shoot_m * margin or (d < 0.25 * tile and c != target_cell):
                continue
            if not within_reach(graph, c, txy, tile, reach):
                continue
            # almost edge-on: the shape cannot be told and the pellet glances off
            if cell_view_deg(c, txy, tile, graph.nx, graph.ny, open_test(graph.open)) > MAX_VIEW_DEG:
                edge_on.append((c, d, c in seen))
                continue
            out.append((c, d, c in seen))
    # First use a clean view. After that fails, retain edge-on cells too because
    # the target's own cell may be the only way to expose a blind corner.
    out = (out + edge_on) if exclude else (out or edge_on)
    if exclude:
        # A neighbouring shot failed: enter the target's own block, look
        # down/straight at its wall, and make the reliable second attempt there.
        out.sort(key=lambda t: (t[0] != target_cell,
                                cell_view_deg(t[0], txy, tile, graph.nx, graph.ny,
                                              open_test(graph.open)) > FRONTAL_DEG,
                                not t[2], t[1]))
    else:
        # First attempt: face-on/proven view, then nearest.
        out.sort(key=lambda t: (cell_view_deg(t[0], txy, tile, graph.nx, graph.ny,
                                              open_test(graph.open)) > FRONTAL_DEG,
                                not t[2], t[1]))
    return out


def plan_leg(graph, start, target, tile, max_shoot_m, exclude=()):
    """Path from start to the best firing cell of one target.

    Returns dict(path, fire_cell, aim_deg, dist_m, known_route) or None."""
    cands = firing_cells(graph, target, tile, max_shoot_m, exclude)
    if not cands:
        return None
    txy = (target["x_m"], target["y_m"])
    for allow_unknown in (False, True):
        best = None
        for seen_only in (True, False):
            goals = [c for c, _, s in cands if s or not seen_only]
            if not goals:
                continue
            path = graph.bfs(start, goals, allow_unknown)
            # a proven viewpoint is worth up to 2 extra moves
            if path and (best is None or len(path) + (0 if seen_only else 2) < len(best)):
                best = path
        if best:
            fc = best[-1]
            return {
                "color": target["color"],
                "id": target.get("id", target["color"]),
                "kind": target.get("kind"),
                "path": best,
                "fire_cell": fc,
                "aim_deg": aim_deg(fc, txy, tile),
                "dist_m": math.hypot((fc[0] + 0.5) * tile - txy[0], (fc[1] + 0.5) * tile - txy[1]),
                "known_route": not allow_unknown,
                "alternatives": [c for c, _, _ in cands if c != fc][:3],
            }
    return None


def driven_edges(data):
    """Edges the robot drove through in that round: they can never be walls."""
    path = [tuple(p) for p in data.get("path", [])]
    return {_edge_key(a, b) for a, b in zip(path, path[1:]) if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1}


def target_kind(t):
    return t.get("kind") or f"{t['color']} {t['shape']}"


def plan_round(data, start, kinds, max_shoot_m):
    """Greedy nearest-target ordering over the cards whose kind ("blue circle", ...)
    is in `kinds`. Returns list of legs (see plan_leg)."""
    nx, ny = data["grid_size"]
    tile = data["tile_m"]
    driven = driven_edges(data)
    graph = GridGraph(nx, ny, parse_edges(data.get("walls")) - driven,
                      parse_edges(data.get("open_edges")) | driven)
    kinds = set(kinds)
    todo = [t for t in data.get("targets", []) if target_kind(t) in kinds]
    legs, cur = [], tuple(start)
    while todo:
        options = [(plan_leg(graph, cur, t, tile, max_shoot_m), t) for t in todo]
        options = [(leg, t) for leg, t in options if leg]
        if not options:
            break
        leg, t = min(options, key=lambda o: len(o[0]["path"]))
        legs.append(leg)
        todo.remove(t)
        cur = leg["fire_cell"]
    return legs, graph


if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)
    legs, _ = plan_round(data, data["start"], {target_kind(t) for t in data["targets"]},
                         2 * data["tile_m"])
    for leg in legs:
        print(f"{leg['color']:<6} fire from {leg['fire_cell']} aim {leg['aim_deg']:.0f} deg "
              f"({leg['dist_m']:.2f} m)  path {leg['path']}")


# =============================================================================
# round 2: several route algorithms, the cheapest route wins
# =============================================================================
import heapq
import itertools
import time as _time

# time model (seconds) used to score every route the same way
COST = {"move_s": 2.6,       # drive one cell (turn to face it is extra)
        "turn90_s": 1.2,     # turn 90 deg in place
        "turn180_s": 2.2,    # U-turn
        "aim_s": 3.0,        # aim + fire at one target from a stop
        "stop_s": 1.5,       # each stop: brake, settle, start again
        "unseen_s": 1.5,     # firing spot the target was never seen from (line of sight unproven)
        "unknown_edge_s": 2.0,  # passage round 1 never saw open (may be a wall)
        "oblique_s": 6.0}    # firing spot that sees the card at a slant (> 35 deg off face-on)


def turn_cost(h, d, cost=COST):
    diff = (d - h) % 4
    return 0.0 if diff == 0 else (cost["turn180_s"] if diff == 2 else cost["turn90_s"])


class RoutePlanner:
    """Plans the round-2 route: which cells to stop at and in which order, so every
    target is shot with the least time. Several algorithms run on the same problem;
    `best()` scores them with one time model and returns the cheapest."""

    def __init__(self, graph, targets, tile, max_shoot_m, exclude=None, cost=None):
        self.g = graph
        self.targets = list(targets)
        self.tile = tile
        self.cost = dict(COST, **(cost or {}))
        exclude = exclude or {}
        # which cells can shoot which targets (+ penalty when that view is unproven)
        self.cover = {}                  # cell -> set(target index)
        self.penalty = {}                # (cell, i) -> seconds
        for i, t in enumerate(self.targets):
            txy = (t["x_m"], t["y_m"])
            for c, _, seen in firing_cells(graph, t, tile, max_shoot_m, exclude.get(t.get("id"), ())):
                self.cover.setdefault(c, set()).add(i)
                pen = 0.0 if seen else self.cost["unseen_s"]
                if cell_view_deg(c, txy, tile, graph.nx, graph.ny, open_test(graph.open)) > FRONTAL_DEG:
                    pen += self.cost["oblique_s"]      # slanted view: shape unsure, may need a re-aim
                if pen:
                    self.penalty[(c, i)] = pen
        self.reachable_targets = set().union(*self.cover.values()) if self.cover else set()
        self._dist_cache = {}

    # ---------------------------------------------------------------- graph helpers
    def _edges(self, cell):
        """(dir, neighbour, extra seconds) - known-open passages first, unknown ones cost extra."""
        for d, nb in self.g.neighbours(cell, allow_unknown=True):
            known = _edge_key(cell, nb) in self.g.open
            yield d, nb, 0.0 if known else self.cost["unknown_edge_s"]

    def path(self, a, b):
        """Fewest-moves path a -> b (BFS, known passages preferred on ties)."""
        a, b = tuple(a), tuple(b)
        if a == b:
            return [a]
        key = ("path", a, b)
        if key not in self._dist_cache:
            prev = {a: None}
            q = [(0.0, 0, a)]
            n = 0
            best = {a: 0.0}
            while q:
                c_cost, _, c = heapq.heappop(q)
                if c == b:
                    break
                if c_cost > best.get(c, 1e9):
                    continue
                for _, nb, extra in self._edges(c):
                    nc = c_cost + 1.0 + extra / 10.0    # moves first, unknown passages as tie-break
                    if nc < best.get(nb, 1e9):
                        best[nb] = nc
                        prev[nb] = c
                        n += 1
                        heapq.heappush(q, (nc, n, nb))
            if b not in prev:
                self._dist_cache[key] = None
            else:
                p = [b]
                while prev[p[-1]] is not None:
                    p.append(prev[p[-1]])
                self._dist_cache[key] = p[::-1]
        return self._dist_cache[key]

    # ---------------------------------------------------------------- scoring
    def leg(self, pos, h, goal):
        """Quickest drive pos -> goal from heading h, turns included (Dijkstra over cell+heading).
        Returns (seconds, path, final heading) or None."""
        key = (tuple(pos), h, tuple(goal))
        if key in self._dist_cache:
            return self._dist_cache[key]
        c = self.cost
        s0 = (tuple(pos), h)
        best, prev, heap, n, end = {s0: 0.0}, {s0: None}, [(0.0, 0, s0)], 0, None
        while heap:
            t, _, st = heapq.heappop(heap)
            if t > best.get(st, 1e18):
                continue
            if st[0] == tuple(goal):
                end = st
                break
            cell, hh = st
            for d, nb, extra in self._edges(cell):
                nt = t + turn_cost(hh, d, c) + c["move_s"] + extra
                s2 = (nb, d)
                if nt < best.get(s2, 1e18):
                    best[s2], prev[s2] = nt, st
                    n += 1
                    heapq.heappush(heap, (nt, n, s2))
        if end is None:
            self._dist_cache[key] = None
            return None
        chain = [end]
        while prev[chain[-1]] is not None:
            chain.append(prev[chain[-1]])
        chain.reverse()
        out = (best[end], [s[0] for s in chain], end[1])
        self._dist_cache[key] = out
        return out

    def evaluate(self, stops, start, heading):
        """Stops [(cell, [target idx])] in order -> time, moves, turns, full cell path.
        Every algorithm's stops are scored with this same model: one Dijkstra through the
        whole stop sequence over (stops done, cell, heading), so the direction the robot
        arrives in at a stop is chosen for the whole route, not leg by leg."""
        c = self.cost
        stops = [(tuple(cell), idx) for cell, idx in stops]
        if not stops:
            return {"time_s": 0.0, "moves": 0, "turns": 0, "route": [tuple(start)]}
        aim = [c["stop_s"] + sum(c["aim_s"] + self.penalty.get((cell, i), 0.0) for i in idx)
               for cell, idx in stops]
        s0 = (0, tuple(start), heading)
        best, prev, heap, n, end = {s0: 0.0}, {s0: None}, [(0.0, 0, s0)], 0, None
        while heap:
            t, _, st = heapq.heappop(heap)
            if t > best.get(st, 1e18):
                continue
            k, cell, h = st
            if k == len(stops):
                end = st
                break
            nxt = []
            if cell == stops[k][0]:
                nxt.append(((k + 1, cell, h), aim[k]))
            for d, nb, extra in self._edges(cell):
                nxt.append(((k, nb, d), turn_cost(h, d, c) + c["move_s"] + extra))
            for s2, dt in nxt:
                nt = t + dt
                if nt < best.get(s2, 1e18):
                    best[s2], prev[s2] = nt, st
                    n += 1
                    heapq.heappush(heap, (nt, n, s2))
        if end is None:
            return None
        chain = [end]
        while prev[chain[-1]] is not None:
            chain.append(prev[chain[-1]])
        chain.reverse()
        route, moves, turns = [chain[0][1]], 0, 0
        for a, b in zip(chain, chain[1:]):
            if b[1] != a[1]:
                route.append(b[1])
                moves += 1
                turns += (b[2] - a[2]) % 4 != 0
        return {"time_s": round(best[end], 1), "moves": moves, "turns": turns, "route": route}

    def restrict_to(self, start):
        """Only cells the robot can reach from start can be firing spots."""
        seen, q = {tuple(start)}, [tuple(start)]
        while q:
            c = q.pop()
            for _, nb, _ in self._edges(c):
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        self.cover = {c: v for c, v in self.cover.items() if c in seen}
        self.reachable_targets = set().union(*self.cover.values()) if self.cover else set()

    def _stops_from_order(self, cells):
        """Cells to stop at, in order -> each stop shoots every still-unshot target it covers."""
        done, stops = set(), []
        for cell in cells:
            new = sorted(self.cover.get(tuple(cell), set()) - done)
            if new:
                stops.append((tuple(cell), new))
                done |= set(new)
        return stops

    # ---------------------------------------------------------------- algorithms
    def greedy_nearest(self, start, heading):
        """Always go to the target whose nearest firing spot is closest."""
        pos, left, order = tuple(start), set(self.reachable_targets), []
        while left:
            best = None
            for c, idx in self.cover.items():
                if idx & left:
                    p = self.path(pos, c)
                    if p is not None and (best is None or len(p) < len(best[1])):
                        best = (c, p)
            if best is None:
                break
            order.append(best[0])
            left -= self.cover[best[0]]
            pos = best[0]
        return self._stops_from_order(order)

    def greedy_cover(self, start, heading):
        """Pick the stop that shoots the most targets per second of driving (set-cover greedy)."""
        c = self.cost
        pos, left, order = tuple(start), set(self.reachable_targets), []
        while left:
            best = None
            for cell, idx in self.cover.items():
                gain = len(idx & left)
                if not gain:
                    continue
                p = self.path(pos, cell)
                if p is None:
                    continue
                score = gain / ((len(p) - 1) * c["move_s"] + c["stop_s"] + c["aim_s"] * gain)
                if best is None or score > best[0]:
                    best = (score, cell)
            if best is None:
                break
            order.append(best[1])
            left -= self.cover[best[1]]
            pos = best[1]
        return self._stops_from_order(order)

    def dfs_branch_bound(self, start, heading, max_targets=7, time_limit_s=1.5):
        """Depth-first over target orders (nearest firing spot for each), pruning any
        partial route already slower than the best complete one."""
        idx = sorted(self.reachable_targets)
        if len(idx) > max_targets:
            return None
        best = {"time": float("inf"), "cells": None}
        t_end = _time.time() + time_limit_s

        def nearest_cell(pos, i):
            cands = [(len(p), c) for c, cov in self.cover.items() if i in cov
                     for p in [self.path(pos, c)] if p is not None]
            return min(cands)[1] if cands else None

        def dfs(pos, h, left, cells, spent):
            if _time.time() > t_end:
                return
            if spent >= best["time"]:
                return                                  # bound: already slower
            if not left:
                best["time"], best["cells"] = spent, list(cells)
                return
            for i in sorted(left, key=lambda i: len(self.path(pos, nearest_cell(pos, i)) or [])):
                c = nearest_cell(pos, i)
                if c is None:
                    continue
                lg = self.leg(pos, h, c)
                if lg is None:
                    continue
                shot = self.cover[c] & left
                aim = self.cost["stop_s"] + sum(self.cost["aim_s"] + self.penalty.get((c, i), 0.0) for i in shot)
                dfs(c, lg[2], left - shot, cells + [c], spent + lg[0] + aim)

        dfs(tuple(start), heading, set(idx), [], 0.0)
        return self._stops_from_order(best["cells"]) if best["cells"] else None

    def bfs_state(self, start, heading, max_targets=10):
        """Breadth-first search over (cell, targets shot): the fewest MOVES that shoots them all."""
        idx = sorted(self.reachable_targets)
        if not idx or len(idx) > max_targets:
            return None
        bit = {i: 1 << k for k, i in enumerate(idx)}
        full = (1 << len(idx)) - 1

        def shoot(cell, mask):
            for i in self.cover.get(cell, ()):
                mask |= bit.get(i, 0)
            return mask

        s0 = (tuple(start), shoot(tuple(start), 0))
        prev = {s0: None}
        q = [s0]
        head = 0
        goal = None
        while head < len(q):
            cell, mask = q[head]
            head += 1
            if mask == full:
                goal = (cell, mask)
                break
            for _, nb, _ in self._edges(cell):
                st = (nb, shoot(nb, mask))
                if st not in prev:
                    prev[st] = (cell, mask)
                    q.append(st)
        if goal is None:
            return None
        chain = [goal]
        while prev[chain[-1]] is not None:
            chain.append(prev[chain[-1]])
        chain.reverse()
        # stop where the shot set grows
        order = [s[0] for k, s in enumerate(chain) if k == 0 and s[1] or k > 0 and s[1] != chain[k - 1][1]]
        return self._stops_from_order(order)

    def dijkstra_turns(self, start, heading, max_targets=9):
        """Exact cheapest route under the time model: search over (cell, heading, targets
        shot) with move, turn and aim costs (Dijkstra)."""
        idx = sorted(self.reachable_targets)
        if not idx or len(idx) > max_targets:
            return None
        c = self.cost
        bit = {i: 1 << k for k, i in enumerate(idx)}
        full = (1 << len(idx)) - 1
        s0 = (tuple(start), heading, 0)
        best = {s0: 0.0}
        prev = {s0: None}
        heap = [(0.0, 0, s0)]
        n = 0
        goal = None
        while heap:
            t, _, st = heapq.heappop(heap)
            if t > best.get(st, 1e18):
                continue
            cell, h, mask = st
            if mask == full:
                goal = st
                break
            # shoot some of what this cell covers: any subset (a slanted card may be cheaper
            # to shoot face-on from a later stop), all of them when there are many
            new = [i for i in self.cover.get(cell, ()) if not mask & bit[i]]
            subsets = ([[i for k, i in enumerate(new) if sm >> k & 1] for sm in range(1, 1 << len(new))]
                       if len(new) <= 5 else [new])
            for sub in subsets:
                m2 = mask
                for i in sub:
                    m2 |= bit[i]
                nt = t + c["stop_s"] + sum(c["aim_s"] + self.penalty.get((cell, i), 0.0) for i in sub)
                s2 = (cell, h, m2)
                if nt < best.get(s2, 1e18):
                    best[s2], prev[s2] = nt, st
                    n += 1
                    heapq.heappush(heap, (nt, n, s2))
            for d, nb, extra in self._edges(cell):
                nt = t + turn_cost(h, d, c) + c["move_s"] + extra
                s2 = (nb, d, mask)
                if nt < best.get(s2, 1e18):
                    best[s2], prev[s2] = nt, st
                    n += 1
                    heapq.heappush(heap, (nt, n, s2))
        if goal is None:
            return None
        chain = [goal]
        while prev[chain[-1]] is not None:
            chain.append(prev[chain[-1]])
        chain.reverse()
        # the stops with exactly the targets chosen there (not "everything it covers")
        stops = []
        for k in range(1, len(chain)):
            gained = chain[k][2] & ~chain[k - 1][2]
            if gained:
                stops.append((chain[k][0], [i for i in idx if gained & bit[i]]))
        return stops

    ALGORITHMS = (("greedy nearest", "greedy_nearest"), ("greedy cover", "greedy_cover"),
                  ("DFS branch&bound", "dfs_branch_bound"), ("BFS state search", "bfs_state"),
                  ("Dijkstra (turns)", "dijkstra_turns"))

    def best(self, start, heading=0):
        """Run every algorithm, score each route with the same time model, keep the cheapest."""
        self.restrict_to(start)
        results = []
        for name, fn in self.ALGORITHMS:
            t0 = _time.time()
            try:
                stops = getattr(self, fn)(tuple(start), heading)
            except Exception as e:           # one algorithm failing never stops round 2
                stops = None
                print(f"[route] {name} failed: {e}")
            ms = (_time.time() - t0) * 1000
            if not stops:
                results.append({"algorithm": name, "ok": False, "ms": round(ms, 1)})
                continue
            ev = self.evaluate(stops, start, heading)
            if ev is None:
                results.append({"algorithm": name, "ok": False, "ms": round(ms, 1)})
                continue
            shot = set().union(*[set(i) for _, i in stops])
            results.append({"algorithm": name, "ok": True, "ms": round(ms, 1), "stops": stops,
                            "targets": len(shot), **ev})
        ok = [r for r in results if r["ok"]]
        if not ok:
            return {"algorithm": None, "stops": [], "comparison": results,
                    "unreachable": [t.get("id") for t in self.targets]}
        # most targets first, then least time, then fewest moves
        win = min(ok, key=lambda r: (-r["targets"], r["time_s"], r["moves"]))
        unreachable = [self.targets[i].get("id") for i in range(len(self.targets))
                       if i not in self.reachable_targets]
        return {"algorithm": win["algorithm"], "stops": win["stops"], "time_s": win["time_s"],
                "moves": win["moves"], "turns": win["turns"], "route": win["route"],
                "comparison": [{k: r[k] for k in ("algorithm", "ok", "ms", "time_s", "moves", "turns", "targets")
                                if k in r} for r in results],
                "unreachable": unreachable}


def plan_best(graph, targets, start, heading, tile, max_shoot_m, exclude=None, cost=None):
    return RoutePlanner(graph, targets, tile, max_shoot_m, exclude, cost).best(start, heading)
