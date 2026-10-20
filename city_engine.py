"""
city_engine.py — Motor de simulação para City Builder / Tycoon (camada Model).

Arquitetura:
    Model  (este arquivo)  -> estado puro + regras. Zero dependência de UI.
    Events (EventBus)      -> o Model publica eventos; a View apenas assina.
    Commands               -> a View/Input emite comandos; o Model valida e aplica.
    View   (Pygame/Arcade) -> lê snapshots imutáveis (CitySnapshot) e renderiza.

Rodar demo headless:  python city_engine.py
"""
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, ClassVar, Iterator

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Utilidades matemáticas
# ══════════════════════════════════════════════════════════════════════════════
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def sigmoid(x: float, k: float = 1.0, x0: float = 0.0) -> float:
    """Curva logística: saturação suave, evita respostas lineares."""
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


def ema(prev: float, new: float, alpha: float) -> float:
    """Média móvel exponencial: dá 'inércia' à opinião pública."""
    return prev + alpha * (new - prev)


# ══════════════════════════════════════════════════════════════════════════════
# Event Bus (Observer) — desacopla Model de View
# ══════════════════════════════════════════════════════════════════════════════
class EventType(Enum):
    TICK = auto()
    ZONE_BUILT = auto()
    ZONE_DEMOLISHED = auto()
    ZONE_ABANDONED = auto()
    BUDGET_CHANGED = auto()
    POLICY_CHANGED = auto()
    BANKRUPTCY_WARNING = auto()
    MILESTONE = auto()


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[EventType, list[Callable[[dict], None]]] = defaultdict(list)
        self._queue: deque[tuple[EventType, dict]] = deque()

    def subscribe(self, et: EventType, fn: Callable[[dict], None]) -> None:
        self._subs[et].append(fn)

    def emit(self, et: EventType, **payload) -> None:
        self._queue.append((et, payload))

    def flush(self) -> None:
        """Entregue em lote no fim do tick: evita reentrância durante a simulação."""
        while self._queue:
            et, p = self._queue.popleft()
            for fn in self._subs[et]:
                fn(p)


# ══════════════════════════════════════════════════════════════════════════════
# Zonas (herança + registro automático de tipos)
# ══════════════════════════════════════════════════════════════════════════════
class ZoneKind(Enum):
    EMPTY = 0
    RESIDENTIAL = 1
    COMMERCIAL = 2
    INDUSTRIAL = 3
    ROAD = 4
    POWER_PLANT = 5
    WATER_TOWER = 6
    SCHOOL = 7
    HOSPITAL = 8
    POLICE = 9
    PARK = 10


@dataclass(slots=True)
class TileContext:
    """Dados ambientais por tile, pré-computados pelo engine a cada tick."""
    road_access: bool
    powered: bool
    watered: bool
    land_value: float      # 0..1
    pollution: float       # 0..1
    service_coverage: float  # 0..1
    crime: float           # 0..1


class Zone(ABC):
    """Base de toda estrutura no grid."""
    REGISTRY: ClassVar[dict[ZoneKind, type["Zone"]]] = {}
    kind: ClassVar[ZoneKind] = ZoneKind.EMPTY
    build_cost: ClassVar[int] = 0
    upkeep: ClassVar[float] = 0.0          # custo mensal
    pollution_emit: ClassVar[float] = 0.0
    land_value_emit: ClassVar[float] = 0.0
    radius: ClassVar[int] = 0

    __slots__ = ("x", "y", "level", "occupancy", "abandoned_ticks")

    def __init_subclass__(cls, **kw) -> None:
        super().__init_subclass__(**kw)
        if cls.kind is not ZoneKind.EMPTY:
            Zone.REGISTRY[cls.kind] = cls

    def __init__(self, x: int, y: int) -> None:
        self.x, self.y = x, y
        self.level = 1
        self.occupancy = 0.0
        self.abandoned_ticks = 0

    @classmethod
    def create(cls, kind: ZoneKind, x: int, y: int) -> "Zone":
        return cls.REGISTRY[kind](x, y)

    @property
    def needs_utilities(self) -> bool:
        return True

    @abstractmethod
    def update(self, ctx: TileContext, city: "CityEngine") -> None: ...

    def is_operational(self, ctx: TileContext) -> bool:
        if not self.needs_utilities:
            return True
        return ctx.road_access and ctx.powered and ctx.watered


# ── Zonas privadas (RCI) ─────────────────────────────────────────────────────
class PrivateZone(Zone, ABC):
    """Zonas que crescem/encolhem organicamente com a demanda."""
    capacity_per_level: ClassVar[int] = 0
    max_level: ClassVar[int] = 3
    __slots__ = ()

    @property
    def capacity(self) -> int:
        return int(self.capacity_per_level * self.level ** 1.6)

    @abstractmethod
    def demand(self, city: "CityEngine") -> float: ...

    def desirability(self, ctx: TileContext) -> float:
        return clamp(0.5 * ctx.land_value + 0.3 * ctx.service_coverage
                     - 0.5 * ctx.pollution - 0.4 * ctx.crime + 0.3)

    def update(self, ctx: TileContext, city: "CityEngine") -> None:
        if not self.is_operational(ctx):
            self.occupancy *= 0.85      # sem infraestrutura → esvaziamento
            self.abandoned_ticks += 1
            if self.abandoned_ticks == 6:
                city.bus.emit(EventType.ZONE_ABANDONED, x=self.x, y=self.y)
            return
        self.abandoned_ticks = 0
        d = self.demand(city)           # -1..1
        want = clamp(self.desirability(ctx) * (0.5 + 0.5 * d))
        # Crescimento logístico: rápido no meio, lento perto da saturação.
        r = 0.25
        self.occupancy += r * self.occupancy * (1 - self.occupancy / max(want, 1e-3)) + 0.02 * d
        self.occupancy = clamp(self.occupancy)
        # Upgrade/downgrade de densidade
        if self.occupancy > 0.95 and ctx.land_value > 0.35 * self.level and self.level < self.max_level:
            self.level += 1
            self.occupancy *= 0.6
        elif self.occupancy < 0.15 and self.level > 1:
            self.level -= 1

    @property
    def filled(self) -> int:
        return int(self.capacity * self.occupancy)


class Residential(PrivateZone):
    kind = ZoneKind.RESIDENTIAL
    build_cost = 100
    capacity_per_level = 20
    __slots__ = ()

    def demand(self, city: "CityEngine") -> float:
        return city.stats.demand_r


class Commercial(PrivateZone):
    kind = ZoneKind.COMMERCIAL
    build_cost = 150
    capacity_per_level = 12
    land_value_emit = 0.05
    radius = 2
    __slots__ = ()

    def demand(self, city: "CityEngine") -> float:
        return city.stats.demand_c


class Industrial(PrivateZone):
    kind = ZoneKind.INDUSTRIAL
    build_cost = 200
    capacity_per_level = 25
    pollution_emit = 0.35
    radius = 4
    __slots__ = ()

    def demand(self, city: "CityEngine") -> float:
        return city.stats.demand_i

    def desirability(self, ctx: TileContext) -> float:
        # Indústria tolera poluição e prefere terra barata.
        return clamp(0.7 - 0.3 * ctx.land_value + 0.2 * ctx.service_coverage)


# ── Infraestrutura / serviços públicos ───────────────────────────────────────
class PublicZone(Zone):
    __slots__ = ()

    def update(self, ctx: TileContext, city: "CityEngine") -> None:
        # Eficiência cai com orçamento cortado (ver Budget.funding).
        self.occupancy = city.budget.funding_for(self.kind)


class Road(PublicZone):
    kind = ZoneKind.ROAD
    build_cost = 10
    upkeep = 1.0
    __slots__ = ()

    @property
    def needs_utilities(self) -> bool:
        return False


class PowerPlant(PublicZone):
    kind = ZoneKind.POWER_PLANT
    build_cost = 3000
    upkeep = 120.0
    pollution_emit = 0.5
    radius = 3
    supply: ClassVar[int] = 400        # tiles abastecidos
    __slots__ = ()

    @property
    def needs_utilities(self) -> bool:
        return False


class WaterTower(PublicZone):
    kind = ZoneKind.WATER_TOWER
    build_cost = 1500
    upkeep = 60.0
    supply: ClassVar[int] = 300
    __slots__ = ()

    @property
    def needs_utilities(self) -> bool:
        return False


class ServiceBuilding(PublicZone):
    coverage_type: ClassVar[str] = ""
    __slots__ = ()


class School(ServiceBuilding):
    kind = ZoneKind.SCHOOL
    build_cost = 800
    upkeep = 70.0
    radius = 8
    land_value_emit = 0.10
    coverage_type = "education"
    __slots__ = ()


class Hospital(ServiceBuilding):
    kind = ZoneKind.HOSPITAL
    build_cost = 1200
    upkeep = 100.0
    radius = 10
    land_value_emit = 0.08
    coverage_type = "health"
    __slots__ = ()


class Police(ServiceBuilding):
    kind = ZoneKind.POLICE
    build_cost = 900
    upkeep = 80.0
    radius = 9
    coverage_type = "safety"
    __slots__ = ()


class Park(ServiceBuilding):
    kind = ZoneKind.PARK
    build_cost = 200
    upkeep = 8.0
    radius = 3
    land_value_emit = 0.20
    pollution_emit = -0.15
    coverage_type = "leisure"
    __slots__ = ()


# ══════════════════════════════════════════════════════════════════════════════
# Políticas públicas (Strategy)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Policy:
    key: str
    name: str
    monthly_cost_per_capita: float = 0.0
    happiness: float = 0.0
    pollution_mult: float = 1.0
    crime_mult: float = 1.0
    demand_i: float = 0.0
    demand_c: float = 0.0
    tax_compliance: float = 0.0


POLICIES: dict[str, Policy] = {p.key: p for p in [
    Policy("green", "Lei Ambiental", 0.15, happiness=0.04, pollution_mult=0.6, demand_i=-0.15),
    Policy("free_transit", "Transporte Gratuito", 0.30, happiness=0.06, demand_c=0.10),
    Policy("curfew", "Toque de Recolher", 0.0, happiness=-0.08, crime_mult=0.6, demand_c=-0.10),
    Policy("tax_amnesty", "Anistia Fiscal", 0.0, happiness=0.03, tax_compliance=-0.10),
    Policy("industry_incentive", "Incentivo Industrial", 0.10, demand_i=0.20, pollution_mult=1.2),
]}


# ══════════════════════════════════════════════════════════════════════════════
# Orçamento
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Budget:
    treasury: float = 20_000.0
    tax_r: float = 0.09   # 0..0.30
    tax_c: float = 0.09
    tax_i: float = 0.09
    # % de financiamento por categoria (0.5..1.5). Corte economiza, mas degrada serviço.
    funding: dict[ZoneKind, float] = field(default_factory=lambda: defaultdict(lambda: 1.0))
    loan: float = 0.0
    loan_rate: float = 0.01  # ao mês
    history: deque = field(default_factory=lambda: deque(maxlen=120))

    def funding_for(self, kind: ZoneKind) -> float:
        return clamp(self.funding[kind], 0.0, 1.5)

    def avg_tax(self) -> float:
        return (self.tax_r + self.tax_c + self.tax_i) / 3


# ══════════════════════════════════════════════════════════════════════════════
# Estatísticas macro
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class CityStats:
    month: int = 0
    population: float = 0.0
    housing_capacity: int = 0
    jobs_total: int = 0
    jobs_filled: int = 0
    unemployment: float = 0.0
    happiness: float = 0.6
    demand_r: float = 0.5
    demand_c: float = 0.0
    demand_i: float = 0.3
    income: float = 0.0
    expenses: float = 0.0
    avg_pollution: float = 0.0
    avg_crime: float = 0.0
    migration: float = 0.0


@dataclass(frozen=True)
class CitySnapshot:
    """Cópia somente-leitura entregue à View (thread-safe por imutabilidade)."""
    stats: CityStats
    treasury: float
    kinds: np.ndarray
    levels: np.ndarray
    occupancy: np.ndarray
    pollution: np.ndarray
    land_value: np.ndarray


# ══════════════════════════════════════════════════════════════════════════════
# Comandos (Command pattern) — única porta de entrada para mutações do jogador
# ══════════════════════════════════════════════════════════════════════════════
class Command(ABC):
    @abstractmethod
    def execute(self, city: "CityEngine") -> bool: ...


@dataclass
class BuildCmd(Command):
    kind: ZoneKind
    x: int
    y: int

    def execute(self, city: "CityEngine") -> bool:
        return city.build(self.kind, self.x, self.y)


@dataclass
class BuildRectCmd(Command):
    kind: ZoneKind
    x0: int
    y0: int
    x1: int
    y1: int

    def execute(self, city: "CityEngine") -> bool:
        ok = False
        for y in range(min(self.y0, self.y1), max(self.y0, self.y1) + 1):
            for x in range(min(self.x0, self.x1), max(self.x0, self.x1) + 1):
                ok |= city.build(self.kind, x, y)
        return ok


@dataclass
class DemolishCmd(Command):
    x: int
    y: int

    def execute(self, city: "CityEngine") -> bool:
        return city.demolish(self.x, self.y)


@dataclass
class SetTaxCmd(Command):
    sector: str  # "r" | "c" | "i"
    rate: float

    def execute(self, city: "CityEngine") -> bool:
        setattr(city.budget, f"tax_{self.sector}", clamp(self.rate, 0.0, 0.30))
        city.bus.emit(EventType.BUDGET_CHANGED, sector=self.sector, rate=self.rate)
        return True


@dataclass
class TogglePolicyCmd(Command):
    key: str

    def execute(self, city: "CityEngine") -> bool:
        if self.key not in POLICIES:
            return False
        city.policies ^= {self.key}
        city.bus.emit(EventType.POLICY_CHANGED, key=self.key, active=self.key in city.policies)
        return True


# ══════════════════════════════════════════════════════════════════════════════
# CityEngine
# ══════════════════════════════════════════════════════════════════════════════
class CityEngine:
    # Constantes de balanceamento (expostas para tuning / data-driven)
    PER_CAPITA_INCOME = 18.0     # base tributável mensal por habitante empregado
    COMMERCE_REVENUE = 30.0      # por vaga comercial ocupada
    INDUSTRY_REVENUE = 22.0
    HAPPINESS_INERTIA = 0.15
    LAFFER_PEAK = 0.12           # alíquota "confortável"

    def __init__(self, width: int = 64, height: int = 64, seed: int | None = None) -> None:
        self.w, self.h = width, height
        self.rng = random.Random(seed)
        self.grid: list[list[Zone | None]] = [[None] * width for _ in range(height)]
        self.bus = EventBus()
        self.budget = Budget()
        self.stats = CityStats()
        self.policies: set[str] = set()
        self._cmd_queue: deque[Command] = deque()

        # Camadas escalares (numpy → difusão vetorizada, O(N) por tick)
        shape = (height, width)
        self.pollution = np.zeros(shape, np.float32)
        self.land_value = np.full(shape, 0.3, np.float32)
        self.crime = np.zeros(shape, np.float32)
        self.coverage: dict[str, np.ndarray] = {
            k: np.zeros(shape, np.float32) for k in ("education", "health", "safety", "leisure")
        }
        self.road_access = np.zeros(shape, bool)
        self.powered = np.zeros(shape, bool)
        self.watered = np.zeros(shape, bool)
        self._dirty_network = True

    # ── API de comandos ──────────────────────────────────────────────────────
    def submit(self, cmd: Command) -> None:
        self._cmd_queue.append(cmd)

    def _drain_commands(self) -> None:
        while self._cmd_queue:
            self._cmd_queue.popleft().execute(self)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h

    def build(self, kind: ZoneKind, x: int, y: int) -> bool:
        if not self.in_bounds(x, y) or self.grid[y][x] is not None:
            return False
        cls = Zone.REGISTRY[kind]
        if self.budget.treasury < cls.build_cost:
            return False
        self.budget.treasury -= cls.build_cost
        z = cls(x, y)
        if isinstance(z, PrivateZone):
            z.occupancy = 0.05
        self.grid[y][x] = z
        self._dirty_network = True
        self.bus.emit(EventType.ZONE_BUILT, kind=kind, x=x, y=y)
        return True

    def demolish(self, x: int, y: int) -> bool:
        if not self.in_bounds(x, y) or self.grid[y][x] is None:
            return False
        kind = self.grid[y][x].kind
        self.grid[y][x] = None
        self.budget.treasury -= 5
        self._dirty_network = True
        self.bus.emit(EventType.ZONE_DEMOLISHED, kind=kind, x=x, y=y)
        return True

    def zones(self) -> Iterator[Zone]:
        for row in self.grid:
            for z in row:
                if z is not None:
                    yield z

    # ── Camadas espaciais ────────────────────────────────────────────────────
    def _recompute_networks(self) -> None:
        """BFS na malha viária; utilidades propagam pela rede a partir das usinas."""
        if not self._dirty_network:
            return
        self.road_access[:] = False
        self.powered[:] = False
        self.watered[:] = False
        roads = np.zeros((self.h, self.w), bool)
        for z in self.zones():
            if z.kind is ZoneKind.ROAD:
                roads[z.y, z.x] = True
        # Acesso viário: vizinho ortogonal (ou o próprio) é estrada
        pad = np.pad(roads, 1)
        self.road_access = roads | pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]

        for src_kind, layer in ((ZoneKind.POWER_PLANT, self.powered), (ZoneKind.WATER_TOWER, self.watered)):
            for src in (z for z in self.zones() if z.kind is src_kind):
                budget = int(src.supply * self.budget.funding_for(src_kind))
                seen = {(src.x, src.y)}
                q = deque([(src.x, src.y)])
                while q and budget > 0:
                    x, y = q.popleft()
                    layer[y, x] = True
                    budget -= 1
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if self.in_bounds(nx, ny) and (nx, ny) not in seen and self.road_access[ny, nx]:
                            seen.add((nx, ny))
                            # Utilidade só "viaja" por estradas, mas abastece lotes adjacentes
                            if roads[ny, nx]:
                                q.append((nx, ny))
                            else:
                                layer[ny, nx] = True
        self._dirty_network = False

    @staticmethod
    def _blur(a: np.ndarray, iters: int) -> np.ndarray:
        """Difusão por kernel 3x3 — barato e dá gradientes orgânicos."""
        for _ in range(iters):
            p = np.pad(a, 1, mode="edge")
            a = (p[1:-1, 1:-1] * 4 + p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 8
        return a

    def _recompute_fields(self) -> None:
        src_pol = np.zeros_like(self.pollution)
        src_lv = np.zeros_like(self.land_value)
        cov_src = {k: np.zeros_like(v) for k, v in self.coverage.items()}
        pol_mult = math.prod(POLICIES[p].pollution_mult for p in self.policies)

        for z in self.zones():
            eff = z.occupancy if isinstance(z, PrivateZone) else self.budget.funding_for(z.kind)
            src_pol[z.y, z.x] += z.pollution_emit * (z.level if isinstance(z, PrivateZone) else 1) * pol_mult * max(eff, 0.2)
            src_lv[z.y, z.x] += z.land_value_emit
            if isinstance(z, ServiceBuilding):
                r = z.radius
                y0, y1, x0, x1 = max(0, z.y - r), min(self.h, z.y + r + 1), max(0, z.x - r), min(self.w, z.x + r + 1)
                yy, xx = np.ogrid[y0:y1, x0:x1]
                dist = np.sqrt((yy - z.y) ** 2 + (xx - z.x) ** 2)
                cov_src[z.coverage_type][y0:y1, x0:x1] += np.clip(1 - dist / r, 0, 1) * eff

        # Poluição tem memória (acumula e decai lentamente)
        self.pollution = np.clip(self.pollution * 0.7 + self._blur(src_pol, 3) * 2.0, 0, 1)
        for k in self.coverage:
            self.coverage[k] = np.clip(cov_src[k], 0, 1)

        safety = self.coverage["safety"]
        crime_mult = math.prod(POLICIES[p].crime_mult for p in self.policies)
        density = np.zeros_like(self.crime)
        for z in self.zones():
            if isinstance(z, PrivateZone):
                density[z.y, z.x] = z.level * z.occupancy / 3
        # Crime ∝ densidade × desemprego, reduzido por policiamento
        self.crime = np.clip(self._blur(density, 2) * (0.3 + 1.5 * self.stats.unemployment)
                             * (1 - 0.85 * safety) * crime_mult, 0, 1)

        services = (self.coverage["education"] + self.coverage["health"] + self.coverage["leisure"]) / 3
        target_lv = 0.25 + self._blur(src_lv, 3) * 3 + 0.35 * services - 0.6 * self.pollution - 0.3 * self.crime
        self.land_value = np.clip(self.land_value + 0.2 * (target_lv - self.land_value), 0, 1).astype(np.float32)

    def _ctx(self, z: Zone) -> TileContext:
        x, y = z.x, z.y
        cov = (self.coverage["education"][y, x] + self.coverage["health"][y, x]
               + self.coverage["safety"][y, x]) / 3
        return TileContext(bool(self.road_access[y, x]), bool(self.powered[y, x]), bool(self.watered[y, x]),
                           float(self.land_value[y, x]), float(self.pollution[y, x]), float(cov),
                           float(self.crime[y, x]))

    # ── Equações macro ───────────────────────────────────────────────────────
    def _tax_penalty(self, rate: float) -> float:
        """Penalidade convexa (quadrática acima do pico): incentiva equilíbrio."""
        excess = max(0.0, rate - self.LAFFER_PEAK)
        return 8.0 * excess ** 2 + 0.6 * excess

    def _update_demographics(self) -> None:
        s = self.stats
        res = [z for z in self.zones() if isinstance(z, Residential)]
        com = [z for z in self.zones() if isinstance(z, Commercial)]
        ind = [z for z in self.zones() if isinstance(z, Industrial)]

        s.housing_capacity = sum(z.capacity for z in res)
        housed = sum(z.filled for z in res)
        jobs_c = sum(z.filled for z in com)
        jobs_i = sum(z.filled for z in ind)
        s.jobs_total = jobs_c + jobs_i

        workforce = s.population * 0.6
        s.jobs_filled = int(min(workforce, s.jobs_total))
        s.unemployment = clamp(1 - s.jobs_filled / workforce) if workforce > 0 else 0.0

        # ── Felicidade (alvo instantâneo + inércia EMA) ─────────────────────
        avg_cov = {k: float(v.mean()) for k, v in self.coverage.items()}
        s.avg_pollution = float(self.pollution.mean())
        s.avg_crime = float(self.crime.mean())
        housing_pressure = s.population / max(s.housing_capacity, 1)
        target = (
            0.55
            - 0.9 * s.unemployment ** 1.5                     # desemprego dói de forma super-linear
            - self._tax_penalty(self.budget.tax_r)
            - 0.5 * self._tax_penalty(self.budget.tax_c)
            - 0.6 * s.avg_pollution
            - 0.5 * s.avg_crime
            - 0.4 * max(0.0, housing_pressure - 0.95)         # superlotação
            + 0.15 * sigmoid(avg_cov["health"], 10, 0.3)
            + 0.15 * sigmoid(avg_cov["education"], 10, 0.3)
            + 0.10 * avg_cov["leisure"]
            + sum(POLICIES[p].happiness for p in self.policies)
            - (0.15 if self.budget.treasury < 0 else 0.0)      # serviços em colapso
        )
        s.happiness = clamp(ema(s.happiness, clamp(target), self.HAPPINESS_INERTIA))

        # ── Migração: modelo atrativo-repulsivo ─────────────────────────────
        # Atração = f(felicidade) × f(vagas) × f(moradia livre); êxodo cresce com infelicidade.
        free_housing = max(0, s.housing_capacity - s.population)
        open_jobs = max(0, s.jobs_total - workforce)
        attract = (sigmoid(s.happiness, 10, 0.5)
                   * (0.4 + 0.6 * sigmoid(open_jobs / max(s.population, 50), 6, 0.1))
                   * free_housing * 0.25)
        exodus = s.population * (0.12 * (1 - sigmoid(s.happiness, 12, 0.35))
                                 + 0.05 * s.unemployment ** 2)
        homeless_exodus = max(0.0, s.population - s.housing_capacity) * 0.5   # sem teto → saem rápido
        births = s.population * 0.002 * s.happiness
        s.migration = attract - exodus - homeless_exodus + births
        s.population = max(0.0, s.population + s.migration)

        # ── Demanda RCI (feedback entre setores) ────────────────────────────
        pol_di = sum(POLICIES[p].demand_i for p in self.policies)
        pol_dc = sum(POLICIES[p].demand_c for p in self.policies)
        job_ratio = s.jobs_total / max(workforce, 1)
        s.demand_r = math.tanh(1.5 * (job_ratio - 0.9) + 2 * (s.happiness - 0.5)
                               - 4 * self._tax_penalty(self.budget.tax_r) + (1 if s.population < 50 else 0))
        # Comércio precisa de consumidores E de bens industriais
        consumers = s.population * (1 - s.unemployment)
        goods = jobs_i * 3
        s.demand_c = math.tanh(min(consumers / 4 - jobs_c, goods - jobs_c) / max(20, jobs_c + 1)
                               - 4 * self._tax_penalty(self.budget.tax_c) + pol_dc)
        s.demand_i = math.tanh((workforce - s.jobs_total) / max(30, s.jobs_total + 1)
                               + 0.3 - 4 * self._tax_penalty(self.budget.tax_i) + pol_di)

        # Ocupação residencial é limitada pela população real
        if s.housing_capacity:
            ratio = clamp(s.population / s.housing_capacity)
            for z in res:
                z.occupancy = ema(z.occupancy, ratio * min(1.0, z.occupancy / max(ratio, 1e-3) + 0.1), 0.3)
        _ = housed

    def _update_economy(self) -> None:
        s, b = self.stats, self.budget
        compliance = clamp(0.95 - 0.8 * max(0.0, b.avg_tax() - 0.15)
                           + sum(POLICIES[p].tax_compliance for p in self.policies)
                           + 0.1 * (s.happiness - 0.5))
        lv = float(self.land_value.mean())
        com_jobs = sum(z.filled for z in self.zones() if isinstance(z, Commercial))
        ind_jobs = sum(z.filled for z in self.zones() if isinstance(z, Industrial))

        income_r = s.jobs_filled * self.PER_CAPITA_INCOME * (0.6 + lv) * b.tax_r
        income_c = com_jobs * self.COMMERCE_REVENUE * (0.5 + s.happiness) * b.tax_c
        income_i = ind_jobs * self.INDUSTRY_REVENUE * b.tax_i
        s.income = (income_r + income_c + income_i) * compliance * 1.5

        upkeep = sum(z.upkeep * b.funding_for(z.kind) for z in self.zones())
        policy_cost = sum(POLICIES[p].monthly_cost_per_capita for p in self.policies) * s.population
        interest = b.loan * b.loan_rate
        # Custo de escala: cidades grandes ficam proporcionalmente mais caras
        scale = 1 + 0.15 * math.log1p(s.population / 1000)
        s.expenses = (upkeep + policy_cost) * scale + interest

        b.treasury += s.income - s.expenses
        b.history.append((s.month, s.income, s.expenses, b.treasury))
        if b.treasury < 0:
            self.bus.emit(EventType.BANKRUPTCY_WARNING, treasury=b.treasury)

    # ── Loop principal ───────────────────────────────────────────────────────
    def tick(self) -> None:
        """Um 'mês'. Pipeline determinístico: input → espaço → zonas → macro → economia → eventos."""
        self._drain_commands()
        self._recompute_networks()
        self._recompute_fields()
        for z in self.zones():
            z.update(self._ctx(z), self)
        self._update_demographics()
        self._update_economy()
        self.stats.month += 1
        for m in (100, 1_000, 10_000, 100_000):
            if self.stats.population >= m > self.stats.population - self.stats.migration:
                self.bus.emit(EventType.MILESTONE, population=m)
        self.bus.emit(EventType.TICK, month=self.stats.month)
        self.bus.flush()

    def snapshot(self) -> CitySnapshot:
        kinds = np.zeros((self.h, self.w), np.int8)
        levels = np.zeros_like(kinds)
        occ = np.zeros((self.h, self.w), np.float32)
        for z in self.zones():
            kinds[z.y, z.x] = z.kind.value
            levels[z.y, z.x] = z.level
            occ[z.y, z.x] = z.occupancy
        from copy import copy
        return CitySnapshot(copy(self.stats), self.budget.treasury, kinds, levels, occ,
                            self.pollution.copy(), self.land_value.copy())


# ══════════════════════════════════════════════════════════════════════════════
# Relógio desacoplado (fixed timestep) — usado por qualquer front-end
# ══════════════════════════════════════════════════════════════════════════════
class SimulationClock:
    SPEEDS = {0: 0.0, 1: 2.0, 2: 1.0, 3: 0.35}  # segundos reais por mês

    def __init__(self, engine: CityEngine) -> None:
        self.engine, self.speed, self._acc = engine, 1, 0.0

    def advance(self, dt: float) -> int:
        """Chamar a cada frame com dt real; executa 0..N ticks. Retorna ticks rodados."""
        period = self.SPEEDS[self.speed]
        if period == 0:
            return 0
        self._acc += dt
        n = 0
        while self._acc >= period and n < 5:   # anti spiral-of-death
            self.engine.tick()
            self._acc -= period
            n += 1
        return n


# ══════════════════════════════════════════════════════════════════════════════
# Demo headless
# ══════════════════════════════════════════════════════════════════════════════
def _demo() -> None:
    city = CityEngine(48, 48, seed=42)
    city.bus.subscribe(EventType.MILESTONE, lambda p: print(f"  🎉 Marco: {p['population']} habitantes"))
    city.bus.subscribe(EventType.BANKRUPTCY_WARNING, lambda p: None)
    Z = ZoneKind
    for cmd in [
        BuildRectCmd(Z.ROAD, 5, 20, 40, 20),
        BuildRectCmd(Z.ROAD, 22, 5, 22, 40),
        BuildCmd(Z.POWER_PLANT, 6, 21),
        BuildCmd(Z.WATER_TOWER, 7, 21),
        BuildRectCmd(Z.RESIDENTIAL, 10, 17, 20, 19),
        BuildRectCmd(Z.RESIDENTIAL, 10, 21, 20, 23),
        BuildRectCmd(Z.COMMERCIAL, 23, 17, 30, 19),
        BuildRectCmd(Z.INDUSTRIAL, 32, 21, 40, 23),
        BuildCmd(Z.SCHOOL, 21, 15),
        BuildCmd(Z.PARK, 15, 24),
    ]:
        city.submit(cmd)

    print(f"{'Mês':>4} {'Pop':>7} {'Casa':>6} {'Vagas':>6} {'Desemp':>7} {'Feliz':>6} "
          f"{'R':>5} {'C':>5} {'I':>5} {'Caixa':>9}")
    for m in range(60):
        if m == 30:
            print("  >> Prefeito sobe imposto residencial para 22% e ativa Lei Ambiental")
            city.submit(SetTaxCmd("r", 0.22))
            city.submit(TogglePolicyCmd("green"))
        city.tick()
        s = city.stats
        if m % 3 == 0:
            print(f"{s.month:>4} {s.population:>7.0f} {s.housing_capacity:>6} {s.jobs_total:>6} "
                  f"{s.unemployment:>6.0%} {s.happiness:>6.2f} {s.demand_r:>5.2f} {s.demand_c:>5.2f} "
                  f"{s.demand_i:>5.2f} {city.budget.treasury:>9.0f}")


if __name__ == "__main__":
    _demo()
