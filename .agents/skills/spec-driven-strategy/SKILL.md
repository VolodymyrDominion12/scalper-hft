---
name: spec-driven-strategy
description: >-
  Протокол Spec-Driven Development для scalper-hft. Використовуй цей скіл
  при БУДЬ-яких змінах торгових стратегій: нова стратегія, зміна логіки,
  нові параметри, зміна family/preferred_regimes. Enforces "spec before code".
---

# Скіл: Spec-Driven Development (SDD) для стратегій

## Принцип
> **Spec before code** — специфікація пишеться ДО реалізації (або оновлюється
> перед будь-якими змінами логіки). Порушення → CI падає.

Структура SDD:
```
specs/strategies/
├── _schema.yaml         ← мета-схема (не редагуй без причини)
├── _validator.py        ← Python-валідатор (не редагуй без причини)
├── pairs_arb.yaml       ← spec стратегії
├── ...
tests/
└── test_strategy_specs.py  ← автоматичні тести зі spec
```

---

## Протокол: НОВА СТРАТЕГІЯ

### Крок 1 — Прочитай схему
```bash
cat specs/strategies/_schema.yaml
```
Зрозумій всі обов'язкові поля для відповідного `status`.

### Крок 2 — Напиши YAML-специфікацію

Створи `specs/strategies/<name>.yaml` ПЕРЕД написанням Python-коду.

**Мінімальний шаблон для нової стратегії:**
```yaml
name: my_strategy        # повинен збігатись із Strategy.name і назвою файлу
version: "1.0"
status: candidate        # нова стратегія = candidate

family: momentum         # momentum|mean_reversion|relative_value|carry|flow|market_making|ml|meta

hypothesis: >
  Чітке фальсифіковане твердження, ЧОМУ ця стратегія має edge.
  "X відбувається за умови Y, покриваючи витрати Z."

edge_conditions:
  - name: condition_1
    description: Кількісна умова перевірки edge (з порогом).
  - name: condition_2
    description: Друга умова.

params:
  param_name:
    default: 2.0
    min: 0.5
    max: 5.0
    step: 0.5
    description: Що контролює цей параметр.

invariants:
  no_lookahead: true      # ОБОВ'ЯЗКОВО true для будь-якої стратегії
  signal_values: [-1, 0, 1]

preferred_regimes: []    # порожній = всі режими; або [range, trend_up, ...]

notes: >
  Будь-які важливі деталі реалізації, посилання, застереження.
```

### Крок 3 — Валідуй spec
```bash
uv run python specs/strategies/_validator.py
# або тільки свою:
uv run pytest tests/test_strategy_specs.py -k my_strategy -v
```
Переконайся що spec проходить валідацію ДО написання коду.

### Крок 4 — Реалізуй стратегію

Дотримуйся скілу `strategy-development` (SKILL.md):
- Успадковуй `Strategy`
- Встанови `name`, `family`, `param_space`, `preferred_regimes`
- Забезпечь no-lookahead у `generate_signals()`
- Зареєструй у `REGISTRY`

**КРИТИЧНО:** `Strategy.family` і `Strategy.preferred_regimes` у класі Python
ПОВИННІ точно збігатись із `spec.family` і `spec.preferred_regimes`.

### Крок 5 — Запусти spec-тести

```bash
uv run pytest tests/test_strategy_specs.py -k my_strategy -v
```

Перевіряється:
1. ✅ Схема spec валідна
2. ✅ name є у REGISTRY
3. ✅ family збігається з класом
4. ✅ preferred_regimes збігається з класом
5. ✅ params є у param_space/__init__
6. ✅ generate_signals() повертає {-1, 0, 1}
7. ✅ Немає NaN у виході
8. ✅ No-lookahead (мутація майбутніх барів не змінює поточний сигнал)

### Крок 6 — Повні тести
```bash
uv run pytest tests/ -q
```

---

## Протокол: ЗМІНА ІСНУЮЧОЇ СТРАТЕГІЇ

При зміні будь-чого в стратегії:

| Що змінюється | Дія зі spec |
|---|---|
| Нові параметри | Додай у `spec.params` |
| Нові edge_conditions | Додай у `spec.edge_conditions` |
| Зміна family/preferred_regimes | Оновити обидва (spec + клас) одночасно |
| OOS audit пройшов | Змінити `status: candidate → validated`, додати `validation_summary` |
| OOS audit провалився | Змінити `status → rejected`, заповнити `rejection_reason` |
| Логіка signal змінилась | Оновити `hypothesis` і `version` (minor bump) |

---

## Протокол: REJECTED СТРАТЕГІЯ

Для rejected стратегії — мінімальна spec:
```yaml
name: my_strategy
version: "1.0"
status: rejected
family: momentum

rejection_reason: >
  Конкретна причина з числами (OOS Sharpe X, DSR=0, тощо).
  Без rejection_reason — spec не валідна.

invariants:
  no_lookahead: true
  signal_values: [-1, 0, 1]
```

---

## Правила валідності spec за статусом

| Status | Обов'язкові поля | Мін. edge_conditions |
|---|---|---|
| `validated` | name, version, status, family, hypothesis, edge_conditions, params, invariants | 3 |
| `candidate` | name, version, status, family, hypothesis, params, invariants | 2 |
| `pending` | name, version, status, family, hypothesis | 0 |
| `rejected` | name, version, status, family, rejection_reason | 0 |

---

## Перевірка покриття реєстру

Щоб переконатись що всі стратегії з REGISTRY мають spec:
```bash
uv run pytest tests/test_strategy_specs.py::test_all_registry_strategies_have_spec -v
```

При додаванні нової стратегії до REGISTRY — цей тест ПАДАТИМЕ
до створення відповідного YAML файлу. Це навмисно.

---

## Корисні команди

```bash
# Повний spec-runner
make spec-check

# Тільки YAML-валідація (швидко)
make spec-validate

# Конкретна стратегія
uv run pytest tests/test_strategy_specs.py -k pairs_arb -v

# Тільки behavioral assertions
uv run pytest tests/test_strategy_specs.py -k "signals or lookahead or nan" -v

# Покриття реєстру
uv run pytest tests/test_strategy_specs.py::test_all_registry_strategies_have_spec -v
```

---

## Чекліст завершення (нова стратегія)

- [ ] `specs/strategies/<name>.yaml` створено І валідує без помилок
- [ ] `spec.family` == `cls.family` у Python-класі
- [ ] `spec.preferred_regimes` == `cls.preferred_regimes` у Python-класі
- [ ] `spec.params` містить всі параметри з `param_space` і `__init__`
- [ ] `uv run pytest tests/test_strategy_specs.py -k <name> -v` — зелений
- [ ] `uv run pytest tests/ -q` — зелений
- [ ] Наступний крок: `backtest-run` → `overfitting-audit`
