# Скіли проєкту scalper-hft

Скіли — перевірені процедури для агентів (AI-асистентів) і розробників.

> **Єдине джерело істини — `.agents/skills/<name>/SKILL.md`** (нативний формат
> agent skills з frontmatter). Раніше тут були дублікати у `skills/*.md` —
> вони розходились з канонічними версіями і були видалені.

## Зміст

- [strategy-development](../.agents/skills/strategy-development/SKILL.md) — створення нової стратегії за стандартом проєкту
- [spec-driven-strategy](../.agents/skills/spec-driven-strategy/SKILL.md) — spec-before-code протокол для змін стратегій
- [backtest-run](../.agents/skills/backtest-run/SKILL.md) — запуск і інтерпретація бектесту
- [overfitting-audit](../.agents/skills/overfitting-audit/SKILL.md) — аудит стратегії на перенавчання (WF → DSR → CSCV/PBO → sensitivity)
- [data-management](../.agents/skills/data-management/SKILL.md) — завантаження та кешування даних
- [hft-risk-execution](../.agents/skills/hft-risk-execution/SKILL.md) — контроль ризиків, sizing та paper/live торгівля
