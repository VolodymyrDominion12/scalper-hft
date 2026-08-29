# MCP-конфігурація scalper-hft

Model Context Protocol-сервери для AI-асистентів (Claude Code / Cursor / інші).
Вказуй шляхи відносно кореня проєкту або абсолютні.

## Сервери

### 1. Filesystem (обов'язковий)
Доступ до коду, даних і звітів проєкту.
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/volodymyr/PycharmProjects/scalper-hft"]
    }
  }
}
```

### 2. Binance (ринкові дані/виконання, опційно)
```json
{
  "mcpServers": {
    "binance": {
      "command": "npx",
      "args": ["-y", "@binance/mcp-server"],
      "env": {
        "BINANCE_API_KEY": "${BINANCE_API_KEY}",
        "BINANCE_API_SECRET": "${BINANCE_API_SECRET}"
      }
    }
  }
}
```
> Ключі підставляються з `.env` проєкту. Якщо сервер не встановлений — використовуй
> власний `scalper_hft.cli download` (безкоштовно, без зовнішніх залежностей).

### 3. Python-виконання (аналіз даних у сесії асистента, опційно)
```json
{
  "mcpServers": {
    "python": {
      "command": "uv",
      "args": ["run", "--project", "/home/volodymyr/PycharmProjects/scalper-hft", "-m", "mcp_simple_python"]
    }
  }
}
```

### 4. scalper-hft (власний сервер трейдінгу — рекомендовано)
Власні інструменти проєкту: бектест, cohort/stress/capacity аналіз, список
стратегій, стан кешу даних, один крок paper-торгівлі. Без зовнішніх залежностей
(чистий stdlib, stdio-транспорт). Ключі НІКОЛИ не повертаються.
```json
{
  "mcpServers": {
    "scalper-hft": {
      "command": "/home/volodymyr/PycharmProjects/scalper-hft/.venv/bin/python",
      "args": ["-m", "scalper_hft.mcp_trading"]
    }
  }
}
```
> Або через CLI: `args: ["-m", "scalper_hft.cli", "mcp"]`.

**Інструменти сервера `scalper-hft`:**
| Інструмент | Що робить |
|---|---|
| `strategy_list` | Список стратегій з реєстру + параметри |
| `settings_summary` | Налаштування БЕЗ секретів (dry_run, комісії, ліміти) |
| `market_status` | Розмір кешу даних по символах (klines/aggTrades/funding) |
| `run_backtest` | Бектест стратегії: strategy, symbol, interval, days, params |
| `run_cohort` | Cohort decay-аналіз угод |
| `run_stress` | Стрес-тест (crash/liquidity/vol_spike/funding_shock) |
| `run_capacity` | Capacity-тест (Sharpe при масштабуванні позицій) |
| `paper_step` | Один крок paper-торгівлі на закритому барі (dry-run) |

## Сумісність
- **Claude Code**: `.mcp.json` у корені проєкту (цей файл).
- **Cursor**: `.cursor/mcp.json` (скопіюй блок `mcpServers`).
- **VS Code**: `scalper-hft.code-workspace` (секція `mcp`).

## Поради
- Ніколи не коміть реальні ключі: вони лише в `.env` (git-ignored).
- Для скриптів, що потребують ключів, використовуй `scalper_hft.config.get_settings()` —
  він сам читає `.env`.
