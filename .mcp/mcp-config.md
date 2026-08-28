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

## Сумісність
- **Claude Code**: `.mcp.json` у корені проєкту (цей файл).
- **Cursor**: `.cursor/mcp.json` (скопіюй блок `mcpServers`).
- **VS Code**: `scalper-hft.code-workspace` (секція `mcp`).

## Поради
- Ніколи не коміть реальні ключі: вони лише в `.env` (git-ignored).
- Для скриптів, що потребують ключів, використовуй `scalper_hft.config.get_settings()` —
  він сам читає `.env`.
