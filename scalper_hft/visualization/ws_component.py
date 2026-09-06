"""Streamlit Custom Component v2 (CCv2): субсекундний WebSocket-моніторинг.

Підключається безпосередньо з браузера клієнта до FastAPI бекенду через WebSocket,
відображає статус з'єднання, затримку (ping RTT), живий список позицій та потік подій.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

_HTML = """\
<div class="ws-monitor-root">
  <div class="ws-header-bar">
    <div class="ws-status-group">
      <div id="ws-status-badge" class="badge badge-disconnected">
        <span class="status-dot"></span>
        <span id="ws-status-text">ІНІЦІАЛІЗАЦІЯ...</span>
      </div>
      <div class="meta-pill">
        <span class="pill-label">Затримка:</span>
        <span id="ws-latency-val" class="pill-value">-- ms</span>
      </div>
      <div class="meta-pill">
        <span class="pill-label">Пакетів:</span>
        <span id="ws-packets-val" class="pill-value">0</span>
      </div>
      <div class="meta-pill">
        <span class="pill-label">Останнє оновлення:</span>
        <span id="ws-last-time-val" class="pill-value">--:--:--</span>
      </div>
    </div>
    <div class="ws-url-tag">
      <span id="ws-endpoint-display">ws://...</span>
    </div>
  </div>

  <div class="ws-content-grid">
    <!-- Секція відкритих позицій -->
    <div class="card positions-card">
      <div class="card-header">
        <span class="card-title">⚡ Відкриті позиції (Real-Time)</span>
        <span id="pos-count-badge" class="count-pill">0</span>
      </div>
      <div class="table-container">
        <table class="positions-table">
          <thead>
            <tr>
              <th>Символ</th>
              <th>Біржа</th>
              <th>Сторона</th>
              <th>Розмір</th>
              <th>Вхід</th>
              <th>Mark</th>
              <th>Unrealized PnL</th>
              <th>Час</th>
            </tr>
          </thead>
          <tbody id="positions-tbody">
            <tr class="empty-row">
              <td colspan="8">Очікування даних від WebSocket сервера...</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Секція живого журналу подій (Event Log) -->
    <div class="card log-card">
      <div class="card-header">
        <span class="card-title">📡 Потік WebSocket подій</span>
        <div class="log-actions">
          <button id="btn-pause-log" class="btn-sm" type="button">Пауза логу</button>
          <button id="btn-clear-log" class="btn-sm" type="button">Очистити</button>
        </div>
      </div>
      <div id="ws-event-log" class="event-log-terminal">
        <div class="log-entry system">[SYSTEM] Ініціалізація WebSocket монітора...</div>
      </div>
    </div>
  </div>
</div>
"""

_CSS = """\
.ws-monitor-root {
  font-family: inherit;
  color: var(--st-text-color, #e0e0e0);
  background: var(--st-background-color, #0e1117);
  border: 1px solid var(--st-border-color, #262730);
  border-radius: 8px;
  padding: 14px;
  box-sizing: border-box;
  width: 100%;
}

.ws-header-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--st-border-color, #262730);
  margin-bottom: 14px;
}

.ws-status-group {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 9999px;
  font-size: 0.8rem;
  font-weight: 600;
  letter-spacing: 0.03em;
  text-transform: uppercase;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: currentColor;
  animation: pulse 1.8s infinite ease-in-out;
}

@keyframes pulse {
  0% { transform: scale(0.9); opacity: 0.7; }
  50% { transform: scale(1.2); opacity: 1.0; }
  100% { transform: scale(0.9); opacity: 0.7; }
}

.badge-connected {
  background: rgba(34, 197, 94, 0.15);
  color: #22c55e;
  border: 1px solid rgba(34, 197, 94, 0.4);
}

.badge-connecting {
  background: rgba(234, 179, 8, 0.15);
  color: #eab308;
  border: 1px solid rgba(234, 179, 8, 0.4);
}

.badge-disconnected {
  background: rgba(239, 68, 68, 0.15);
  color: #ef4444;
  border: 1px solid rgba(239, 68, 68, 0.4);
}

.meta-pill {
  display: inline-flex;
  gap: 5px;
  align-items: center;
  background: var(--st-secondary-background-color, #262730);
  padding: 3px 8px;
  border-radius: 6px;
  font-size: 0.75rem;
}

.pill-label {
  opacity: 0.7;
}

.pill-value {
  font-weight: 600;
  font-family: monospace;
}

.ws-url-tag {
  font-family: monospace;
  font-size: 0.75rem;
  opacity: 0.6;
  max-width: 280px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ws-content-grid {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.card {
  background: var(--st-secondary-background-color, #1a1c24);
  border: 1px solid var(--st-border-color, #2d3139);
  border-radius: 6px;
  padding: 12px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.card-title {
  font-weight: 600;
  font-size: 0.9rem;
}

.count-pill {
  background: var(--st-background-color, #0e1117);
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: bold;
}

.table-container {
  overflow-x: auto;
}

.positions-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82rem;
  text-align: left;
}

.positions-table th {
  padding: 6px 8px;
  border-bottom: 1px solid var(--st-border-color, #3a3f4d);
  opacity: 0.7;
  font-weight: 600;
}

.positions-table td {
  padding: 8px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.side-badge {
  display: inline-block;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 0.7rem;
  font-weight: bold;
  text-transform: uppercase;
}

.side-long {
  background: rgba(34, 197, 94, 0.2);
  color: #22c55e;
}

.side-short {
  background: rgba(239, 68, 68, 0.2);
  color: #ef4444;
}

.pnl-pos {
  color: #22c55e;
  font-weight: bold;
}

.pnl-neg {
  color: #ef4444;
  font-weight: bold;
}

.empty-row td {
  text-align: center;
  padding: 20px;
  opacity: 0.6;
  font-style: italic;
}

.log-actions {
  display: flex;
  gap: 6px;
}

.btn-sm {
  background: var(--st-background-color, #0e1117);
  color: var(--st-text-color, #fff);
  border: 1px solid var(--st-border-color, #3a3f4d);
  padding: 3px 8px;
  border-radius: 4px;
  font-size: 0.72rem;
  cursor: pointer;
}

.btn-sm:hover {
  background: rgba(255, 255, 255, 0.1);
}

.event-log-terminal {
  background: #090a0f;
  border: 1px solid #1f232d;
  border-radius: 4px;
  height: 140px;
  overflow-y: auto;
  font-family: monospace;
  font-size: 0.75rem;
  padding: 8px;
  display: flex;
  flex-direction: column-reverse;
  gap: 4px;
}

.log-entry {
  line-height: 1.4;
  word-break: break-all;
}

.log-entry.system { color: #38bdf8; }
.log-entry.pos { color: #a78bfa; }
.log-entry.telem { color: #fbbf24; }
.log-entry.ctrl { color: #f43f5e; }
.log-entry.pong { color: #64748b; }
"""

_JS = """\
export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  if (!parentElement) return;

  const wsUrl = data?.ws_url || "ws://localhost:8080/ws";
  const token = data?.token || "";
  const fullUrl = token ? `${wsUrl}?token=${encodeURIComponent(token)}` : wsUrl;

  const statusBadge = parentElement.querySelector("#ws-status-badge");
  const statusText = parentElement.querySelector("#ws-status-text");
  const latencyVal = parentElement.querySelector("#ws-latency-val");
  const packetsVal = parentElement.querySelector("#ws-packets-val");
  const lastTimeVal = parentElement.querySelector("#ws-last-time-val");
  const endpointDisplay = parentElement.querySelector("#ws-endpoint-display");
  const posCountBadge = parentElement.querySelector("#pos-count-badge");
  const positionsTbody = parentElement.querySelector("#positions-tbody");
  const eventLog = parentElement.querySelector("#ws-event-log");
  const btnPauseLog = parentElement.querySelector("#btn-pause-log");
  const btnClearLog = parentElement.querySelector("#btn-clear-log");

  if (endpointDisplay) endpointDisplay.textContent = wsUrl;

  let socket = null;
  let pingInterval = null;
  let pingSentTime = null;
  let packetCount = 0;
  let isLogPaused = false;
  let reconnectTimer = null;
  let reconnectAttempts = 0;

  function updateStatus(state, message) {
    if (!statusBadge || !statusText) return;
    statusBadge.className = `badge badge-${state}`;
    statusText.textContent = message;
  }

  function appendLog(text, category = "info") {
    if (isLogPaused || !eventLog) return;
    const div = document.createElement("div");
    div.className = `log-entry ${category}`;
    const timeStr = new Date().toTimeString().split(" ")[0];
    div.textContent = `[${timeStr}] ${text}`;
    eventLog.prepend(div);

    // Обмеження розміру логу (не більше 60 записів)
    while (eventLog.children.length > 60) {
      eventLog.removeChild(eventLog.lastChild);
    }
  }

  if (btnPauseLog) {
    btnPauseLog.onclick = () => {
      isLogPaused = !isLogPaused;
      btnPauseLog.textContent = isLogPaused ? "Відновити лог" : "Пауза логу";
    };
  }

  if (btnClearLog) {
    btnClearLog.onclick = () => {
      if (eventLog) eventLog.innerHTML = "";
    };
  }

  function renderPositions(positions) {
    if (!positionsTbody) return;
    if (posCountBadge) posCountBadge.textContent = positions ? positions.length : 0;

    if (!positions || positions.length === 0) {
      positionsTbody.innerHTML = `
        <tr class="empty-row">
          <td colspan="8">Немає відкритих позицій (усі закрито).</td>
        </tr>`;
      return;
    }

    let html = "";
    for (const p of positions) {
      const isLong = (p.side || "").toLowerCase() === "long";
      const sideClass = isLong ? "side-long" : "side-short";
      const pnl = Number(p.unrealized_pnl || 0);
      const pnlClass = pnl >= 0 ? "pnl-pos" : "pnl-neg";
      const pnlSign = pnl >= 0 ? "+" : "";

      html += `
        <tr>
          <td><strong>${p.symbol}</strong></td>
          <td>${p.exchange || "binance"} (${p.mode || "paper"})</td>
          <td><span class="side-badge ${sideClass}">${p.side}</span></td>
          <td style="font-family: monospace;">${Number(p.size).toFixed(4)}</td>
          <td style="font-family: monospace;">$${Number(p.entry_price).toFixed(2)}</td>
          <td style="font-family: monospace;">$${Number(p.mark_price || p.entry_price).toFixed(2)}</td>
          <td class="${pnlClass}" style="font-family: monospace;">${pnlSign}$${pnl.toFixed(2)}</td>
          <td style="opacity: 0.7; font-size: 0.75rem;">${p.ts ? p.ts.split("T")[1]?.slice(0, 8) || p.ts : ""}</td>
        </tr>`;
    }
    positionsTbody.innerHTML = html;
  }

  function connect() {
    if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    updateStatus("connecting", "ПІДКЛЮЧЕННЯ...");
    appendLog(`Спроба підключення до ${wsUrl}...`, "system");

    try {
      socket = new WebSocket(fullUrl);
    } catch (e) {
      updateStatus("disconnected", "ПОМИЛКА URL");
      appendLog(`Помилка створення сокета: ${e.message}`, "system");
      return;
    }

    socket.onopen = () => {
      reconnectAttempts = 0;
      updateStatus("connected", "ПІДКЛЮЧЕНО");
      appendLog("Успішно підключено до WebSocket бекенду", "system");

      // Періодичний ping кожні 3 секунди для вимірювання затримки RTT
      pingInterval = setInterval(() => {
        if (socket && socket.readyState === WebSocket.OPEN) {
          pingSentTime = performance.now();
          socket.send("ping");
        }
      }, 3000);
    };

    socket.onmessage = (event) => {
      packetCount++;
      if (packetsVal) packetsVal.textContent = packetCount;
      if (lastTimeVal) lastTimeVal.textContent = new Date().toTimeString().split(" ")[0];

      if (event.data === "pong") {
        if (pingSentTime && latencyVal) {
          const rtt = Math.round(performance.now() - pingSentTime);
          latencyVal.textContent = `${rtt} ms`;
        }
        return;
      }

      try {
        const msg = JSON.parse(event.data);
        const type = msg.type;

        if (type === "positions_update") {
          renderPositions(msg.data);
          appendLog(`positions_update: ${msg.count ?? (msg.data ? msg.data.length : 0)} позицій`, "pos");
          // Відправляємо тригер у Streamlit
          if (typeof setTriggerValue === "function") {
            setTriggerValue("last_ws_event", msg);
          }
        } else if (type === "snapshot") {
          if (msg.data?.positions) renderPositions(msg.data.positions);
          appendLog("Отримано початковий знімок стану (snapshot)", "system");
        } else if (type === "telemetry_update") {
          appendLog(`telemetry_update: акаунти і боти синхронізовано`, "telem");
        } else if (type === "control_update") {
          appendLog(`control_update: pause=${msg.data?.pause} flatten=${msg.data?.flatten}`, "ctrl");
        } else {
          appendLog(`Повідомлення: ${type}`, "info");
        }
      } catch (err) {
        appendLog(`Помилка парсингу: ${event.data.slice(0, 60)}`, "system");
      }
    };

    socket.onerror = (err) => {
      updateStatus("disconnected", "ПОМИЛКА З'ЄДНАННЯ");
      appendLog("Помилка WebSocket з'єднання (можливо, невірний токен або бекенд офлайн)", "system");
    };

    socket.onclose = (e) => {
      updateStatus("disconnected", "ВІДКЛЮЧЕНО");
      appendLog(`З'єднання закрито (код ${e.code})`, "system");
      if (pingInterval) {
        clearInterval(pingInterval);
        pingInterval = null;
      }
      if (latencyVal) latencyVal.textContent = "-- ms";

      // Спроба автоматичного перепідключення з затримкою
      reconnectAttempts++;
      const delay = Math.min(10000, 1500 * Math.pow(1.5, reconnectAttempts));
      appendLog(`Спроба повторного підключення через ${(delay / 1000).toFixed(1)} с...`, "system");
      reconnectTimer = setTimeout(connect, delay);
    };
  }

  connect();

  return () => {
    if (reconnectTimer) clearTimeout(reconnectTimer);
    if (pingInterval) clearInterval(pingInterval);
    if (socket) {
      socket.onclose = null;
      socket.close();
    }
  };
}
"""

_WS_COMPONENT = st.components.v2.component(
    "ws_live_monitor",
    html=_HTML,
    css=_CSS,
    js=_JS,
)


def ws_live_monitor_component(
    ws_url: str,
    token: str,
    *,
    key: str = "ws_live_monitor",
) -> Any:
    """Монтує інтерактивний субсекундний WebSocket монітор у сторінку Streamlit."""
    return _WS_COMPONENT(
        data={"ws_url": ws_url, "token": token},
        key=key,
        on_last_ws_event_change=lambda: None,
    )
