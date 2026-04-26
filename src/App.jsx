import { useState, useRef, useCallback, useEffect } from "react";

const css = `
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:        #0a0a0b;
  --surface:   #111114;
  --surface2:  #16161a;
  --border:    #222228;
  --border2:   #2e2e38;
  --amber:     #f5a623;
  --amber-dim: #7d5210;
  --amber-bg:  rgba(245,166,35,.06);
  --green:     #3ddc84;
  --red:       #ff4757;
  --blue:      #82b1ff;
  --purple:    #c3a6ff;
  --gold:      #f9c784;
  --muted:     #4a4a5a;
  --muted2:    #6a6a7a;
  --text:      #c8c8d8;
  --text-hi:   #eeeef8;
  --mono: 'IBM Plex Mono', monospace;
  --r: 3px;
}

html, body { height: 100%; background: var(--bg); overflow: hidden; }
body { font-family: var(--mono); color: var(--text); }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

/* ── App shell ── */
.app { height: 100vh; display: flex; flex-direction: column; }

/* ── Header ── */
.hdr {
  flex-shrink: 0;
  height: 48px;
  display: flex; align-items: center; gap: 12px;
  padding: 0 18px;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
  z-index: 20;
}
.hdr-logo { font-size: 11px; font-weight: 600; letter-spacing: .18em; color: var(--amber); text-transform: uppercase; white-space: nowrap; }
.hdr-div  { width: 1px; height: 16px; background: var(--border2); flex-shrink: 0; }
.hdr-sub  { font-size: 10px; color: var(--muted); white-space: nowrap; }
.hdr-sep  { flex: 1; }
.hdr-tag  { font-size: 9px; color: var(--muted); letter-spacing: .1em; padding: 3px 8px; border: 1px solid var(--border); border-radius: 2px; white-space: nowrap; flex-shrink: 0; }

/* ── Body: side-by-side desktop ── */
.body {
  flex: 1; overflow: hidden;
  display: grid;
  grid-template-columns: 310px 1fr;
}

/* ── Left panel ── */
.pnl-left {
  border-right: 1px solid var(--border);
  background: var(--surface);
  display: flex; flex-direction: column;
  overflow: hidden;
}

/* Scrollable area inside left panel */
.pnl-left-scroll {
  flex: 1; overflow-y: auto; overflow-x: hidden;
}

/* ── Section wrapper ── */
.sec { padding: 14px 16px; border-bottom: 1px solid var(--border); }
.sec-label {
  display: block; font-size: 9px; letter-spacing: .18em;
  color: var(--muted); text-transform: uppercase; margin-bottom: 10px;
}

/* ── Drop zone ── */
.dropzone {
  position: relative;
  border: 1px dashed var(--border2); border-radius: var(--r);
  padding: 20px 14px; text-align: center; cursor: pointer;
  transition: border-color .15s, background .15s, box-shadow .15s;
  user-select: none; -webkit-tap-highlight-color: transparent;
}
.dropzone:hover { border-color: var(--amber); background: var(--amber-bg); box-shadow: 0 0 0 1px var(--amber-dim) inset; }
.dropzone:active { transform: scale(.997); }
.dropzone.over  { border-color: var(--amber); background: var(--amber-bg); box-shadow: 0 0 18px rgba(245,166,35,.1); }
.dropzone { cursor: pointer; user-select: none; }
.dropzone input { display: none; }
.dz-icon   { font-size: 24px; opacity: .4; margin-bottom: 6px; pointer-events: none; }
.dz-cta    { font-size: 11px; color: var(--amber); font-weight: 500; margin-bottom: 2px; pointer-events: none; }
.dz-or     { font-size: 10px; color: var(--text); margin-bottom: 2px; pointer-events: none; }
.dz-hint   { font-size: 9px; color: var(--muted); letter-spacing: .04em; pointer-events: none; }

/* ── File pill ── */
.file-pill {
  display: flex; align-items: center; gap: 8px;
  padding: 9px 12px;
  background: var(--amber-bg); border: 1px solid var(--amber-dim);
  border-radius: var(--r);
}
.fp-icon { font-size: 14px; flex-shrink: 0; }
.fp-name { font-size: 11px; color: var(--amber); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.fp-size { font-size: 9px; color: var(--muted); flex-shrink: 0; }
.fp-del  { font-size: 15px; color: var(--muted); cursor: pointer; flex-shrink: 0; padding: 1px 4px; border-radius: 2px; transition: color .1s, background .1s; line-height: 1; }
.fp-del:hover { color: var(--red); background: rgba(255,71,87,.1); }

/* ── Setting row ── */
.setting { display: flex; align-items: center; gap: 8px; margin-top: 10px; }
.setting-lbl { font-size: 10px; color: var(--muted); flex: 1; }
.setting-sel {
  background: var(--bg); border: 1px solid var(--border2);
  color: var(--text); font-family: var(--mono); font-size: 10px;
  padding: 4px 8px; border-radius: 2px; cursor: pointer;
}
.setting-sel:focus { outline: none; border-color: var(--amber); }

/* ── Run button ── */
.run-btn {
  width: 100%; margin-top: 12px; padding: 11px;
  background: var(--amber); color: #0a0a0b;
  border: none; border-radius: var(--r);
  font-family: var(--mono); font-size: 11px; font-weight: 600;
  letter-spacing: .14em; text-transform: uppercase;
  cursor: pointer; position: relative; overflow: hidden;
  transition: opacity .15s, transform .08s, box-shadow .15s;
}
.run-btn::after {
  content: ''; position: absolute; inset: 0;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,.16), transparent);
  transform: translateX(-100%);
}
.run-btn.spin::after { animation: shim 1.4s infinite; }
@keyframes shim { to { transform: translateX(100%); } }
.run-btn:hover:not(:disabled) { opacity: .88; box-shadow: 0 0 16px rgba(245,166,35,.22); }
.run-btn:active:not(:disabled) { transform: scale(.988); }
.run-btn:disabled { opacity: .26; cursor: not-allowed; }

/* ── Progress ── */
.prog-wrap { margin-top: 10px; height: 2px; background: var(--border); border-radius: 1px; overflow: hidden; }
.prog-bar  { height: 100%; background: var(--amber); border-radius: 1px; transition: width .32s ease; box-shadow: 0 0 7px rgba(245,166,35,.5); }
.prog-lbl  { font-size: 9px; color: var(--muted); margin-top: 5px; letter-spacing: .05em; min-height: 13px; }

/* ── Preview ── */
.preview-frame {
  width: 100%; aspect-ratio: 4/3; max-height: 130px;
  background: var(--bg); border: 1px solid var(--border); border-radius: var(--r);
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 5px; overflow: hidden; position: relative;
}
.preview-img { width: 100%; height: 100%; object-fit: contain; }
.prev-badge  {
  position: absolute; bottom: 5px; right: 6px;
  font-size: 8px; padding: 2px 5px;
  background: var(--surface); border: 1px solid var(--border2);
  border-radius: 2px; color: var(--muted);
}
.prev-icon { font-size: 20px; opacity: .18; }
.prev-txt  { font-size: 9px; color: var(--muted); letter-spacing: .1em; text-transform: uppercase; }

/* ── Pipeline stages ── */
.stage-list { display: flex; flex-direction: column; gap: 3px; }
.stage-row  { display: flex; align-items: center; gap: 8px; padding: 2px 0; }
.s-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; transition: background .2s, box-shadow .2s; }
.s-dot.idle   { background: var(--border2); }
.s-dot.active { background: var(--amber); box-shadow: 0 0 6px var(--amber); animation: pdot .9s infinite; }
.s-dot.done   { background: var(--green); }
@keyframes pdot { 0%,100%{opacity:1}50%{opacity:.35} }
.s-name { flex: 1; font-size: 9px; color: var(--muted); }
.s-name.active { color: var(--text); }
.s-name.done   { color: var(--muted2); }
.s-ms { font-size: 8px; color: var(--muted); }

/* ── Recent jobs ── */
.recent-list { display: flex; flex-direction: column; gap: 5px; }
.job-item {
  display: flex; align-items: center; gap: 8px;
  padding: 7px 10px;
  background: var(--bg); border: 1px solid var(--border);
  border-radius: var(--r); cursor: pointer;
  transition: border-color .12s, background .12s;
}
.job-item:hover { border-color: var(--border2); background: var(--surface2); }
.job-item.sel   { border-color: var(--amber-dim); background: var(--amber-bg); }
.job-ico  { font-size: 12px; flex-shrink: 0; opacity: .7; }
.job-info { flex: 1; overflow: hidden; }
.job-name { font-size: 10px; color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.job-meta { font-size: 9px; color: var(--muted); margin-top: 1px; }
.job-conf { font-size: 9px; font-weight: 600; flex-shrink: 0; padding: 2px 6px; border-radius: 2px; }
.job-conf.hi  { background:rgba(61,220,132,.1);  color:var(--green); }
.job-conf.mid { background:rgba(245,166,35,.1);   color:var(--amber); }
.job-conf.lo  { background:rgba(255,71,87,.1);    color:var(--red);   }
.recent-empty { font-size: 10px; color: var(--muted); text-align: center; padding: 16px 0; }

/* ═══════════ RIGHT PANEL ═══════════ */
.pnl-right { display: flex; flex-direction: column; overflow: hidden; background: var(--bg); }

/* ── Tabs ── */
.tabs {
  flex-shrink: 0; height: 40px;
  display: flex; align-items: stretch;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
  overflow-x: auto; overflow-y: hidden; scrollbar-width: none;
}
.tabs::-webkit-scrollbar { display: none; }
.tab {
  padding: 0 15px; height: 100%;
  font-size: 9px; letter-spacing: .14em; text-transform: uppercase;
  color: var(--muted); cursor: pointer;
  border: none; border-bottom: 2px solid transparent;
  background: none; font-family: var(--mono);
  display: flex; align-items: center; gap: 5px;
  transition: color .12s; flex-shrink: 0; white-space: nowrap;
}
.tab:hover { color: var(--text); }
.tab.active { color: var(--amber); border-bottom-color: var(--amber); }
.tab-ct { font-size: 8px; padding: 1px 5px; background: var(--border); border-radius: 10px; color: var(--muted); }
.tab.active .tab-ct { background: var(--amber-dim); color: var(--amber); }

/* ── Content ── */
.content { flex: 1; overflow: hidden; position: relative; }

/* Empty state */
.empty {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; height: 100%; gap: 10px;
  color: var(--muted); padding: 32px;
}
.empty-ico   { font-size: 38px; opacity: .14; }
.empty-title { font-size: 11px; letter-spacing: .12em; text-transform: uppercase; }
.empty-hint  { font-size: 10px; text-align: center; line-height: 1.65; max-width: 260px; }

/* ── Features table ── */
.feat-wrap { height: 100%; overflow: auto; }
.feat-tbl  { width: 100%; border-collapse: collapse; font-size: 10px; min-width: 500px; }
.feat-tbl thead { position: sticky; top: 0; background: var(--surface); z-index: 2; }
.feat-tbl th {
  padding: 9px 13px; text-align: left; font-size: 8px;
  letter-spacing: .16em; text-transform: uppercase;
  color: var(--muted); font-weight: 500; white-space: nowrap;
  border-bottom: 1px solid var(--border2);
}
.feat-tbl td { padding: 8px 13px; border-bottom: 1px solid var(--border); vertical-align: middle; }
.feat-tbl tr:last-child td { border-bottom: none; }
.feat-tbl tr:hover td { background: rgba(255,255,255,.018); }

.badge {
  display: inline-block; padding: 2px 7px;
  border-radius: 2px; font-size: 8px; letter-spacing: .1em;
  text-transform: uppercase; font-weight: 600; white-space: nowrap;
}
.b-hole, .b-tap_hole { background:rgba(61,220,132,.1);  color:var(--green);  border:1px solid rgba(61,220,132,.2);  }
.b-thread             { background:rgba(195,166,255,.1); color:var(--purple); border:1px solid rgba(195,166,255,.2); }
.b-slot               { background:rgba(245,166,35,.1);  color:var(--amber);  border:1px solid rgba(245,166,35,.2);  }
.b-pocket             { background:rgba(130,177,255,.1); color:var(--blue);   border:1px solid rgba(130,177,255,.2); }
.b-contour            { background:rgba(74,74,90,.25);   color:var(--text);   border:1px solid var(--border2);       }
.b-radius, .b-chamfer { background:rgba(249,199,132,.1); color:var(--gold);   border:1px solid rgba(249,199,132,.2); }

.cp { font-size: 9px; font-weight: 600; }
.cp.hi  { color:var(--green); }
.cp.mid { color:var(--amber); }
.cp.lo  { color:var(--red);   }
.num  { font-variant-numeric: tabular-nums; color: var(--text-hi); }
.dim  { font-variant-numeric: tabular-nums; color: var(--text-hi); }
.mut  { color: var(--muted); font-size: 9px; }
.mfg-tags { display: flex; flex-wrap: wrap; gap: 3px; }
.mfg-tag {
  font-size: 8px; padding: 1px 5px;
  background: var(--surface); border: 1px solid var(--border2);
  border-radius: 2px; color: var(--muted2);
}

/* ── JSON ── */
.json-wrap { position: relative; height: 100%; }
.json-view {
  height: 100%; overflow: auto;
  padding: 16px 18px;
  font-size: 11px; line-height: 1.75;
  white-space: pre; tab-size: 2;
}
.j-key  { color: var(--blue); }
.j-str  { color: #a8e6cf; }
.j-num  { color: var(--gold); }
.j-bool { color: var(--purple); }
.j-null { color: var(--muted); }
.json-bar {
  position: absolute; top: 10px; right: 12px;
  display: flex; gap: 6px; z-index: 4;
}
.j-btn {
  font-size: 8px; letter-spacing: .1em; text-transform: uppercase;
  padding: 4px 9px; background: var(--surface);
  border: 1px solid var(--border2); border-radius: 2px;
  cursor: pointer; color: var(--muted); font-family: var(--mono);
  transition: color .12s, border-color .12s;
}
.j-btn:hover { color: var(--amber); border-color: var(--amber-dim); }
.j-btn.ok    { color: var(--green); border-color: rgba(61,220,132,.3); }

/* ── Diagnostics ── */
.diag-view  { height: 100%; overflow-y: auto; padding: 12px 14px; }
.diag-grid  { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.d-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); overflow: hidden; }
.d-card.full { grid-column: 1 / -1; }
.d-hdr {
  display: flex; align-items: center; justify-content: space-between;
  padding: 7px 10px; border-bottom: 1px solid var(--border);
  background: rgba(255,255,255,.02);
  font-size: 8px; letter-spacing: .14em; text-transform: uppercase; color: var(--muted);
}
.d-body { padding: 8px 10px; display: flex; flex-direction: column; gap: 4px; }
.d-row  { display: flex; justify-content: space-between; align-items: baseline; gap: 6px; }
.d-k { font-size: 9px; color: var(--muted); white-space: nowrap; }
.d-v { font-size: 9px; color: var(--text-hi); text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 140px; }
.d-v.ok   { color: var(--green); }
.d-v.warn { color: var(--amber); }
.d-v.err  { color: var(--red);   }

/* Gauge */
.gauge-wrap { padding: 3px 0 2px; }
.gauge-track { height: 5px; background: var(--border); border-radius: 3px; overflow: hidden; }
.gauge-fill  { height: 100%; border-radius: 3px; transition: width .7s cubic-bezier(.4,0,.2,1); }
.gauge-row   { display: flex; justify-content: space-between; align-items: baseline; margin-top: 5px; }
.gauge-pct   { font-size: 18px; font-weight: 600; }
.gauge-lbl   { font-size: 8px; color: var(--muted); letter-spacing: .08em; text-transform: uppercase; }

/* Timing bars */
.t-bars { display: flex; flex-direction: column; gap: 3px; }
.t-row  { display: flex; align-items: center; gap: 6px; }
.t-lbl  { font-size: 8px; color: var(--muted); width: 120px; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.t-trk  { flex: 1; height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }
.t-fill { height: 100%; background: var(--amber); border-radius: 2px; opacity: .65; }
.t-ms   { font-size: 8px; color: var(--muted2); width: 34px; text-align: right; flex-shrink: 0; }

/* Signals */
.sig-list { display: flex; flex-direction: column; gap: 4px; }
.sig {
  font-size: 9px; line-height: 1.5; padding: 5px 8px;
  border-left: 2px solid; border-radius: 0 2px 2px 0;
}
.sig.HIGH   { border-color:var(--red);   background:rgba(255,71,87,.06);  color:#ff9099; }
.sig.MEDIUM { border-color:var(--amber); background:rgba(245,166,35,.05); color:#c8922a; }
.sig.LOW    { border-color:var(--muted); background:rgba(74,74,90,.1);    color:var(--muted2); }

/* Suppressed */
.supp-item { padding: 5px 0; border-bottom: 1px solid var(--border); }
.supp-item:last-child { border-bottom: none; }
.supp-id  { font-size: 9px; color: var(--amber); margin-bottom: 1px; }
.supp-why { font-size: 8px; color: var(--muted); line-height: 1.5; }

/* ═══════════ RESPONSIVE ═══════════ */
@media (max-width: 767px) {
  html, body { overflow: auto; height: auto; }
  #root { height: auto; }
  .app  { height: auto; min-height: 100vh; overflow: visible; }
  .body { grid-template-columns: 1fr; grid-template-rows: auto auto; overflow: visible; }
  .pnl-left { border-right: none; border-bottom: 1px solid var(--border); overflow: visible; }
  .pnl-left-scroll { overflow: visible; height: auto; }
  .pnl-right { overflow: visible; min-height: 70vh; }
  .content { overflow: visible; min-height: 60vh; }
  .feat-wrap { height: auto; min-height: 60vh; }
  .json-wrap { height: auto; }
  .json-view { height: auto; min-height: 60vh; }
  .diag-view { height: auto; }
  .empty     { height: 40vh; }
  .diag-grid { grid-template-columns: 1fr; }
  .hdr-sub, .hdr-div { display: none; }
  .hdr-tag + .hdr-tag { display: none; }
}

@media (max-width: 480px) {
  .hdr { padding: 0 14px; gap: 10px; }
  .sec  { padding: 12px 14px; }
}
`;

// ─── Constants ───────────────────────────────────────────────────────────────

const STAGES = [
  { key:"stage0_ingest",       label:"Ingest"       },
  { key:"stage1_preprocess",   label:"Preprocess"   },
  { key:"stage2_ocr",          label:"OCR"          },
  { key:"stage25_titleblock",  label:"Title Block"  },
  { key:"stage3_geometry",     label:"Geometry"     },
  { key:"stage4_scale",        label:"Scale"        },
  { key:"stage5_assemble",     label:"Assemble"     },
  { key:"stage6_diagnostics",  label:"Diagnostics"  },
];

const RECENT_SEED = [
  { id:1, name:"bracket_v3.pdf",    conf:0.89, type:"pdf",   features:7,  time:"2 min ago"  },
  { id:2, name:"flange_photo.jpg",  conf:0.71, type:"photo", features:5,  time:"18 min ago" },
  { id:3, name:"shaft_detail.pdf",  conf:0.93, type:"pdf",   features:11, time:"1 hr ago"   },
];

// ─── Utils ───────────────────────────────────────────────────────────────────

const fmtN   = (n, d=3) => (n==null||typeof n!=="number"||isNaN(n)) ? "—" : parseFloat(n.toFixed(d)).toString();
const fmtB   = (b) => b<1024?b+" B":b<1048576?(b/1024).toFixed(1)+" KB":(b/1048576).toFixed(1)+" MB";
const ccls   = (c) => c>=.8?"hi":c>=.5?"mid":"lo";
const ccol   = (c) => c>=.8?"var(--green)":c>=.5?"var(--amber)":"var(--red)";
const slevel = (s) => s.includes("[HIGH]")?"HIGH":s.includes("[MEDIUM]")?"MEDIUM":"LOW";
const stext  = (s) => s.replace(/^\[(HIGH|MEDIUM|LOW)\]\s*/,"");

function hlJson(json) {
  return json.replace(
    /("(?:\\u[0-9a-fA-F]{4}|\\[^u]|[^\\"])*"(?:\s*:)?|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)/g,
    m => {
      const c = /^"/.test(m)?(/:$/.test(m)?"j-key":"j-str"):/true|false/.test(m)?"j-bool":m==="null"?"j-null":"j-num";
      return `<span class="${c}">${m}</span>`;
    }
  );
}

// ─── Real pipeline — POST /api/recognize ─────────────────────────────────────
//
// Backend contract:
//   POST /api/recognize
//     multipart: file=<drawing>, dpi=<number>
//   Response A — streaming SSE (Content-Type: text/event-stream):
//     data: {"stage":"stage0_ingest"}
//     data: {"stage":"stage2_ocr"}
//     ...
//     data: {"done":true,"result":{partData,diagnostics}}
//   Response B — plain JSON (Content-Type: application/json):
//     {partData, diagnostics}
//   Error response (any 4xx/5xx):
//     {error: "human-readable message"}
//
// Stage keys match STAGES[].key so the progress bar lights up correctly.

// Map stage key → index for fast lookup
const STAGE_INDEX = Object.fromEntries(STAGES.map((s,i)=>[s.key,i]));

async function runPipeline(file, dpi, onStage, signal) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("dpi",  String(dpi));

  let res;
  try {
    res = await fetch("/api/recognize", {
      method: "POST",
      body:   fd,
      signal,                           // AbortController signal for cancel
    });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new Error(`Network error: ${err.message}`);
  }

  if (!res.ok) {
    // Try to get a structured error message from the server
    let msg = `Server error ${res.status}`;
    try {
      const body = await res.json();
      if (body?.error) msg = body.error;
    } catch (_) { /* ignore parse failure */ }
    throw new Error(msg);
  }

  const contentType = res.headers.get("content-type") || "";

  // ── Path A: Server-Sent Events (streaming progress) ──────────────────────
  if (contentType.includes("text/event-stream")) {
    return await _readSSE(res, onStage);
  }

  // ── Path B: Plain JSON (pipeline ran synchronously on server) ────────────
  // Simulate progressive stage advancement so the UI feels live
  const json = await res.json();
  if (json?.error) throw new Error(json.error);

  // Replay stages using actual timings from diagnostics if present
  const timings = json?.diagnostics?.stage_timings_ms ?? {};
  for (let i = 0; i < STAGES.length; i++) {
    const key = STAGES[i].key;
    const ms  = timings[key] ?? 80;
    await new Promise(r => setTimeout(r, Math.min(ms, 400)));   // cap replay at 400ms/stage
    onStage(i);
  }

  return json;
}

async function _readSSE(res, onStage) {
  const reader  = res.body.getReader();
  const decoder = new TextDecoder();
  let   buffer  = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();                   // last (possibly incomplete) line stays

    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const raw = line.slice(5).trim();
      if (!raw) continue;

      let msg;
      try { msg = JSON.parse(raw); } catch (_) { continue; }

      if (msg.stage !== undefined) {
        const idx = STAGE_INDEX[msg.stage];
        if (idx !== undefined) onStage(idx);
      }

      if (msg.done && msg.result) {
        reader.cancel();
        return msg.result;
      }

      if (msg.error) throw new Error(msg.error);
    }
  }

  throw new Error("Stream ended without a result.");
}

// ─── Components ──────────────────────────────────────────────────────────────

function DR({k,v,cls}){
  return(
    <div className="d-row">
      <span className="d-k">{k}</span>
      <span className={`d-v${cls?" "+cls:""}`}>{v??'—'}</span>
    </div>
  );
}

function DCard({title,badge,full,children}){
  return(
    <div className={`d-card${full?" full":""}`}>
      <div className="d-hdr">
        <span>{title}</span>
        {badge&&<span style={{color:badge.color,fontSize:8}}>{badge.text}</span>}
      </div>
      <div className="d-body">{children}</div>
    </div>
  );
}

function Gauge({value}){
  const pct=Math.round((value??0)*100);
  return(
    <div className="gauge-wrap">
      <div className="gauge-track">
        <div className="gauge-fill" style={{width:pct+"%",background:ccol(value)}}/>
      </div>
      <div className="gauge-row">
        <span className="gauge-pct" style={{color:ccol(value)}}>{pct}%</span>
        <span className="gauge-lbl">overall confidence</span>
      </div>
    </div>
  );
}

function TimingBars({timings}){
  if(!timings) return null;
  const entries=Object.entries(timings);
  const max=Math.max(...entries.map(([,v])=>v),1);
  return(
    <div className="t-bars">
      {entries.map(([k,v])=>(
        <div className="t-row" key={k}>
          <span className="t-lbl">{k.replace(/_/g," ")}</span>
          <div className="t-trk"><div className="t-fill" style={{width:(v/max*100)+"%"}}/></div>
          <span className="t-ms">{v}</span>
        </div>
      ))}
    </div>
  );
}

function DiagView({diag}){
  if(!diag) return <div className="empty"><div className="empty-ico">⬡</div><div className="empty-title">No Data</div></div>;
  const supp=diag.suppressed?.length??0;
  const sig=diag.weak_signals?.length??0;
  return(
    <div className="diag-view">
      <div className="diag-grid">
        <DCard title="Recognition Quality" full>
          <Gauge value={diag.overall_confidence}/>
          <DR k="pipeline"    v={diag.pipeline}/>
          <DR k="benchmark"   v={diag.recommend_benchmark?"yes":"no"}/>
        </DCard>
        <DCard title="OCR" badge={{text:diag.ocr?.engine,color:"var(--muted)"}}>
          <DR k="regions"    v={diag.ocr?.regions_found}/>
          <DR k="avg conf"   v={((diag.ocr?.avg_confidence??0)*100).toFixed(0)+"%"}
              cls={diag.ocr?.avg_confidence>=.8?"ok":diag.ocr?.avg_confidence>=.6?"warn":"err"}/>
          <DR k="garbled"    v={diag.ocr?.garbled_count}  cls={diag.ocr?.garbled_count>0?"warn":"ok"}/>
          <DR k="parsed"     v={diag.ocr?.known_symbols_parsed}/>
        </DCard>
        <DCard title="Geometry">
          <DR k="candidates" v={diag.geometry?.total_candidates}/>
          <DR k="avg conf"   v={((diag.geometry?.avg_confidence??0)*100).toFixed(0)+"%"}/>
          <DR k="masked"     v={(diag.geometry?.text_mask_regions??0)+" regions"}/>
          {Object.entries(diag.geometry?.kinds??{}).map(([k,v])=>(
            <DR key={k} k={"· "+k} v={v}/>
          ))}
        </DCard>
        <DCard title="Assembly">
          <DR k="pairs"      v={diag.cross_link?.text_geometry_pairs}/>
          <DR k="suppressed" v={(diag.cross_link?.suppressed_by_linker??0)+(diag.cross_link?.suppressed_by_conflict??0)}
              cls={supp>3?"warn":"ok"}/>
          <DR k="features"   v={diag.cross_link?.features_assembled}/>
        </DCard>
        <DCard title="Stage Timings" full>
          <TimingBars timings={diag.stage_timings_ms}/>
        </DCard>
        {sig>0&&(
          <DCard title={`Signals (${sig})`} full>
            <div className="sig-list">
              {diag.weak_signals.map((s,i)=>(
                <div key={i} className={`sig ${slevel(s)}`}>{stext(s)}</div>
              ))}
            </div>
          </DCard>
        )}
        {supp>0&&(
          <DCard title={`Suppressed (${supp})`} full>
            {diag.suppressed.map((s,i)=>(
              <div key={i} className="supp-item">
                <div className="supp-id">{s.candidate_id} · {s.candidate_kind}</div>
                <div className="supp-why">{s.reason}</div>
              </div>
            ))}
          </DCard>
        )}
      </div>
    </div>
  );
}

function FeatView({features}){
  if(!features?.length) return(
    <div className="empty">
      <div className="empty-ico">⬡</div>
      <div className="empty-title">No Features</div>
      <div className="empty-hint">Run recognition on a drawing to see detected features.</div>
    </div>
  );
  return(
    <div className="feat-wrap">
      <table className="feat-tbl">
        <thead>
          <tr>
            <th>ID</th><th>Kind</th><th>Position</th>
            <th>Dimensions</th><th>Conf</th><th>Mfg</th><th>Src</th>
          </tr>
        </thead>
        <tbody>
          {features.map(f=>{
            let dims="";
            if(f.diameter!=null)       dims=`Ø${fmtN(f.diameter)}`;
            else if(f.length!=null)    dims=`${fmtN(f.length)} × ${fmtN(f.width)}`;
            else if(f.width!=null)     dims=`${fmtN(f.width)} × ${fmtN(f.height)}`;
            else if(f.radius!=null)    dims=`R${fmtN(f.radius)}`;
            const mfg=f.manufacturing||{};
            const tags=[
              mfg.fitClass,
              mfg.threadPitch&&`p${mfg.threadPitch}`,
              mfg.tolerancePlus!=null&&`+${mfg.tolerancePlus}/${-(mfg.toleranceMinus||0)}`,
            ].filter(Boolean);
            return(
              <tr key={f.id}>
                <td className="mut">{f.id}</td>
                <td><span className={`badge b-${f.kind}`}>{f.kind}</span></td>
                <td className="num">x{fmtN(f.x)} y{fmtN(f.y)}</td>
                <td className="dim">{dims||"—"}</td>
                <td><span className={`cp ${ccls(f.confidence)}`}>{Math.round(f.confidence*100)}%</span></td>
                <td>{tags.length
                  ?<div className="mfg-tags">{tags.map((t,i)=><span key={i} className="mfg-tag">{t}</span>)}</div>
                  :<span className="mut">—</span>}
                </td>
                <td className="mut">{f.source}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function JsonView({data}){
  const [copied,setCopied]=useState(false);
  const json=JSON.stringify(data,null,2);
  const copy=()=>{navigator.clipboard?.writeText(json);setCopied(true);setTimeout(()=>setCopied(false),1800);};
  return(
    <div className="json-wrap">
      <div className="json-bar">
        <button className={`j-btn${copied?" ok":""}`} onClick={copy}>{copied?"Copied ✓":"Copy"}</button>
      </div>
      <div className="json-view" dangerouslySetInnerHTML={{__html:hlJson(json)}}/>
    </div>
  );
}

// ─── App ─────────────────────────────────────────────────────────────────────

const TABS=[
  {id:"features",    label:"Features",    count:true},
  {id:"partdata",    label:"Part Data"},
  {id:"diagnostics", label:"Diagnostics", signals:true},
  {id:"full",        label:"Full JSON"},
];

export default function App(){
  const [file,       setFile]      = useState(null);
  const [prevURL,    setPrevURL]   = useState(null);
  const [dpi,        setDpi]       = useState(300);
  const [dragOver,   setDragOver]  = useState(false);
  const [running,    setRunning]   = useState(false);
  const [stageIdx,   setStageIdx]  = useState(-1);
  const [doneIdx,    setDoneIdx]   = useState([]);
  const [result,     setResult]    = useState(null);
  const [tab,        setTab]       = useState("features");
  const [recent,     setRecent]    = useState(RECENT_SEED);
  const [selJob,     setSelJob]    = useState(null);
  const [error,      setError]     = useState(null);
  const abortRef = useRef(null);

  const acceptFile=useCallback(f=>{
    if(!f)return;
    setFile(f); setResult(null); setError(null); setStageIdx(-1); setDoneIdx([]); setSelJob(null);
    // Create preview URL for images and PDFs
    if(/\.(png|jpe?g|tiff?|pdf)$/i.test(f.name)){
      setPrevURL(URL.createObjectURL(f));
    } else {
      setPrevURL(null);
    }
  },[]);

  useEffect(()=>()=>{if(prevURL)URL.revokeObjectURL(prevURL);},[prevURL]);

  // Cancel in-flight request when component unmounts
  useEffect(()=>()=>{ abortRef.current?.abort(); },[]);

  const onDrop=useCallback(e=>{
    e.preventDefault(); setDragOver(false);
    const f=e.dataTransfer?.files?.[0]; if(f)acceptFile(f);
  },[acceptFile]);

  const cancel=()=>{
    abortRef.current?.abort();
    setRunning(false);
    setStageIdx(-1);
    setDoneIdx([]);
    setError("Cancelled.");
  };

  const run=async()=>{
    if(!file||running)return;
    abortRef.current?.abort();                       // cancel any prior request
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setRunning(true); setResult(null); setError(null); setStageIdx(0); setDoneIdx([]);
    try{
      const out=await runPipeline(file, dpi, i=>{
        setDoneIdx(p=>[...p,i]);
        setStageIdx(i+1);
      }, ctrl.signal);

      if(!out?.partData) throw new Error("Server returned an unexpected response shape.");

      setResult(out);
      setStageIdx(STAGES.length);
      const conf=out.diagnostics?.overall_confidence??0;
      const nf=out.partData?.features?.length??0;
      setRecent(p=>[
        {id:Date.now(),name:file.name,conf,
         type:/\.pdf$/i.test(file.name)?"pdf":"photo",features:nf,time:"just now"},
        ...p.slice(0,4),
      ]);
    } catch(e) {
      if(e.name==="AbortError") return;              // silent on user cancel
      setError(e.message||"Unspecified pipeline error.");
      setStageIdx(-1);
      setDoneIdx([]);
      console.error("[M2 pipeline]", e);
    } finally {
      setRunning(false);
    }
  };

  const clearFile=()=>{
    abortRef.current?.abort();
    setFile(null); setResult(null); setError(null);
    setPrevURL(null); setStageIdx(-1); setDoneIdx([]);
  };

  const features=result?.partData?.features||[];
  const diag=result?.diagnostics;

  const progress=stageIdx<0?0:stageIdx>=STAGES.length?1:stageIdx/STAGES.length;
  const activeLabel=running&&stageIdx<STAGES.length
    ?STAGES[stageIdx]?.label+"…"
    :stageIdx>=STAGES.length?"Complete.":"";

  const stageStatus=i=>doneIdx.includes(i)?"done":stageIdx===i&&running?"active":"idle";

  return(
    <>
      <style>{css}</style>
      <div className="app">

        {/* Header */}
        <header className="hdr">
          <span className="hdr-logo">⬡ CNC·AI</span>
          <span className="hdr-div"/>
          <span className="hdr-sub">Drawing Recognition</span>
          <span className="hdr-sep"/>
          <span className="hdr-tag">Module 2 · v1.0.0</span>
          <span className="hdr-tag" style={{marginLeft:6}}>Schema v2</span>
        </header>

        <div className="body">

          {/* ══ LEFT ══ */}
          <aside className="pnl-left">
            <div className="pnl-left-scroll">

              {/* Upload */}
              <div className="sec">
                <span className="sec-label">Input Drawing</span>

                {!file?(
                  <label
                    className={`dropzone${dragOver?" over":""}`}
                    onDragOver={e=>{e.preventDefault();setDragOver(true);}}
                    onDragLeave={e=>{e.preventDefault();setDragOver(false);}}
                    onDragEnd={()=>setDragOver(false)}
                    onDrop={onDrop}
                    aria-label="Upload a drawing file"
                  >
                    <input
                      type="file"
                      accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff"
                      style={{display:"none"}}
                      onChange={e=>{
                        const f=e.target.files?.[0];
                        if(f) acceptFile(f);
                        e.target.value="";   // reset so same file can be re-selected
                      }}
                    />
                    <div className="dz-icon">⬡</div>
                    <div className="dz-cta">Click to upload</div>
                    <div className="dz-or">or drag a drawing here</div>
                    <div className="dz-hint">PDF · PNG · JPG · TIF</div>
                  </label>
                ):(
                  <div className="file-pill">
                    <span className="fp-icon">{/\.pdf$/i.test(file.name)?"📄":"🖼"}</span>
                    <span className="fp-name">{file.name}</span>
                    <span className="fp-size">{fmtB(file.size)}</span>
                    <span className="fp-del" onClick={clearFile} role="button" aria-label="Remove file">×</span>
                  </div>
                )}

                <div className="setting">
                  <span className="setting-lbl">PDF Render DPI</span>
                  <select className="setting-sel" value={dpi} onChange={e=>setDpi(Number(e.target.value))}>
                    {[150,200,300,400,600].map(v=><option key={v} value={v}>{v}</option>)}
                  </select>
                </div>

                <button
                  className={`run-btn${running?" spin":""}`}
                  onClick={run} disabled={!file||running}
                  aria-label="Run recognition pipeline"
                >
                  {running?"Recognizing…":file?"Run Recognition":"Select a Drawing First"}
                </button>

                {running&&(
                  <button
                    onClick={cancel}
                    style={{
                      width:"100%", marginTop:6, padding:"7px",
                      background:"transparent", border:"1px solid var(--border2)",
                      color:"var(--muted)", fontFamily:"var(--mono)", fontSize:10,
                      letterSpacing:".1em", textTransform:"uppercase", cursor:"pointer",
                      borderRadius:2, transition:"color .12s, border-color .12s",
                    }}
                    onMouseOver={e=>{e.target.style.color="var(--red)";e.target.style.borderColor="var(--red)";}}
                    onMouseOut={e=>{e.target.style.color="var(--muted)";e.target.style.borderColor="var(--border2)";}}
                  >Cancel</button>
                )}

                {error&&!running&&(
                  <div style={{
                    marginTop:10, padding:"8px 10px",
                    background:"rgba(255,71,87,.08)",
                    border:"1px solid rgba(255,71,87,.3)",
                    borderRadius:3,
                    display:"flex", alignItems:"flex-start", gap:8,
                  }}>
                    <span style={{color:"var(--red)", fontSize:13, lineHeight:1, flexShrink:0}}>✕</span>
                    <span style={{fontSize:10, color:"#ff8090", lineHeight:1.5}}>{error}</span>
                  </div>
                )}

                {stageIdx>=0&&(
                  <>
                    <div className="prog-wrap">
                      <div className="prog-bar" style={{width:(progress*100)+"%"}}/>
                    </div>
                    <div className="prog-lbl">{activeLabel}</div>
                  </>
                )}
                )}
              </div>

              {/* Preview */}
              <div className="sec">
                <span className="sec-label">Drawing Preview</span>
                <div className="preview-frame">
                  {file && prevURL && /\.(png|jpe?g|tiff?)$/i.test(file.name) ? (
                    <>
                      <img className="preview-img" src={prevURL} alt="Drawing preview"/>
                      <span className="prev-badge">{file.name.split(".").pop().toUpperCase()}</span>
                    </>
                  ) : file && prevURL && /\.pdf$/i.test(file.name) ? (
                    <>
                      <iframe
                        src={prevURL}
                        title="PDF preview"
                        style={{width:"100%",height:"100%",border:"none",borderRadius:3}}
                      />
                      <span className="prev-badge">PDF</span>
                    </>
                  ) : file ? (
                    <>
                      <div className="prev-icon">📄</div>
                      <div className="prev-txt">{file.name} · {fmtB(file.size)}</div>
                    </>
                  ) : (
                    <>
                      <div className="prev-icon">⬡</div>
                      <div className="prev-txt">No drawing loaded</div>
                    </>
                  )}
                </div>
              </div>

              {/* Pipeline status */}
              <div className="sec">
                <span className="sec-label">Pipeline Status</span>
                <div className="stage-list">
                  {STAGES.map((s,i)=>{
                    const st=stageStatus(i);
                    const ms=result?.diagnostics?.stage_timings_ms?.[s.key];
                    return(
                      <div key={s.key} className="stage-row">
                        <div className={`s-dot ${st}`}/>
                        <span className={`s-name ${st}`}>{s.label}</span>
                        {ms!=null&&<span className="s-ms">{ms} ms</span>}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Recent jobs */}
              <div className="sec">
                <span className="sec-label">Recent Jobs</span>
                {recent.length===0
                  ?<div className="recent-empty">No recent jobs</div>
                  :<div className="recent-list">
                    {recent.map(job=>(
                      <div key={job.id}
                        className={`job-item${selJob===job.id?" sel":""}`}
                        onClick={()=>setSelJob(job.id===selJob?null:job.id)}
                        role="button" tabIndex={0}
                      >
                        <span className="job-ico">{job.type==="pdf"?"📄":"🖼"}</span>
                        <div className="job-info">
                          <div className="job-name">{job.name}</div>
                          <div className="job-meta">{job.features} features · {job.time}</div>
                        </div>
                        <span className={`job-conf ${ccls(job.conf)}`}>{Math.round(job.conf*100)}%</span>
                      </div>
                    ))}
                  </div>
                }
              </div>

            </div>{/* /pnl-left-scroll */}
          </aside>

          {/* ══ RIGHT ══ */}
          <main className="pnl-right">
            <div className="tabs" role="tablist">
              {TABS.map(t=>{
                const ct=t.count?features.length:t.signals?(diag?.weak_signals?.length??0):null;
                return(
                  <button key={t.id} role="tab"
                    className={`tab${tab===t.id?" active":""}`}
                    onClick={()=>setTab(t.id)}
                    aria-selected={tab===t.id}
                  >
                    {t.label}
                    {ct>0&&<span className="tab-ct">{ct}</span>}
                  </button>
                );
              })}
            </div>

            <div className="content" role="tabpanel">
              {!result?(
                <div className="empty">
                  <div className="empty-ico">⬡</div>
                  <div className="empty-title">No Result Yet</div>
                  <div className="empty-hint">Upload a technical drawing and press Run Recognition to extract features, dimensions, and diagnostics.</div>
                </div>
              ):tab==="features"?(
                <FeatView features={features}/>
              ):tab==="partdata"?(
                <JsonView data={result.partData}/>
              ):tab==="diagnostics"?(
                <DiagView diag={diag}/>
              ):(
                <JsonView data={result}/>
              )}
            </div>
          </main>

        </div>
      </div>
    </>
  );
}
