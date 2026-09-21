/* ==========================================================================
   GraphRAG front end.

   No framework and no build step: the whole client is one file that talks to
   the FastAPI service over HTTP. Three jobs:

     1. stream an answer over Server-Sent Events and render tokens as they land
     2. show the evidence behind that answer (passages, triples, subgraph)
     3. add a document to the index without interrupting the conversation

   The SSE stream is read with fetch + ReadableStream rather than EventSource,
   because EventSource can only issue GET requests and the chat endpoint takes
   a JSON body.
   ========================================================================== */

const API = "";                       // same origin: FastAPI serves this page
const $ = (id) => document.getElementById(id);

const els = {
  thread:   $("thread"),
  opening:  $("opening"),
  composer: $("composer"),
  input:    $("input"),
  send:     $("send"),
  file:     $("file"),
  docName:  $("doc-name"),
  ingest:   $("ingest"),
  fill:     $("ingest-fill"),
  label:    $("ingest-label"),
  close:    $("ingest-close"),
};

let busy = false;        // a question is in flight
let history = [];        // prior turns, sent for follow-up resolution
let turnSeq = 0;

/* ── Utilities ─────────────────────────────────────────────────────────── */

/** Escape text before it is ever placed in innerHTML. */
function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/**
 * Render the small subset of Markdown the model actually emits.
 *
 * Hand-rolled rather than pulling in a Markdown library: the input is escaped
 * first, so nothing here can inject HTML, and the whole grammar we need is
 * bold, italic, inline code, lists and paragraphs. `[n]` citation markers
 * become buttons that reveal the passage they refer to.
 */
function renderMarkdown(src, citations = []) {
  let t = esc(src);

  t = t.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  t = t.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  // Citation markers carry the page number inline. Seeing "1 p.12" in the
  // sentence means you know where a claim came from without opening anything;
  // clicking still jumps to the passage itself.
  // Some models emit fullwidth brackets (【2】) instead of ASCII [2];
  // normalise so every citation becomes a clickable marker.
  t = t.replace(/【(\d+)】/g, "[$1]");
  t = t.replace(/\[(\d+)\]/g, (_m, n) => {
    const c = citations[Number(n) - 1];
    const page = c?.pages?.length
      ? (c.pages.length > 1 ? `pp.${c.pages[0]}–${c.pages[c.pages.length - 1]}`
                            : `p.${c.pages[0]}`)
      : "";
    const where = c ? [pagesLabel(c.pages), c.section].filter(Boolean).join(" · ") : "";
    return `<button class="cite" data-n="${n}" title="${esc(where)}">` +
           `<span class="cite__n">${n}</span>` +
           (page ? `<span class="cite__page">${page}</span>` : "") +
           `</button>`;
  });

  const out = [];
  let list = null;
  for (const raw of t.split("\n")) {
    const line = raw.trim();
    if (!line) { if (list) { out.push(`</${list}>`); list = null; } continue; }

    const bullet = line.match(/^[-*•]\s+(.*)$/);
    const number = line.match(/^(\d+)\.\s+(.*)$/);
    if (bullet) {
      if (list !== "ul") { if (list) out.push(`</${list}>`); out.push("<ul>"); list = "ul"; }
      out.push(`<li>${bullet[1]}</li>`);
    } else if (number) {
      if (list !== "ol") { if (list) out.push(`</${list}>`); out.push("<ol>"); list = "ol"; }
      out.push(`<li>${number[2]}</li>`);
    } else {
      if (list) { out.push(`</${list}>`); list = null; }
      out.push(`<p>${line}</p>`);
    }
  }
  if (list) out.push(`</${list}>`);
  return out.join("");
}

/**
 * Build a link into the source PDF at the cited page.
 *
 * The `#page=N` fragment is understood by every built-in browser PDF viewer,
 * so a citation resolves to the actual page of the actual document rather
 * than to a claim that such a page exists.
 */
function pdfLink(citation, label) {
  const page = citation?.pages?.[0];
  const docId = citation?.doc_id;
  if (!docId) return esc(label);
  const href = `${API}/api/v1/document/${encodeURIComponent(docId)}` + (page ? `#page=${page}` : "");
  return `<a class="pdflink" href="${href}" target="_blank" rel="noopener">${esc(label)}</a>`;
}

function pagesLabel(pages) {
  if (!pages || !pages.length) return "page n/a";
  if (pages.length === 1) return `page ${pages[0]}`;
  return `pages ${pages[0]}–${pages[pages.length - 1]}`;
}

function scrollToEnd() {
  els.thread.scrollTop = els.thread.scrollHeight;
}

/* ── Status bar ────────────────────────────────────────────────────────── */

/** Refresh the index counters. Silent on failure: the bar is informational. */
async function refreshStats() {
  try {
    const r = await fetch(`${API}/api/v1/graph/stats`);
    if (!r.ok) return;
    const s = await r.json();
    els.docName.textContent = s.documents?.length
      ? s.documents.join(", ").replace(/_/g, " ")
      : "no document indexed";
  } catch { /* the bar simply keeps its last known values */ }
}

/* ── Document upload ───────────────────────────────────────────────────── */

/**
 * Send a PDF to the ingestion endpoint and follow the job.
 *
 * The endpoint returns 202 immediately and does the work in the background,
 * so uploading never blocks the conversation. Progress is polled and shown in
 * a strip under the status bar; asking questions meanwhile is fine.
 */
async function uploadDocument(file) {
  const body = new FormData();
  body.append("file", file);
  body.append("extract_graph", "true");
  body.append("reset", "false");

  showIngest(`Uploading ${file.name}`, 0.04, "");

  let job;
  try {
    const r = await fetch(`${API}/api/v1/ingest/upload`, { method: "POST", body });
    if (!r.ok) throw new Error(await r.text());
    job = await r.json();
  } catch (e) {
    showIngest(`Upload failed: ${String(e).slice(0, 120)}`, 1, "is-failed");
    return;
  }
  pollJob(job.job_id);
}

/** Poll one ingestion job until it reaches a terminal state. */
async function pollJob(jobId) {
  let stop = false;
  while (!stop) {
    await new Promise((r) => setTimeout(r, 2500));
    let job;
    try {
      const r = await fetch(`${API}/api/v1/ingest/${jobId}`);
      if (!r.ok) throw new Error("status unavailable");
      job = await r.json();
    } catch { continue; }

    if (job.status === "completed") {
      showIngest(
        `Indexed ${job.pages} pages — ${job.parent_chunks} passages, ` +
        `${job.entities} entities, ${job.relationships} relationships`,
        1, "is-done"
      );
      refreshStats();
      stop = true;
    } else if (job.status === "failed") {
      showIngest(`Ingestion failed: ${(job.error || "").slice(0, 140)}`, 1, "is-failed");
      stop = true;
    } else {
      showIngest(job.stage || job.status, job.progress || 0.05, "");
    }
  }
}

function showIngest(text, progress, state) {
  els.ingest.hidden = false;
  els.ingest.className = `ingest ${state}`.trim();
  els.label.textContent = text;
  els.fill.style.width = `${Math.round(Math.min(Math.max(progress, 0), 1) * 100)}%`;
}

els.close.addEventListener("click", () => { els.ingest.hidden = true; });
els.file.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if (file) uploadDocument(file);
  e.target.value = "";           // allow re-picking the same file
});

/* ── Evidence rendering ────────────────────────────────────────────────── */

/**
 * Build the evidence block beneath an answer.
 *
 * Three views of the same retrieval: the passages the model read, the graph
 * relationships it was given, and the subgraph those relationships came from.
 * Keeping them attached to their own answer means there is never any doubt
 * about which question a piece of evidence belongs to.
 */
/**
 * Build the sources block beneath an answer.
 *
 * One list, no tabs. A reader wants two things after reading an answer: which
 * pages it came from, and the ability to check one. Everything else -- triple
 * counts, retrieval strategy, chunk ids -- is diagnostics, and diagnostics do
 * not belong in the reading path.
 *
 * The relationship graph stays available but folded away, because it answers a
 * different question ("how are these things connected?") that most readers are
 * not asking on most answers.
 */
function buildEvidence(turnEl, citations, triples, subgraph) {
  const wrap = document.createElement("div");
  wrap.className = "sources";

  if (citations.length) {
    const list = document.createElement("div");
    list.className = "sources__list";

    citations.forEach((c, i) => {
      const row = document.createElement("div");
      row.className = "passage";
      row.dataset.n = String(i + 1);
      row.innerHTML = `
        <button class="passage__head" type="button" aria-expanded="false">
          <span class="passage__n">${i + 1}</span>
          <span class="passage__page">${esc(pagesLabel(c.pages))}</span>
          <span class="passage__where">${esc(c.section || "")}</span>
          <span class="passage__chev" aria-hidden="true"></span>
        </button>
        <div class="passage__body" hidden>${esc(c.text)}
          <p class="passage__open">${pdfLink(c, "Open this page in the PDF")}</p>
        </div>`;

      const head = row.querySelector(".passage__head");
      const body = row.querySelector(".passage__body");
      head.addEventListener("click", () => {
        const open = !body.hidden;
        body.hidden = open;
        head.setAttribute("aria-expanded", String(!open));
      });
      list.appendChild(row);
    });

    const label = document.createElement("p");
    label.className = "sources__label";
    label.textContent = citations.length === 1 ? "1 source" : `${citations.length} sources`;
    wrap.append(label, list);
  }

  // The graph is opt-in: one disclosure, only when there is something to draw.
  if ((subgraph?.nodes || []).length) {
    const details = document.createElement("details");
    details.className = "graphfold";
    details.innerHTML = `
      <summary>How these are connected<span>${subgraph.nodes.length} entities</span></summary>
      <div class="well"><div class="well__canvas"></div></div>`;
    details.addEventListener("toggle", () => {
      if (details.open) drawGraph(details, subgraph);
    }, { once: false });
    wrap.appendChild(details);
  }

  if (wrap.children.length) turnEl.appendChild(wrap);
}

/* ── Graph well ────────────────────────────────────────────────────────── */

const NODE_COLOURS = {
  Company: "#7FA7FF", Subsidiary: "#A78BFA", BusinessSegment: "#5EEAD4",
  Product: "#6EE7B7", Service: "#6EE7B7", Person: "#FCA5A5", Role: "#FDBA74",
  Location: "#BEF264", Metric: "#FDE68A", Technology: "#F0ABFC",
  Initiative: "#93C5FD", Regulator: "#CBD5E1", Award: "#FCD34D", Currency: "#5EEAD4",
};

/**
 * Draw the subgraph into the dark well.
 *
 * Rendered lazily, the first time the Graph tab is opened: a vis-network
 * instance per answer is expensive, and most answers are read without ever
 * opening the graph.
 */
function drawGraph(container, subgraph) {
  const host = container.querySelector(".well__canvas");
  if (!host || host.dataset.drawn === "1" || typeof vis === "undefined") return;
  host.dataset.drawn = "1";

  const seeds = new Set(subgraph.seeds || []);
  const nodes = subgraph.nodes.map((n) => {
    const colour = NODE_COLOURS[n.type] || "#94A3B8";
    const seed = seeds.has(n.id);
    return {
      id: n.id,
      label: n.label,
      title: `${n.label}\n${n.type}${n.description ? "\n" + n.description : ""}`,
      shape: "dot",
      size: seed ? 15 : 9,
      color: {
        background: colour,
        border: seed ? "#FFFFFF" : colour,
        highlight: { background: colour, border: "#FFFFFF" },
      },
      borderWidth: seed ? 2.5 : 0,
      font: { color: seed ? "#FFFFFF" : "rgba(255,255,255,.72)",
              size: seed ? 14 : 12, face: "IBM Plex Sans" },
    };
  });

  const edges = subgraph.edges.map((e) => ({
    from: e.source, to: e.target, label: e.type, title: e.evidence || e.type,
    color: { color: "rgba(255,255,255,.22)", highlight: "#5EEAD4" },
    font: { color: "rgba(255,255,255,.45)", size: 9, strokeWidth: 0,
            face: "IBM Plex Mono", align: "middle" },
    arrows: { to: { enabled: true, scaleFactor: 0.42 } },
    smooth: { type: "continuous" },
  }));

  new vis.Network(host, { nodes, edges }, {
    physics: {
      barnesHut: { gravitationalConstant: -8000, springLength: 130, springConstant: 0.03 },
      stabilization: { iterations: 180 },
    },
    interaction: { hover: true, tooltipDelay: 150, dragView: true, zoomView: true },
    nodes: { shadow: false },
  });
}

/* ── Asking ────────────────────────────────────────────────────────────── */

/**
 * Ask a question and stream the answer.
 *
 * Events arrive in a fixed order: the routing decision, then the evidence,
 * then the answer tokens. Showing the evidence *before* the prose means the
 * grounding is visible while the answer is still being written.
 */
async function ask(question) {
  if (busy || !question.trim()) return;
  busy = true;
  els.send.disabled = true;
  els.opening?.remove();

  const id = ++turnSeq;
  const turn = document.createElement("article");
  turn.className = "turn";
  turn.innerHTML = `
    <h2 class="turn__question">${esc(question)}</h2>
    <p class="status">Searching the document…</p>
    <div class="answer"></div>`;
  els.thread.appendChild(turn);
  scrollToEnd();

  const statusEl = turn.querySelector(".status");
  const answerEl = turn.querySelector(".answer");

  let answer = "";
  let citations = [], triples = [], subgraph = { nodes: [], edges: [], seeds: [] };

  try {
    const res = await fetch(`${API}/api/v1/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history, use_graph: true }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "", event = "message";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line; fields by newlines.
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";

      for (const frame of frames) {
        for (const line of frame.split("\n")) {
          if (line.startsWith("event:")) { event = line.slice(6).trim(); continue; }
          if (!line.startsWith("data:")) continue;

          let data;
          try { data = JSON.parse(line.slice(5).trim()); } catch { continue; }

          if (event === "status") {
            // Two plain states -- searching, then writing. The routing
            // decision is recorded server-side; narrating it here just adds
            // jargon to the moment someone is waiting for an answer.
            statusEl.textContent = data.stage === "generating"
              ? "Writing answer…" : "Searching the document…";
          } else if (event === "citations") {
            citations = data;
          } else if (event === "triples") {
            triples = data;
          } else if (event === "subgraph") {
            subgraph = data;
          } else if (event === "token") {
            answer += data;
            statusEl.remove();   // first token: the wait is over
            answerEl.innerHTML = renderMarkdown(answer, citations) + '<span class="caret"></span>';
            scrollToEnd();
          } else if (event === "done") {
            answer = data.answer || answer;
          } else if (event === "error") {
            throw new Error(data.message || "stream failed");
          }
        }
      }
    }

    statusEl.remove();
    answerEl.innerHTML = renderMarkdown(answer, citations);
    buildEvidence(turn, citations, triples, subgraph);
    wireCitations(turn);

    history.push({ role: "user", content: question });
    history.push({ role: "assistant", content: answer });
    history = history.slice(-6);
    refreshStats();
  } catch (e) {
    statusEl.isConnected
      ? (statusEl.textContent = `Could not answer: ${String(e)}`)
      : answerEl.insertAdjacentHTML("beforeend",
          `<p class="status">Could not answer: ${esc(String(e))}</p>`);
  } finally {
    busy = false;
    els.send.disabled = !els.input.value.trim();
    scrollToEnd();
  }
}

/** Make `[n]` markers open and highlight the passage they cite. */
function wireCitations(turn) {
  turn.querySelectorAll(".cite").forEach((btn) => {
    btn.addEventListener("click", () => {
      const row = turn.querySelector(`.passage[data-n="${btn.dataset.n}"]`);
      if (!row) return;
      const body = row.querySelector(".passage__body");
      if (body?.hidden) row.querySelector(".passage__head").click();
      row.scrollIntoView({ block: "center", behavior: "smooth" });
      row.classList.add("is-flash");
      setTimeout(() => row.classList.remove("is-flash"), 1200);
    });
  });
}

/* ── Composer wiring ───────────────────────────────────────────────────── */

function autosize() {
  els.input.style.height = "auto";
  els.input.style.height = `${Math.min(els.input.scrollHeight, 160)}px`;
}

els.input.addEventListener("input", () => {
  autosize();
  els.send.disabled = busy || !els.input.value.trim();
});

els.input.addEventListener("keydown", (e) => {
  // Enter sends; Shift+Enter is a newline.
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    els.composer.requestSubmit();
  }
});

els.composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = els.input.value.trim();
  if (!q) return;
  els.input.value = "";
  autosize();
  ask(q);
});

document.getElementById("starters")?.addEventListener("click", (e) => {
  const b = e.target.closest("button[data-q]");
  if (b) ask(b.dataset.q);
});

refreshStats();
setInterval(refreshStats, 20000);
