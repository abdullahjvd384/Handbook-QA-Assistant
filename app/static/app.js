// Handbook Assistant UI. Everything from the server is inserted with textContent, never
// innerHTML, so document text (including uploads) cannot inject markup.
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = text;
    return n;
  };

  const LIFECYCLE_LABEL = {
    in_force: "in force",
    not_yet_effective: "not yet in effect",
    superseded: "superseded",
    deprecated: "deprecated",
  };

  const state = { history: [], busy: false, docs: [] };

  // ---------------------------------------------------------------- status
  async function loadStatus() {
    try {
      const r = await fetch("/api/health");
      const h = await r.json();
      const badge = $("#status");
      const llm = h.answer_mode === "llm";
      badge.textContent = `${llm ? h.llm_model : "Extractive"} · ${h.documents} docs`;
      badge.classList.toggle("llm", llm);
      badge.title = llm
        ? `Answers written by ${h.llm_model}, quotes verified against the source`
        : "No LLM key configured: answers quote the best-matching passages verbatim";
    } catch {
      $("#status").textContent = "offline";
    }
  }

  // ---------------------------------------------------------------- chat
  const messages = $("#messages");

  function scrollToBottom() {
    messages.scrollTop = messages.scrollHeight;
  }

  function addUser(text) {
    $("#empty")?.remove();
    const m = el("div", "msg user");
    m.append(el("div", "bubble", text));
    messages.append(m);
    scrollToBottom();
  }

  function renderAnswerText(container, text, citations) {
    const byN = new Map(citations.map((c) => [c.n, c]));
    for (const para of text.split(/\n{2,}/)) {
      const p = el("p");
      const parts = para.split(/(\[\d+\])/g);
      for (const part of parts) {
        const m = part.match(/^\[(\d+)\]$/);
        if (m && byN.has(Number(m[1]))) {
          const c = byN.get(Number(m[1]));
          const b = el("button", "cite-ref", m[1]);
          b.type = "button";
          b.title = `${c.title}${c.section ? " — " + c.section : ""}`;
          b.addEventListener("click", () => openDoc(c.doc_id, c.start, c.end));
          p.append(b);
        } else if (part) {
          p.append(document.createTextNode(part));
        }
      }
      container.append(p);
    }
  }

  function addBot(res) {
    const m = el("div", "msg bot");
    const bubble = el("div", "bubble");
    const answer = el("div", "answer" + (res.answerable ? "" : " unanswered"));
    renderAnswerText(answer, res.answer, res.citations);
    bubble.append(answer);

    if (res.citations.length) {
      const list = el("div", "sources");
      for (const c of res.citations) {
        const b = el("button", "source");
        b.type = "button";
        b.append(el("span", "n", `[${c.n}]`));
        const body = el("div");
        const head = el("div", "head");
        head.append(el("span", null, c.title));
        if (c.section) head.append(el("span", "section", c.section));
        if (c.lifecycle !== "in_force") head.append(el("span", `lc ${c.lifecycle}`, LIFECYCLE_LABEL[c.lifecycle]));
        body.append(head, el("div", "quote", `“${c.quote}”`));
        b.append(body);
        b.addEventListener("click", () => openDoc(c.doc_id, c.start, c.end));
        list.append(b);
      }
      bubble.append(list);
    }

    const meta = el("div", "meta-line");
    meta.append(el("span", null, res.mode === "llm" ? `answered by ${res.model}` : `mode: ${res.mode}`));
    meta.append(el("span", null, `as of ${res.as_of}`));
    meta.append(el("span", null, `${res.latency_ms} ms`));
    if (res.standalone_question && res.standalone_question !== res.question) {
      meta.append(el("span", null, `searched: “${res.standalone_question}”`));
    }
    bubble.append(meta);
    for (const w of res.warnings || []) bubble.append(el("div", "warn-line", w));
    m.append(bubble);
    messages.append(m);
    scrollToBottom();
  }

  async function ask(question) {
    if (state.busy || !question.trim()) return;
    state.busy = true;
    $("#send").disabled = true;
    addUser(question);
    const pending = el("div", "msg bot");
    pending.append(el("div", "bubble typing", "Searching the handbook…"));
    messages.append(pending);
    scrollToBottom();
    try {
      const r = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history: state.history }),
      });
      const res = await r.json();
      pending.remove();
      if (!r.ok) throw new Error(res.detail ? JSON.stringify(res.detail) : r.statusText);
      addBot(res);
      state.history.push({ role: "user", content: question }, { role: "assistant", content: res.answer });
      state.history = state.history.slice(-12);
    } catch (err) {
      pending.remove();
      const m = el("div", "msg bot");
      m.append(el("div", "bubble warn-line", `Something went wrong: ${err.message}`));
      messages.append(m);
    } finally {
      state.busy = false;
      $("#send").disabled = false;
    }
  }

  const form = $("#ask-form");
  const input = $("#question");
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value;
    input.value = "";
    autoGrow();
    ask(q);
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });
  function autoGrow() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
  }
  input.addEventListener("input", autoGrow);
  document.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => ask(c.textContent)));
  $("#btn-new").addEventListener("click", () => {
    state.history = [];
    messages.replaceChildren();
    const e = el("div", "empty");
    e.id = "empty";
    e.append(el("h2", null, "New conversation"), el("p", null, "Earlier questions are forgotten."));
    messages.append(e);
  });

  // ---------------------------------------------------------------- document viewer
  const viewer = $("#viewer");

  function closeViewer() {
    viewer.hidden = true;
    document.querySelector(".layout").classList.remove("with-viewer");
    history.replaceState(null, "", "/");
  }
  $("#viewer-close").addEventListener("click", closeViewer);

  // Append `text` to `parent`, wrapping the part that falls inside [hs, he) in <mark>.
  function appendHighlighted(parent, text, offset, hs, he, marks) {
    const s = Math.max(hs - offset, 0);
    const e = Math.min(he - offset, text.length);
    if (hs === null || e <= 0 || s >= text.length || e <= s) {
      parent.append(document.createTextNode(text));
      return;
    }
    if (s > 0) parent.append(document.createTextNode(text.slice(0, s)));
    const mk = el("mark", null, text.slice(s, e));
    marks.push(mk);
    parent.append(mk);
    if (e < text.length) parent.append(document.createTextNode(text.slice(e)));
  }

  async function openDoc(docId, hs = null, he = null) {
    const r = await fetch(`/api/documents/${encodeURIComponent(docId)}`);
    if (!r.ok) return;
    const d = await r.json();
    $("#doc-title").textContent = d.title;
    const meta = $("#doc-meta");
    meta.replaceChildren(
      el("span", `lc ${d.lifecycle}`, LIFECYCLE_LABEL[d.lifecycle]),
      el("span", null, d.category),
      el("span", null, d.owner),
      el("span", null, d.effective_date ? `effective ${d.effective_date}` : "no effective date"),
      el("span", null, `${d.format.toUpperCase()} · ${d.doc_id}`)
    );
    const notice = $("#doc-notice");
    notice.hidden = d.lifecycle === "in_force";
    notice.textContent = d.lifecycle_note || "";
    $("#doc-download").href = `/api/documents/${encodeURIComponent(docId)}/file`;

    const body = $("#doc-body");
    body.replaceChildren();
    const marks = [];
    let list = null;
    let lastPage = null;
    for (const b of d.blocks) {
      const text = d.text.slice(b.start, b.end);
      if (b.page && b.page !== lastPage && d.format === "pdf") {
        body.append(el("div", "page", `Page ${b.page}`));
        lastPage = b.page;
      }
      let node;
      if (b.kind === "list_item") {
        if (!list) { list = el("ul"); body.append(list); }
        node = el("li");
        list.append(node);
      } else {
        list = null;
        const tag = b.kind === "title" ? "h1" : b.kind === "heading" ? (b.level <= 2 ? "h2" : "h3") : "div";
        const cls = b.kind === "table_row" ? "trow" : b.kind === "table_header" ? "thead" : b.kind === "paragraph" ? "para" : null;
        node = el(b.kind === "paragraph" ? "p" : tag, cls);
        body.append(node);
      }
      appendHighlighted(node, text, b.start, hs, he, marks);
    }
    viewer.hidden = false;
    document.querySelector(".layout").classList.add("with-viewer");
    const params = new URLSearchParams({ doc: docId });
    if (hs !== null) { params.set("start", hs); params.set("end", he); }
    history.replaceState(null, "", `/?${params}`);

    // PDFs: show the original page (PDF.js) with the passage highlighted; text view as a tab.
    const canPdf = d.format === "pdf" && window.pdfjsLib;
    $("#doc-tabs").hidden = !canPdf;
    const showText = () => {
      setView("text");
      if (marks.length) marks[0].scrollIntoView({ block: "center", behavior: "smooth" });
      else body.scrollTop = 0;
    };
    if (!canPdf) { showText(); return; }
    const quote = hs !== null ? d.text.slice(hs, he) : "";
    const block = hs !== null ? d.blocks.find((b) => b.start <= hs && hs < b.end) : null;
    viewState.showText = showText;
    viewState.showPdf = () => {
      setView("pdf");
      renderPdf(`/api/documents/${encodeURIComponent(docId)}/file`, quote, block ? block.page : null)
        .catch(() => showText());   // PDF.js failed (offline?) -> extracted text still works
    };
    viewState.showPdf();
  }

  // ---------------------------------------------------------------- original-PDF view (PDF.js)
  const viewState = { showText: null, showPdf: null, renderId: 0 };
  function setView(view) {
    $("#doc-body").hidden = view !== "text";
    $("#doc-pdf").hidden = view !== "pdf";
    document.querySelectorAll("#doc-tabs button").forEach((b) =>
      b.setAttribute("aria-selected", String(b.dataset.view === view)));
  }
  document.querySelectorAll("#doc-tabs button").forEach((b) =>
    b.addEventListener("click", () => (b.dataset.view === "pdf" ? viewState.showPdf : viewState.showText)?.()));

  // Lowercase, drop whitespace, unify quotes/dashes: PDF.js and our extractor split lines differently.
  const squash = (s) => s.toLowerCase().replace(/[\u2018\u2019]/g, "'").replace(/[\u201c\u201d]/g, '"')
    .replace(/[\u2013\u2014]/g, "-").replace(/\s+/g, "");

  // Highlight exactly the characters of `needle` across the text-layer spans (a quote can
  // start or end mid-line). Returns the first highlight element.
  function highlightSpans(spans, needle) {
    let joined = "";
    const ranges = [];   // [start, end) of each span in the squashed text
    const maps = [];     // squashed index -> original character index, per span
    for (const span of spans) {
      const text = span.textContent || "";
      const map = [];
      for (let i = 0; i < text.length; i++) if (!/\s/.test(text[i])) map.push(i);
      const t = squash(text);
      ranges.push([joined.length, joined.length + t.length]);
      maps.push(t.length === map.length ? map : null);   // null: can't map precisely, mark whole span
      joined += t;
    }
    let at = joined.indexOf(needle), len = needle.length;
    if (at < 0 && needle.length > 60) {          // tolerate small extraction differences at the ends
      at = joined.indexOf(needle.slice(0, 60));
      len = at < 0 ? 0 : Math.min(needle.length, joined.length - at);
    }
    if (at < 0) return null;
    let first = null;
    ranges.forEach(([s, e], i) => {
      if (!(e > at && s < at + len && e > s)) return;
      const span = spans[i], text = span.textContent, map = maps[i];
      const a = Math.max(at, s) - s, b = Math.min(at + len, e) - s;
      const from = map ? map[a] : 0, to = map ? map[b - 1] + 1 : text.length;
      const mark = el("span", "hl", text.slice(from, to));
      span.replaceChildren(document.createTextNode(text.slice(0, from)), mark, document.createTextNode(text.slice(to)));
      first = first || mark;
    });
    return first;
  }

  async function renderPdf(url, quote, targetPage) {
    const myRender = ++viewState.renderId;
    const box = $("#doc-pdf");
    box.replaceChildren(el("p", "hint", "Loading PDF\u2026"));
    const lib = window.pdfjsLib;
    lib.GlobalWorkerOptions.workerSrc =
      "/static/vendor/pdfjs-3.11.174/pdf.worker.min.js";
    const pdf = await lib.getDocument(url).promise;
    if (myRender !== viewState.renderId) return;     // another document was opened meanwhile
    box.replaceChildren();
    const width = Math.max(box.clientWidth - 36, 320);
    const needle = quote ? squash(quote) : "";
    let firstMark = null;
    for (let n = 1; n <= pdf.numPages; n++) {
      const page = await pdf.getPage(n);
      const scale = Math.min(2, width / page.getViewport({ scale: 1 }).width);
      const viewport = page.getViewport({ scale });
      const wrap = el("div", "pdf-page");
      wrap.style.width = `${viewport.width}px`;
      wrap.style.height = `${viewport.height}px`;
      wrap.style.setProperty("--scale-factor", scale);
      const ratio = window.devicePixelRatio || 1;
      const canvas = document.createElement("canvas");
      canvas.width = Math.floor(viewport.width * ratio);
      canvas.height = Math.floor(viewport.height * ratio);
      canvas.style.width = `${viewport.width}px`;
      canvas.style.height = `${viewport.height}px`;
      const layer = el("div", "textLayer");
      wrap.append(canvas, layer);
      box.append(el("div", "page", `Page ${n} of ${pdf.numPages}`), wrap);
      await page.render({ canvasContext: canvas.getContext("2d"), viewport,
                          transform: ratio !== 1 ? [ratio, 0, 0, ratio, 0, 0] : null }).promise;
      const textDivs = [];
      await lib.renderTextLayer({ textContentSource: await page.getTextContent(), container: layer,
                                  viewport, textDivs }).promise;
      if (myRender !== viewState.renderId) return;
      if (needle && !firstMark && (!targetPage || targetPage === n)) firstMark = highlightSpans(textDivs, needle);
    }
    if (firstMark) firstMark.scrollIntoView({ block: "center", behavior: "smooth" });
    else box.scrollTop = 0;
  }

  // ---------------------------------------------------------------- dialogs
  document.querySelectorAll("dialog [data-close]").forEach((b) =>
    b.addEventListener("click", () => b.closest("dialog").close())
  );

  async function loadDocs() {
    const r = await fetch("/api/documents");
    state.docs = await r.json();
    renderDocList();
  }

  function renderDocList() {
    const f = $("#doc-filter").value.toLowerCase();
    const ul = $("#doc-list");
    ul.replaceChildren();
    for (const d of state.docs) {
      if (f && !`${d.title} ${d.category} ${d.doc_id}`.toLowerCase().includes(f)) continue;
      const li = el("li");
      const b = el("button");
      b.type = "button";
      const left = el("span");
      left.append(el("span", null, d.title), document.createTextNode(" "), el("span", "cat", d.category));
      b.append(left, el("span", `lc ${d.lifecycle}`, LIFECYCLE_LABEL[d.lifecycle]));
      b.addEventListener("click", () => { $("#docs-dialog").close(); openDoc(d.doc_id); });
      li.append(b);
      ul.append(li);
    }
  }
  $("#doc-filter").addEventListener("input", renderDocList);
  $("#btn-docs").addEventListener("click", async () => { await loadDocs(); $("#docs-dialog").showModal(); });

  $("#btn-upload").addEventListener("click", () => {
    $("#upload-result").textContent = "";
    $("#upload-dialog").showModal();
  });
  $("#upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const out = $("#upload-result");
    const fd = new FormData(e.target);
    for (const [k, v] of [...fd.entries()]) if (v === "") fd.delete(k);
    out.className = "hint";
    out.textContent = "Uploading and indexing…";
    try {
      const r = await fetch("/api/documents", { method: "POST", body: fd });
      const res = await r.json();
      if (!r.ok) throw new Error(typeof res.detail === "string" ? res.detail : JSON.stringify(res.detail));
      out.textContent = `Indexed “${res.title}” (${res.chunks} passages). You can ask about it now.`;
      e.target.reset();
      loadStatus();
    } catch (err) {
      out.className = "hint error";
      out.textContent = `Upload failed: ${err.message}`;
    }
  });

  // ---------------------------------------------------------------- boot
  loadStatus();
  const p = new URLSearchParams(location.search);
  if (p.get("doc")) {
    const s = p.get("start"), e = p.get("end");
    openDoc(p.get("doc"), s !== null ? Number(s) : null, e !== null ? Number(e) : null);
  }
})();
