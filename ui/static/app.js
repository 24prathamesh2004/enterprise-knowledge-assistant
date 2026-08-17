const form = document.getElementById("ask-form");
const questionEl = document.getElementById("question");
const submitBtn = document.getElementById("submit-btn");
const resultEl = document.getElementById("result");
const answerEl = document.getElementById("answer");
const summaryEl = document.getElementById("summary");
const scoresEl = document.getElementById("scores");
const traceEl = document.getElementById("trace");
const escalationEl = document.getElementById("escalation");
const loadingEl = document.getElementById("loading");
const errorEl = document.getElementById("error");

function scoreClass(v) {
  if (v >= 0.8) return "good";
  if (v >= 0.6) return "warn";
  return "bad";
}

function renderScores(scores) {
  scoresEl.innerHTML = "";
  const entries = Object.entries(scores || {}).filter(
    ([k]) => !k.startsWith("_")
  );
  if (!entries.length) {
    scoresEl.textContent = "(no scores)";
    return;
  }
  for (const [key, val] of entries) {
    const num = typeof val === "number" ? val : NaN;
    const div = document.createElement("div");
    div.className = "score";
    const label = key.replace(/_/g, " ");
    const shown = Number.isNaN(num) ? "n/a" : num.toFixed(3);
    const cls = Number.isNaN(num) ? "" : scoreClass(num);
    div.innerHTML = `<span class="label">${label}</span><span class="val ${cls}">${shown}</span>`;
    scoresEl.appendChild(div);
  }
}

function renderTrace(trace) {
  traceEl.innerHTML = "";
  for (const node of trace || []) {
    const span = document.createElement("span");
    span.className = "node";
    span.textContent = node;
    traceEl.appendChild(span);
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionEl.value.trim();
  if (!question) return;

  submitBtn.disabled = true;
  errorEl.classList.add("hidden");
  resultEl.classList.add("hidden");
  loadingEl.classList.remove("hidden");

  try {
    const resp = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!resp.ok) throw new Error(`Server returned ${resp.status}`);
    const data = await resp.json();

    answerEl.textContent = data.answer || "(no answer)";
    summaryEl.textContent = data.evaluation_summary || "";
    renderScores(data.scores);
    renderTrace(data.trace);

    if (data.escalated) {
      escalationEl.textContent = `Escalated to a human — ticket ${data.ticket_id}`;
      escalationEl.classList.remove("hidden");
    } else {
      escalationEl.classList.add("hidden");
    }

    resultEl.classList.remove("hidden");
  } catch (err) {
    errorEl.textContent = `Error: ${err.message}`;
    errorEl.classList.remove("hidden");
  } finally {
    loadingEl.classList.add("hidden");
    submitBtn.disabled = false;
  }
});
