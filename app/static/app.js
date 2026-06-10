const form = document.getElementById("ask-form");
const input = document.getElementById("question");
const sendBtn = document.getElementById("send-btn");
const answerEl = document.getElementById("answer");
const sourcesEl = document.getElementById("sources");
const feedbackEl = document.getElementById("feedback");
const thanksEl = document.getElementById("feedback-thanks");

let lastQuestion = "";
let lastAnswer = "";

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  askQuestion(question);
});

function askQuestion(question) {
  lastQuestion = question;
  lastAnswer = "";
  answerEl.textContent = "";
  answerEl.classList.remove("error");
  sourcesEl.innerHTML = "";
  feedbackEl.hidden = true;
  thanksEl.hidden = true;
  sendBtn.disabled = true;

  const es = new EventSource("/chat/stream?q=" + encodeURIComponent(question));

  es.addEventListener("token", (ev) => {
    lastAnswer += ev.data;
    answerEl.textContent = lastAnswer;
  });

  es.addEventListener("sources", (ev) => {
    const sources = JSON.parse(ev.data);
    renderSources(sources);
  });

  es.addEventListener("done", () => {
    es.close();
    sendBtn.disabled = false;
    if (lastAnswer.trim()) feedbackEl.hidden = false;
  });

  es.addEventListener("error", (ev) => {
    es.close();
    sendBtn.disabled = false;
    // ev.data is set for our explicit error event; network errors have none.
    answerEl.textContent = ev.data || "Something went wrong. Please try again.";
    answerEl.classList.add("error");
  });
}

function renderSources(sources) {
  if (!sources.length) {
    sourcesEl.innerHTML = '<p class="sources-empty">No external sources for this answer.</p>';
    return;
  }
  sourcesEl.innerHTML = sources.map((s) => `
    <div class="source-item">
      <div class="name">${escapeHtml(s.source_name || s.topic || "Source")}</div>
      <a href="${escapeHtml(s.source_url)}" target="_blank" rel="noopener">${escapeHtml(s.source_url)}</a>
      <div class="date">retrieved ${escapeHtml(s.fetched_at || "")}</div>
    </div>
  `).join("");
}

feedbackEl.addEventListener("click", (e) => {
  const btn = e.target.closest(".thumb");
  if (!btn) return;
  fetch("/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: lastQuestion, answer: lastAnswer, rating: btn.dataset.rating }),
  });
  thanksEl.hidden = false;
});

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
