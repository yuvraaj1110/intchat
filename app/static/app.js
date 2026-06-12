const form = document.getElementById("ask-form");
const input = document.getElementById("question");
const sendBtn = document.getElementById("send-btn");
const thread = document.getElementById("thread");

let busy = false;

// Example chips fill the input and submit immediately.
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    if (busy) return;
    input.value = chip.textContent;
    form.requestSubmit();
  });
});

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question || busy) return;
  input.value = "";
  ask(question);
});

function ask(question) {
  busy = true;
  sendBtn.disabled = true;
  removeWelcome();
  addUserMessage(question);

  const bot = addBotMessage();            // returns handles to its parts
  let answer = "";
  let gotToken = false;

  const es = new EventSource("/chat/stream?q=" + encodeURIComponent(question));

  es.addEventListener("token", (ev) => {
    if (!gotToken) { bot.clearTyping(); gotToken = true; }
    answer += ev.data;
    bot.textEl.textContent = answer;
    scrollDown();
  });

  es.addEventListener("sources", (ev) => {
    const sources = JSON.parse(ev.data);
    bot.renderSources(sources);
  });

  es.addEventListener("done", () => {
    es.close();
    finish();
    if (answer.trim()) bot.addFeedback(question, answer);
    scrollDown();
  });

  es.addEventListener("error", (ev) => {
    es.close();
    bot.clearTyping();
    bot.bubble.classList.add("error");
    bot.textEl.textContent = ev.data || "Something went wrong. Please try again.";
    finish();
  });

  function finish() {
    busy = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

function removeWelcome() {
  const w = document.getElementById("welcome");
  if (w) w.remove();
}

function addUserMessage(text) {
  const msg = el("div", "msg user");
  const bubble = el("div", "bubble");
  bubble.textContent = text;
  msg.appendChild(bubble);
  thread.appendChild(msg);
  scrollDown();
}

function addBotMessage() {
  const msg = el("div", "msg bot");
  const bubble = el("div", "bubble");
  // typing indicator
  const typing = el("div", "dots");
  typing.innerHTML = "<span></span><span></span><span></span>";
  bubble.appendChild(typing);
  const textEl = el("span");
  msg.appendChild(bubble);
  thread.appendChild(msg);
  scrollDown();

  return {
    bubble,
    textEl,
    clearTyping() {
      bubble.innerHTML = "";
      bubble.appendChild(textEl);
    },
    renderSources(sources) {
      if (!sources || !sources.length) return;
      const wrap = el("div", "sources");
      const toggle = el("button", "sources-toggle");
      toggle.innerHTML =
        `<span class="caret">▸</span> ${sources.length} source${sources.length > 1 ? "s" : ""}`;
      const list = el("div", "sources-list");
      sources.forEach((s) => {
        const item = el("div", "source-item");
        const name = el("div", "name");
        name.textContent = s.source_name || s.topic || "Source";
        const link = el("a");
        link.href = s.source_url;
        link.target = "_blank";
        link.rel = "noopener";
        link.textContent = s.source_url;
        const date = el("div", "date");
        date.textContent = s.fetched_at ? "retrieved " + s.fetched_at : "";
        item.append(name, link, date);
        list.appendChild(item);
      });
      toggle.addEventListener("click", () => wrap.classList.toggle("open"));
      wrap.append(toggle, list);
      bubble.appendChild(wrap);
    },
    addFeedback(question, answer) {
      const fb = el("div", "fb");
      const up = el("button"); up.textContent = "👍";
      const down = el("button"); down.textContent = "👎";
      const thanks = el("span", "thanks");
      const send = (rating) => {
        fetch("/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, answer, rating }),
        });
        up.disabled = down.disabled = true;
        thanks.textContent = "Thanks!";
      };
      up.addEventListener("click", () => send("up"));
      down.addEventListener("click", () => send("down"));
      fb.append(up, down, thanks);
      bubble.appendChild(fb);
    },
  };
}

function el(tag, className) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}

function scrollDown() {
  thread.scrollTop = thread.scrollHeight;
}
