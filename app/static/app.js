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
  document.body.classList.add("chatting");   // hide campus background + college UI
  stopSlideshow();
  removeWelcome();
  addUserTurn(question);

  const bot = addBotTurn();
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
    bot.renderSources(JSON.parse(ev.data));
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
    bot.contentEl.classList.add("error");
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

function addUserTurn(text) {
  const turn = el("div", "turn user");
  const role = el("div", "role");
  const av = el("span", "avatar you"); av.textContent = "Y";
  role.append(av, document.createTextNode(" You"));
  const content = el("div", "content");
  content.textContent = text;
  turn.append(role, content);
  thread.appendChild(turn);
  scrollDown();
}

function addBotTurn() {
  const turn = el("div", "turn bot");

  const role = el("div", "role");
  const av = el("span", "avatar bot"); av.textContent = "✦";
  role.append(av, document.createTextNode(" Assistant"));

  const contentEl = el("div", "content");
  const typing = el("div", "dots");
  typing.innerHTML = "<span></span><span></span><span></span>";
  contentEl.appendChild(typing);

  const textEl = el("span");

  turn.append(role, contentEl);
  thread.appendChild(turn);
  scrollDown();

  return {
    turn,
    contentEl,
    textEl,
    clearTyping() {
      contentEl.innerHTML = "";
      contentEl.appendChild(textEl);
    },
    renderSources(sources) {
      if (!sources || !sources.length) return;
      const wrap = el("div", "sources open");  // expanded by default (boxes shown)
      const toggle = el("button", "src-toggle");
      toggle.innerHTML =
        `<span class="caret">▸</span> ${sources.length} source${sources.length > 1 ? "s" : ""}`;
      const list = el("div", "src-list");
      sources.forEach((s) => {
        const item = el("div", "src");
        const name = el("div", "n");
        name.textContent = s.source_name || s.topic || "Source";
        const link = el("a");
        link.href = s.source_url; link.target = "_blank"; link.rel = "noopener";
        link.textContent = s.source_url;
        const date = el("div", "d");
        date.textContent = s.fetched_at ? "retrieved " + s.fetched_at : "";
        item.append(name, link, date);
        list.appendChild(item);
      });
      toggle.addEventListener("click", () => wrap.classList.toggle("open"));
      wrap.append(toggle, list);
      turn.appendChild(wrap);
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
      turn.appendChild(fb);
    },
  };
}

function el(tag, className) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str == null ? "" : str;
  return div.innerHTML;
}

function scrollDown() {
  thread.scrollTop = thread.scrollHeight;
}


/* ── Campus-photo background slideshow + "Is this your college?" ───────── */
let slideshowTimer = null;
let stopSlideshow = () => {};

(function collegeBackground() {
  const stage = document.getElementById("stage");
  const layers = [document.getElementById("bg-a"), document.getElementById("bg-b")];
  const nameEl = document.getElementById("college-name");
  const btn = document.getElementById("college-btn");
  const reveal = document.getElementById("college-reveal");
  if (!stage || !btn) return;

  let colleges = [];
  let idx = 0;
  let active = 0;
  let current = null;
  let locked = false;   // once the user picks their college, keep the reveal for the session

  stopSlideshow = () => {
    if (slideshowTimer) clearInterval(slideshowTimer);
    slideshowTimer = null;
  };

  function shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }

  function showCollege(college) {
    // Preload, then crossfade onto the inactive layer.
    const img = new Image();
    img.onload = () => {
      const next = layers[1 - active];
      next.style.backgroundImage = `url("${college.image}")`;
      next.classList.add("show");
      layers[active].classList.remove("show");
      active = 1 - active;
      current = college;
      // Keep the reveal pinned once the user has picked their college;
      // hide the cycling caption so it doesn't conflict with the pinned reveal.
      if (locked) {
        nameEl.classList.remove("show");
      } else {
        nameEl.textContent = college.name;
        nameEl.classList.add("show");
        reveal.hidden = true;
        btn.style.display = "";
      }
    };
    img.onerror = nextCollege;  // skip broken images
    img.src = college.image;
  }

  function nextCollege() {
    if (!colleges.length) return;
    idx = (idx + 1) % colleges.length;
    showCollege(colleges[idx]);
  }

  btn.addEventListener("click", () => {
    if (!current) return;
    // Lock in the user's college for the whole session — the reveal stays,
    // the button never comes back. Background photos keep cycling behind it.
    locked = true;
    reveal.innerHTML =
      `🎉 Oh, so you're a ${escapeHtml(current.demonym)}!` +
      `<span class="sub">${escapeHtml(current.name)}</span>`;
    reveal.hidden = false;
    btn.style.display = "none";
  });

  fetch("/static/colleges.json")
    .then((r) => r.json())
    .then((data) => {
      if (!data || !data.length) { stage.style.display = "none"; return; }
      colleges = shuffle(data.slice());
      idx = 0;
      showCollege(colleges[0]);
      slideshowTimer = setInterval(nextCollege, 7000);
    })
    .catch(() => { stage.style.display = "none"; });
})();
