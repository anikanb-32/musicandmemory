// ── Constants ─────────────────────────────────────────────────────────────────
const OPENAI_URL = "https://api.openai.com/v1/chat/completions";
const LS_KEY     = "mm_api_key";
const LS_PROFILE = "mm_profile";

const PHRASES = [
  "Parsing memory library",
  "Scanning Billboard archives",
  "Mapping reminiscence bump",
  "Curating biographical context",
  "Weaving musical memories",
  "Composing your playlist",
];

const SYSTEM_PROMPT = `You are an expert music therapist. Respond with valid JSON only.`;

function buildPrompt(profile) {
  const bump_start = profile.birth_year + 15;
  const bump_end   = profile.birth_year + 25;
  return `You are a music therapist creating a personalized therapeutic playlist for a dementia patient.

PATIENT PROFILE:
${JSON.stringify(profile, null, 2)}

TASK: Create a ranked playlist of exactly 10 songs and 3 caregiver conversation cards.

GUIDELINES:
1. Focus on their reminiscence bump: ages 15–25 (years ${bump_start}–${bump_end})
2. Prioritize songs relevant to their cultural background and geographic region
3. Link songs to their specific life events where possible
4. Explain concisely why each song matters for this specific patient
5. Caregiver cards should be gentle, open-ended prompts connecting a song to a memory

Respond in this exact JSON format — no markdown, no commentary:
{
  "playlist": [
    {
      "rank": 1,
      "song": "Song Title",
      "artist": "Artist Name",
      "year": 1965,
      "relevance": "Why this song matters for this specific patient"
    }
  ],
  "caregiver_cards": [
    {
      "song": "Song Title",
      "prompt": "A gentle question or statement linking this song to the patient's life"
    }
  ]
}`;
}

// ── API key storage ────────────────────────────────────────────────────────────
function getApiKey()        { return localStorage.getItem(LS_KEY) || ""; }
function setApiKey(key)     { localStorage.setItem(LS_KEY, key); }
function clearApiKey()      { localStorage.removeItem(LS_KEY); }

// ── View switching ─────────────────────────────────────────────────────────────
function showView(id) {
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  document.getElementById(id).classList.add("active");
}

// ── LocalStorage profile ───────────────────────────────────────────────────────
function saveProfile(profile) { localStorage.setItem(LS_PROFILE, JSON.stringify(profile)); }
function loadSavedProfile() {
  try { return JSON.parse(localStorage.getItem(LS_PROFILE)); } catch { return null; }
}

// ── Custom Dropdown ────────────────────────────────────────────────────────────
function closeAllDropdowns() {
  document.querySelectorAll(".dropdown-trigger.open").forEach(t => t.classList.remove("open"));
  document.querySelectorAll(".dropdown-list.open").forEach(l => l.classList.remove("open"));
}
document.addEventListener("click", closeAllDropdowns);

function buildDropdown(selectEl) {
  const options = Array.from(selectEl.options);
  const wrapper = document.createElement("div");
  wrapper.className = "custom-dropdown";

  const trigger = document.createElement("div");
  trigger.className = "dropdown-trigger placeholder";
  trigger.tabIndex = 0;
  trigger.setAttribute("role", "combobox");

  const label = document.createElement("span");
  label.className = "dropdown-label";
  label.textContent = options[0]?.text || "Select";

  const arrowSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  arrowSvg.setAttribute("viewBox", "0 0 16 16");
  arrowSvg.setAttribute("fill", "none");
  arrowSvg.setAttribute("stroke", "currentColor");
  arrowSvg.setAttribute("stroke-width", "2");
  arrowSvg.classList.add("dropdown-arrow");
  const poly = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  poly.setAttribute("points", "4,6 8,10 12,6");
  arrowSvg.appendChild(poly);

  trigger.appendChild(label);
  trigger.appendChild(arrowSvg);

  const list = document.createElement("div");
  list.className = "dropdown-list";
  list.setAttribute("role", "listbox");

  options.slice(1).forEach(opt => {
    const item = document.createElement("div");
    item.className = "dropdown-option";
    item.textContent = opt.text;
    item.dataset.value = opt.value;
    item.setAttribute("role", "option");
    item.addEventListener("click", () => {
      selectEl.value = opt.value;
      label.textContent = opt.text;
      trigger.classList.remove("placeholder");
      list.querySelectorAll(".dropdown-option").forEach(o => o.classList.remove("selected"));
      item.classList.add("selected");
      trigger.classList.remove("open");
      list.classList.remove("open");
    });
    list.appendChild(item);
  });

  trigger.addEventListener("click", (e) => {
    e.stopPropagation();
    if (list.classList.contains("open")) {
      trigger.classList.remove("open");
      list.classList.remove("open");
    } else {
      closeAllDropdowns();
      trigger.classList.add("open");
      list.classList.add("open");
    }
  });
  list.addEventListener("click", e => e.stopPropagation());

  trigger.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); trigger.click(); }
    if (e.key === "Escape") { trigger.classList.remove("open"); list.classList.remove("open"); }
  });

  wrapper.appendChild(trigger);
  wrapper.appendChild(list);
  selectEl.classList.add("hidden-select");
  selectEl.parentNode.insertBefore(wrapper, selectEl);

  function setValue(val) {
    selectEl.value = val;
    const item = list.querySelector(`[data-value="${CSS.escape(val)}"]`);
    if (item) {
      label.textContent = item.textContent;
      trigger.classList.remove("placeholder");
      list.querySelectorAll(".dropdown-option").forEach(o => o.classList.remove("selected"));
      item.classList.add("selected");
    }
  }
  return { setValue };
}

// ── Custom Number Input ────────────────────────────────────────────────────────
function buildNumberInput(inputEl) {
  const wrapper = document.createElement("div");
  wrapper.className = "num-wrap";

  const btnDec = document.createElement("button");
  btnDec.type = "button";
  btnDec.className = "num-btn";
  btnDec.textContent = "−";
  btnDec.addEventListener("click", () => {
    const min = parseInt(inputEl.min, 10);
    const val = parseInt(inputEl.value, 10);
    if (!isNaN(val) && (isNaN(min) || val > min)) inputEl.value = val - 1;
  });

  const btnInc = document.createElement("button");
  btnInc.type = "button";
  btnInc.className = "num-btn";
  btnInc.textContent = "+";
  btnInc.addEventListener("click", () => {
    const max = parseInt(inputEl.max, 10);
    const val = parseInt(inputEl.value, 10);
    if (!isNaN(val) && (isNaN(max) || val < max)) inputEl.value = val + 1;
  });

  inputEl.parentNode.insertBefore(wrapper, inputEl);
  wrapper.appendChild(btnDec);
  wrapper.appendChild(inputEl);
  wrapper.appendChild(btnInc);
}

// ── Init custom inputs ─────────────────────────────────────────────────────────
const genderDropdown  = buildDropdown(document.getElementById("gender"));
const cultureDropdown = buildDropdown(document.getElementById("cultural_background"));
buildNumberInput(document.getElementById("birth_year"));

// ── Life Events ────────────────────────────────────────────────────────────────
function addEventRow(year = "", event = "") {
  const list = document.getElementById("life-events-list");
  const row = document.createElement("div");
  row.className = "life-event-row";
  row.innerHTML = `
    <input type="number" placeholder="Year" min="1920" max="2000" value="${year}" class="event-year" />
    <input type="text" placeholder="Moved to Atlanta, Georgia" value="${event}" class="event-desc" />
    <button class="btn-remove" title="Remove">✕</button>
  `;
  row.querySelector(".btn-remove").addEventListener("click", () => row.remove());
  list.appendChild(row);
}

document.getElementById("add-event").addEventListener("click", () => addEventRow());

// ── Restore saved profile ──────────────────────────────────────────────────────
const saved = loadSavedProfile();
if (saved) {
  document.getElementById("name").value       = saved.name || "";
  document.getElementById("birth_year").value = saved.birth_year || "";
  document.getElementById("hometown").value   = saved.hometown || "";
  if (saved.gender)              genderDropdown.setValue(saved.gender);
  if (saved.cultural_background) cultureDropdown.setValue(saved.cultural_background);
  (saved.life_events || []).forEach(e => addEventRow(e.year, e.event));
  if (!saved.life_events?.length) { addEventRow(); addEventRow(); }
} else {
  addEventRow(); addEventRow(); addEventRow();
}

// ── API Key Setup view ─────────────────────────────────────────────────────────
function updateApiKeyStatus() {
  const key = getApiKey();
  const statusEl = document.getElementById("api-key-status");
  if (statusEl) statusEl.textContent = key ? "API key set" : "";
}

document.getElementById("setup-btn").addEventListener("click", () => {
  const key   = document.getElementById("setup-key").value.trim();
  const errEl = document.getElementById("setup-error");
  if (!key.startsWith("sk-")) {
    errEl.textContent = "Please enter a valid OpenAI API key (starts with sk-).";
    errEl.hidden = false;
    return;
  }
  setApiKey(key);
  updateApiKeyStatus();
  errEl.hidden = true;
  showView("view-form");
});

document.getElementById("setup-key").addEventListener("keydown", (e) => {
  if (e.key === "Enter") document.getElementById("setup-btn").click();
});

document.getElementById("change-key-btn").addEventListener("click", () => {
  clearApiKey();
  document.getElementById("setup-key").value = "";
  document.getElementById("setup-error").hidden = true;
  showView("view-setup");
});

// Show setup if no key, otherwise show form
if (getApiKey()) {
  updateApiKeyStatus();
  showView("view-form");
}

// ── Collect form ───────────────────────────────────────────────────────────────
function getProfile() {
  const name                = document.getElementById("name").value.trim();
  const gender              = document.getElementById("gender").value;
  const birth_year          = parseInt(document.getElementById("birth_year").value, 10);
  const hometown            = document.getElementById("hometown").value.trim();
  const cultural_background = document.getElementById("cultural_background").value;

  const life_events = [];
  document.querySelectorAll(".life-event-row").forEach(row => {
    const year  = parseInt(row.querySelector(".event-year").value, 10);
    const event = row.querySelector(".event-desc").value.trim();
    if (!isNaN(year) && event) life_events.push({ year, event });
  });

  const errors = [];
  if (!name)                errors.push("Patient name is required.");
  if (!gender)              errors.push("Sex is required.");
  if (!birth_year || birth_year < 1920 || birth_year > 1975) errors.push("Enter a valid birth year (1920–1975).");
  if (!hometown)            errors.push("Hometown is required.");
  if (!cultural_background) errors.push("Cultural background is required.");
  if (!life_events.length)  errors.push("Add at least one life event.");
  if (errors.length) throw new Error(errors.join(" "));

  return { name, gender, birth_year, hometown, cultural_background, life_events };
}

// ── Loading phrase animation ───────────────────────────────────────────────────
let phraseTimer = null;

function startPhraseLoop() {
  const textEl = document.getElementById("phrase-text");
  let phraseIdx = 0;
  let cancelled = false;

  function schedule(fn, delay) { phraseTimer = setTimeout(fn, delay); }

  function runPhrase() {
    if (cancelled) return;
    const phrase = PHRASES[phraseIdx % PHRASES.length];
    phraseIdx++;
    textEl.style.transition = "none";
    textEl.style.opacity = "1";
    textEl.offsetHeight;
    textEl.textContent = "";
    let charIdx = 0;

    function typeNext() {
      if (cancelled) return;
      if (charIdx < phrase.length) {
        textEl.textContent = phrase.slice(0, charIdx + 1);
        charIdx++;
        schedule(typeNext, 50);
      } else {
        schedule(() => {
          if (cancelled) return;
          textEl.style.transition = "opacity 0.45s ease";
          textEl.style.opacity = "0";
          schedule(runPhrase, 520);
        }, 1100);
      }
    }
    typeNext();
  }

  runPhrase();
  return function stop() {
    cancelled = true;
    if (phraseTimer) { clearTimeout(phraseTimer); phraseTimer = null; }
  };
}

let stopPhrases = null;
function stopPhraseLoop() {
  if (stopPhrases) { stopPhrases(); stopPhrases = null; }
}

// ── Call OpenAI directly ───────────────────────────────────────────────────────
async function callOpenAI(profile) {
  const apiKey = getApiKey();
  if (!apiKey) throw new Error("No API key set.");

  const res = await fetch(OPENAI_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: "gpt-4o",
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user",   content: buildPrompt(profile) },
      ],
      temperature: 0.4,
    }),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const msg  = body?.error?.message || `OpenAI returned ${res.status}`;
    if (res.status === 401) throw new Error("Invalid API key. Click 'Change API Key' to update it.");
    if (res.status === 429) throw new Error("Rate limit or quota exceeded. Try again shortly.");
    throw new Error(msg);
  }

  const data = await res.json();
  let text = data.choices[0].message.content.trim();
  text = text.replace(/^```json\s*/i, "").replace(/\s*```$/, "").trim();
  return JSON.parse(text);
}

// ── Generate ───────────────────────────────────────────────────────────────────
document.getElementById("generate-btn").addEventListener("click", async () => {
  const errEl = document.getElementById("error-msg");
  errEl.hidden = true;

  let profile;
  try { profile = getProfile(); }
  catch (e) {
    errEl.textContent = e.message;
    errEl.hidden = false;
    return;
  }

  saveProfile(profile);
  showView("view-loading");
  stopPhrases = startPhraseLoop();

  try {
    const data = await callOpenAI(profile);
    stopPhraseLoop();
    lastResults = { profile, data };
    renderResults(profile, data);
    showView("view-results");
  } catch (e) {
    stopPhraseLoop();
    showView("view-form");
    errEl.textContent = `Error: ${e.message}`;
    errEl.hidden = false;
  }
});

// ── Back ───────────────────────────────────────────────────────────────────────
let lastResults = null;

function goToForm() {
  document.getElementById("back-to-playlist-btn").hidden = lastResults === null;
  showView("view-form");
}

document.getElementById("back-btn").addEventListener("click", goToForm);

document.getElementById("back-to-playlist-btn").addEventListener("click", () => {
  if (lastResults) {
    renderResults(lastResults.profile, lastResults.data);
    showView("view-results");
  }
});

// ── Render results ─────────────────────────────────────────────────────────────
function renderResults(profile, data) {
  const bump_start = profile.birth_year + 15;
  const bump_end   = profile.birth_year + 25;

  const summaryEl = document.getElementById("profile-summary");
  const eventsHtml = profile.life_events.map(e =>
    `<span>${e.year} — ${escHtml(e.event)}</span>`
  ).join("  ·  ");

  summaryEl.innerHTML = `
    <div class="profile-pre">A playlist personalized for</div>
    <div class="profile-name">${escHtml(profile.name)}</div>
    <div class="profile-tags">
      <span class="profile-tag">${escHtml(profile.gender)}</span>
      <span class="profile-tag">b. ${profile.birth_year}</span>
      <span class="profile-tag">${escHtml(profile.cultural_background)}</span>
      <span class="profile-tag">${escHtml(profile.hometown)}</span>
      <span class="profile-tag">Reminiscence bump ${bump_start}–${bump_end}</span>
    </div>
    ${eventsHtml ? `<div class="profile-events">${eventsHtml}</div>` : ""}
  `;

  const playlistEl = document.getElementById("playlist-cards");
  playlistEl.innerHTML = "";

  const count = (data.playlist || []).length;
  const totalMin = Math.round(count * 3.6);
  const durationStr = totalMin >= 60
    ? `${Math.floor(totalMin / 60)} hr ${totalMin % 60} min`
    : `${totalMin} min`;
  const playlistMeta = document.querySelector(".playlist-meta");
  if (playlistMeta) playlistMeta.textContent = `${count} songs · about ${durationStr}`;

  (data.playlist || []).forEach((song, i) => {
    const card = document.createElement("div");
    card.className = "song-card fade-in";
    card.style.animationDelay = `${i * 0.04}s`;
    card.innerHTML = `
      <div class="song-rank">${song.rank}</div>
      <div>
        <div class="song-title">${escHtml(song.song)}</div>
        <div class="song-artist">${escHtml(song.artist)} · ${song.year}</div>
        ${song.relevance ? `<div class="song-relevance">${escHtml(song.relevance)}</div>` : ""}
      </div>
    `;
    playlistEl.appendChild(card);
  });

  const caregiverEl = document.getElementById("caregiver-cards");
  caregiverEl.innerHTML = "";
  (data.caregiver_cards || []).forEach((card, i) => {
    const el = document.createElement("div");
    el.className = "caregiver-card fade-in";
    el.style.animationDelay = `${i * 0.07 + 0.2}s`;
    el.innerHTML = `
      <div class="caregiver-song">${escHtml(card.song)}</div>
      <div class="caregiver-prompt">${escHtml(card.prompt)}</div>
    `;
    caregiverEl.appendChild(el);
  });
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
