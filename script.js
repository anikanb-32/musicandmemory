const API_URL = "http://localhost:5050/generate";

const PHRASES = [
  "Parsing memory library",
  "Scanning Billboard archives",
  "Mapping reminiscence bump",
  "Curating biographical context",
  "Weaving musical memories",
  "Composing your playlist",
];

// ── View switching ────────────────────────────────────────────────────────────
function showView(id) {
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  document.getElementById(id).classList.add("active");
}

// ── LocalStorage profile ──────────────────────────────────────────────────────
function saveProfile(profile) {
  localStorage.setItem("mm_profile", JSON.stringify(profile));
}

function loadSavedProfile() {
  try { return JSON.parse(localStorage.getItem("mm_profile")); } catch { return null; }
}

// ── Custom Dropdown ───────────────────────────────────────────────────────────
// Close all open dropdowns (used by document click and when opening a new one)
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

  // stopPropagation on trigger so the document listener doesn't immediately close
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

  // stopPropagation inside the list so clicks on options don't bubble to document
  list.addEventListener("click", e => e.stopPropagation());

  trigger.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      trigger.click();
    }
    if (e.key === "Escape") {
      trigger.classList.remove("open");
      list.classList.remove("open");
    }
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

// ── Custom Number Input ───────────────────────────────────────────────────────
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

// ── Init custom inputs ────────────────────────────────────────────────────────
const genderDropdown  = buildDropdown(document.getElementById("gender"));
const cultureDropdown = buildDropdown(document.getElementById("cultural_background"));
buildNumberInput(document.getElementById("birth_year"));

// ── Life Events ───────────────────────────────────────────────────────────────
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

// ── Restore saved profile or seed blank rows ──────────────────────────────────
const saved = loadSavedProfile();
if (saved) {
  document.getElementById("name").value      = saved.name || "";
  document.getElementById("birth_year").value = saved.birth_year || "";
  document.getElementById("hometown").value   = saved.hometown || "";
  if (saved.gender)              genderDropdown.setValue(saved.gender);
  if (saved.cultural_background) cultureDropdown.setValue(saved.cultural_background);
  (saved.life_events || []).forEach(e => addEventRow(e.year, e.event));
  if (!saved.life_events?.length) { addEventRow(); addEventRow(); }
} else {
  addEventRow(); addEventRow(); addEventRow();
}

// ── Collect form ──────────────────────────────────────────────────────────────
function getProfile() {
  const name               = document.getElementById("name").value.trim();
  const gender             = document.getElementById("gender").value;
  const birth_year         = parseInt(document.getElementById("birth_year").value, 10);
  const hometown           = document.getElementById("hometown").value.trim();
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

// ── Loading phrase animation (type → pause → fade → next) ────────────────────
let phraseTimer = null;

function startPhraseLoop() {
  const textEl = document.getElementById("phrase-text");
  let phraseIdx = 0;
  let cancelled = false;

  function schedule(fn, delay) {
    phraseTimer = setTimeout(fn, delay);
  }

  function runPhrase() {
    if (cancelled) return;
    const phrase = PHRASES[phraseIdx % PHRASES.length];
    phraseIdx++;

    // Snap to visible with no transition
    textEl.style.transition = "none";
    textEl.style.opacity = "1";
    textEl.offsetHeight; // force reflow so the snap takes effect before typing starts
    textEl.textContent = "";

    let charIdx = 0;

    function typeNext() {
      if (cancelled) return;
      if (charIdx < phrase.length) {
        textEl.textContent = phrase.slice(0, charIdx + 1);
        charIdx++;
        schedule(typeNext, 50);
      } else {
        // Pause, then fade out
        schedule(() => {
          if (cancelled) return;
          textEl.style.transition = "opacity 0.45s ease";
          textEl.style.opacity = "0";
          // After fade, start next phrase
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

// ── Generate ──────────────────────────────────────────────────────────────────
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
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(profile),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: "Server error" }));
      throw new Error(err.error || `Server returned ${res.status}`);
    }
    const data = await res.json();
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

// ── Back / Edit ───────────────────────────────────────────────────────────────
let lastResults = null; // { profile, data } — kept so back-to-playlist can re-render

function goToForm() {
  document.getElementById("back-to-playlist-btn").hidden = lastResults === null;
  showView("view-form");
}

document.getElementById("back-btn").addEventListener("click", goToForm);
document.getElementById("edit-btn").addEventListener("click", goToForm);

document.getElementById("back-to-playlist-btn").addEventListener("click", () => {
  if (lastResults) {
    renderResults(lastResults.profile, lastResults.data);
    showView("view-results");
  }
});

// ── Render results ────────────────────────────────────────────────────────────
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
