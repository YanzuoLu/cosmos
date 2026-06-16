(function () {
  "use strict";

  const HF_BASE = "https://huggingface.co/datasets/oliveryanzuolu/cosmos3-t2v-release/resolve/main/compare";
  const DEFAULT_IDS = ["039", "048", "049", "079"];
  const FINAL8_IDS = ["006", "014", "028", "039", "048", "049", "068", "079"];
  const VIDEO_TIMEOUT_MS = 18000;
  const STALLED_TIMEOUT_MS = 4500;

  let selectedIds = [...DEFAULT_IDS];
  let armedId = null;
  let videoObserver = null;
  const videoTimers = new WeakMap();

  const posterSvg = [
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 2560 720'>",
    "<defs><linearGradient id='g' x1='0' x2='1'><stop stop-color='#11141a'/><stop offset='1' stop-color='#0b0d10'/></linearGradient></defs>",
    "<rect width='2560' height='720' fill='url(#g)'/>",
    "<path d='M0 719H2560' stroke='rgba(255,255,255,0.08)'/>",
    "<text x='1280' y='360' fill='#8b96a5' font-family='ui-monospace,Menlo,monospace' font-size='42' text-anchor='middle'>comparison video</text>",
    "</svg>"
  ].join("");
  const POSTER_URL = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(posterSvg);

  document.addEventListener("DOMContentLoaded", () => {
    initPromptPicker();
    initCopyButtons();
    renderVideos();
    initNavHighlighter();
    renderTimeBreakdownChart();
  });

  function buildVideoUrl(group, id) {
    return `${HF_BASE}/${group}/prompt_${id}.mp4`;
  }

  function initPromptPicker() {
    const chipButtons = Array.from(document.querySelectorAll(".prompt-chip"));
    const form = document.getElementById("customPromptForm");
    const input = document.getElementById("customPromptInput");

    chipButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const id = button.dataset.promptId;
        if (!id) return;

        if (selectedIds.includes(id)) {
          // Active chip: arm it (mark to swap out), or un-arm if tapped again.
          armedId = armedId === id ? null : id;
          setPromptMessage(
            armedId
              ? `${id} marked to swap out — now tap an inactive prompt to swap it in.`
              : "Swap cancelled."
          );
          syncPromptUi();
          return;
        }

        // Inactive chip: it replaces the armed active chip (explicit swap).
        if (!armedId) {
          setPromptMessage("Tap one of the 4 active prompts first to mark it for replacement.");
          pulseActiveChips();
          return;
        }

        const swappedOut = armedId;
        selectedIds = selectedIds.map((selected) => (selected === armedId ? id : selected));
        armedId = null;
        input.removeAttribute("aria-invalid");
        setPromptMessage(`Swapped ${swappedOut} → ${id}.`);
        syncPromptUi();
        renderVideos();
      });
    });

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const parsed = parsePromptIds(input.value);

      if (!parsed.ok) {
        input.setAttribute("aria-invalid", "true");
        setPromptMessage(parsed.message, true);
        return;
      }

      selectedIds = parsed.ids;
      armedId = null;
      input.removeAttribute("aria-invalid");
      setPromptMessage("Applied custom prompt ids.");
      syncPromptUi();
      renderVideos();
    });

    syncPromptUi();
  }

  function parsePromptIds(rawValue) {
    const parts = rawValue
      .split(/[,\s]+/)
      .map((part) => part.trim())
      .filter(Boolean);

    if (parts.length < 4) {
      return { ok: false, message: "Enter at least four ids from 001 to 100." };
    }

    const ids = [];
    const seen = new Set();

    for (const part of parts) {
      if (!/^\d{1,3}$/.test(part)) {
        return { ok: false, message: `Invalid id "${part}". Use 001 to 100.` };
      }

      const numberValue = Number(part);
      if (numberValue < 1 || numberValue > 100) {
        return { ok: false, message: `Out of range "${part}". Use 001 to 100.` };
      }

      const normalized = String(numberValue).padStart(3, "0");
      if (!seen.has(normalized)) {
        ids.push(normalized);
        seen.add(normalized);
      }
    }

    if (ids.length < 4) {
      return { ok: false, message: "Need four unique valid ids." };
    }

    return { ok: true, ids: ids.slice(0, 4) };
  }

  function syncPromptUi() {
    document.querySelectorAll(".prompt-chip").forEach((button) => {
      const id = button.dataset.promptId;
      const isActive = selectedIds.includes(id);
      button.classList.toggle("is-active", isActive);
      button.classList.toggle("is-armed", id === armedId);
      button.setAttribute("aria-pressed", isActive ? "true" : "false");
    });

    const output = document.getElementById("selectedPromptOutput");
    if (!output) return;
    output.replaceChildren(
      ...selectedIds.map((id) => {
        const token = document.createElement("span");
        token.className = "selected-token";
        token.textContent = id;
        return token;
      })
    );
  }

  function setPromptMessage(message, isError) {
    const status = document.getElementById("promptStatus");
    if (!status) return;
    status.textContent = message;
    status.classList.toggle("is-error", Boolean(isError));
  }

  function pulseActiveChips() {
    document.querySelectorAll(".prompt-chip.is-active").forEach((chip) => {
      chip.classList.remove("pulse");
      void chip.offsetWidth;
      chip.classList.add("pulse");
      window.setTimeout(() => chip.classList.remove("pulse"), 520);
    });
  }

  function renderVideos() {
    if (videoObserver) {
      videoObserver.disconnect();
    }

    videoObserver = createVideoObserver();

    document.querySelectorAll("section[data-group]").forEach((section) => {
      const grid = section.querySelector("[data-video-grid]");
      if (!grid) return;

      const group = section.dataset.group;
      const left = section.dataset.left || "LEFT";
      const right = section.dataset.right || "RIGHT";
      grid.replaceChildren();

      selectedIds.forEach((id) => {
        const card = createVideoCard({ group, id, left, right });
        grid.appendChild(card);

        if (videoObserver) {
          videoObserver.observe(card);
        } else {
          loadVideo(card);
        }
      });
    });
  }

  function createVideoCard({ group, id, left, right }) {
    const card = document.createElement("article");
    card.className = "video-card";
    card.dataset.group = group;
    card.dataset.id = id;
    card.dataset.loaded = "false";

    const frame = document.createElement("div");
    frame.className = "video-frame";

    const video = document.createElement("video");
    video.muted = true;
    video.loop = true;
    video.playsInline = true;
    video.preload = "none";
    video.poster = POSTER_URL;
    video.setAttribute("aria-label", `Prompt ${id} ${left} versus ${right}`);

    const placeholder = document.createElement("div");
    placeholder.className = "video-placeholder";
    placeholder.innerHTML = '<span class="placeholder-line"><span class="status-dot" aria-hidden="true"></span><span data-placeholder-text>queued</span></span>';

    frame.append(video, placeholder);

    const meta = document.createElement("div");
    meta.className = "video-meta";

    const prompt = document.createElement("div");
    prompt.className = "prompt-label";
    prompt.textContent = `Prompt ${id}`;

    const caption = document.createElement("div");
    caption.className = "config-label";
    caption.innerHTML = `<span>LEFT: ${escapeHtml(left)}</span><span class="caption-divider">|</span><span>RIGHT: ${escapeHtml(right)}</span>`;

    meta.append(prompt, caption);
    card.append(frame, meta);

    video.addEventListener("loadstart", () => markVideoLoading(card));
    video.addEventListener("loadeddata", () => markVideoReady(card));
    video.addEventListener("canplay", () => markVideoReady(card));
    video.addEventListener("error", () => markVideoUnavailable(card));
    video.addEventListener("stalled", () => scheduleVideoTimeout(card, STALLED_TIMEOUT_MS));
    video.addEventListener("waiting", () => setPlaceholderText(card, "loading..."));

    return card;
  }

  function createVideoObserver() {
    if (!("IntersectionObserver" in window)) {
      return null;
    }

    return new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const card = entry.target;
          const video = card.querySelector("video");
          if (!video) return;

          if (entry.isIntersecting) {
            loadVideo(card);
            if (card.classList.contains("is-loaded")) {
              playVideo(video);
            }
          } else {
            video.pause();
          }
        });
      },
      {
        root: null,
        rootMargin: "260px 0px",
        threshold: 0.08
      }
    );
  }

  function loadVideo(card) {
    if (card.dataset.loaded === "true") {
      const video = card.querySelector("video");
      if (video) playVideo(video);
      return;
    }

    const video = card.querySelector("video");
    if (!video) return;

    card.dataset.loaded = "true";
    video.src = buildVideoUrl(card.dataset.group, card.dataset.id);
    markVideoLoading(card);
    scheduleVideoTimeout(card, VIDEO_TIMEOUT_MS);
    video.load();
    playVideo(video);
  }

  function playVideo(video) {
    const attempt = video.play();
    if (attempt && typeof attempt.catch === "function") {
      attempt.catch(() => {
        return undefined;
      });
    }
  }

  function markVideoLoading(card) {
    card.classList.add("is-loading");
    card.classList.remove("is-loaded", "is-error");
    setPlaceholderText(card, "loading...");
  }

  function markVideoReady(card) {
    clearVideoTimeout(card);
    card.classList.add("is-loaded");
    card.classList.remove("is-loading", "is-error");
  }

  function markVideoUnavailable(card) {
    clearVideoTimeout(card);
    card.classList.add("is-error");
    card.classList.remove("is-loading", "is-loaded");
    setPlaceholderText(card, "uploading...");
  }

  function scheduleVideoTimeout(card, delay) {
    clearVideoTimeout(card);
    const timer = window.setTimeout(() => {
      markVideoUnavailable(card);
    }, delay);
    videoTimers.set(card, timer);
  }

  function clearVideoTimeout(card) {
    const timer = videoTimers.get(card);
    if (timer) {
      window.clearTimeout(timer);
      videoTimers.delete(card);
    }
  }

  function setPlaceholderText(card, text) {
    const target = card.querySelector("[data-placeholder-text]");
    if (target) {
      target.textContent = text;
    }
  }

  function initCopyButtons() {
    document.querySelectorAll("[data-copy]").forEach((button) => {
      button.addEventListener("click", async () => {
        const panel = button.closest(".code-panel");
        const code = panel ? panel.querySelector("code") : null;
        if (!code) return;

        const text = code.textContent.trimEnd();
        const copied = await copyText(text);

        button.textContent = copied ? "Copied" : "Select";
        button.classList.toggle("is-copied", copied);
        window.setTimeout(() => {
          button.textContent = "Copy";
          button.classList.remove("is-copied");
        }, 1400);
      });
    });
  }

  async function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(text);
        return true;
      } catch (error) {
        return fallbackCopy(text);
      }
    }

    return fallbackCopy(text);
  }

  function fallbackCopy(text) {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.top = "-1000px";
    document.body.appendChild(textarea);
    textarea.select();

    let copied = false;
    try {
      copied = document.execCommand("copy");
    } catch (error) {
      copied = false;
    }

    textarea.remove();
    return copied;
  }

  function initNavHighlighter() {
    const links = Array.from(document.querySelectorAll("[data-nav-link]"));
    const sections = links
      .map((link) => document.querySelector(link.getAttribute("href")))
      .filter(Boolean);

    if (!links.length || !sections.length) return;

    const setActive = (id) => {
      links.forEach((link) => {
        link.classList.toggle("is-active", link.getAttribute("href") === `#${id}`);
      });
    };

    const updateActive = () => {
      const line = window.innerHeight * 0.34;
      let activeId = sections[0].id;

      sections.forEach((section) => {
        const rect = section.getBoundingClientRect();
        if (rect.top <= line && rect.bottom > 0) {
          activeId = section.id;
        }
      });

      setActive(activeId);
    };

    if ("IntersectionObserver" in window) {
      const observer = new IntersectionObserver(() => requestAnimationFrame(updateActive), {
        root: null,
        rootMargin: "-18% 0px -62% 0px",
        threshold: [0, 0.15, 0.5, 1]
      });
      sections.forEach((section) => observer.observe(section));
    }

    let ticking = false;
    window.addEventListener(
      "scroll",
      () => {
        if (ticking) return;
        ticking = true;
        requestAnimationFrame(() => {
          updateActive();
          ticking = false;
        });
      },
      { passive: true }
    );

    updateActive();
  }

  function renderTimeBreakdownChart() {
    const mount = document.getElementById("timeBreakdownChart");
    if (!mount) return;

    const TIME_BREAKDOWN_DATA = [
      { config: "BF16", attention: 144.9, linear_gemm: 58.2, quant_cast: 0.0, norm_rope: 0.2, vae_decode: 6.3, other: 42.6 },
      { config: "+cache", attention: 83.5, linear_gemm: 34.5, quant_cast: 0.0, norm_rope: 0.2, vae_decode: 6.3, other: 29.2 },
      { config: "+M1", attention: 86.8, linear_gemm: 18.7, quant_cast: 9.2, norm_rope: 0.5, vae_decode: 6.3, other: 10.3 },
      { config: "+M2", attention: 61.4, linear_gemm: 19.1, quant_cast: 7.9, norm_rope: 0.5, vae_decode: 6.3, other: 12.9 },
      { config: "+A2", attention: 33.6, linear_gemm: 18.4, quant_cast: 6.6, norm_rope: 0.5, vae_decode: 6.3, other: 15.5 }
    ];

    const categories = [
      { key: "attention", label: "attention", color: "#4fd1c5" },
      { key: "linear_gemm", label: "linear-GEMM", color: "#7c8df6" },
      { key: "quant_cast", label: "quant/cast", color: "#d9b56f" },
      { key: "norm_rope", label: "norm/rope", color: "#9ac5a9" },
      { key: "vae_decode", label: "VAE decode", color: "#6f9bd9" },
      { key: "other", label: "other", color: "#8d99a6" }
    ];

    const width = 980;
    const height = 430;
    const margin = { top: 76, right: 64, bottom: 86, left: 94 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const rowGap = 19;
    const barHeight = 31;
    const totals = TIME_BREAKDOWN_DATA.map((row) => categories.reduce((sum, category) => sum + row[category.key], 0));
    const maxTotal = Math.ceil(Math.max(...totals) / 50) * 50;
    const rowStep = (plotHeight - barHeight) / (TIME_BREAKDOWN_DATA.length - 1);
    const tickValues = [0, maxTotal * 0.25, maxTotal * 0.5, maxTotal * 0.75, maxTotal];

    const bars = TIME_BREAKDOWN_DATA.map((row, rowIndex) => {
      const y = margin.top + rowIndex * rowStep;
      let cursorX = margin.left;
      const segments = categories
        .map((category) => {
          const value = row[category.key];
          const segmentWidth = (value / maxTotal) * plotWidth;
          const rect = `<rect x="${round(cursorX)}" y="${round(y)}" width="${round(segmentWidth)}" height="${barHeight}" rx="4" fill="${category.color}" opacity="0.88"><title>${escapeHtml(category.label)}: ${value.toFixed(1)} s</title></rect>`;
          cursorX += segmentWidth;
          return rect;
        })
        .join("");

      const total = totals[rowIndex];
      const totalX = margin.left + (total / maxTotal) * plotWidth + 10;

      return [
        `<text x="${margin.left - 14}" y="${round(y + barHeight * 0.68)}" text-anchor="end" fill="#cfd6df" font-family="ui-monospace, Menlo, monospace" font-size="14">${escapeHtml(row.config)}</text>`,
        segments,
        `<text x="${round(totalX)}" y="${round(y + barHeight * 0.68)}" fill="#a3adba" font-family="ui-monospace, Menlo, monospace" font-size="12">${total.toFixed(1)} s</text>`
      ].join("");
    }).join("");

    const grid = tickValues.map((value) => {
      const x = margin.left + (value / maxTotal) * plotWidth;
      const label = Math.round(value);
      return [
        `<line x1="${round(x)}" y1="${margin.top - 12}" x2="${round(x)}" y2="${margin.top + plotHeight + rowGap}" stroke="rgba(255,255,255,0.08)"/>`,
        `<text x="${round(x)}" y="${height - 48}" text-anchor="middle" fill="#717b88" font-family="ui-monospace, Menlo, monospace" font-size="12">${label}</text>`
      ].join("");
    }).join("");

    const legend = categories.map((category, index) => {
      const x = margin.left + index * 132;
      const y = height - 24;
      return [
        `<rect x="${x}" y="${y - 10}" width="10" height="10" rx="2" fill="${category.color}" opacity="0.9"/>`,
        `<text x="${x + 16}" y="${y}" fill="#a3adba" font-family="ui-monospace, Menlo, monospace" font-size="12">${escapeHtml(category.label)}</text>`
      ].join("");
    }).join("");

    mount.innerHTML = [
      `<svg class="time-chart" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="timeChartTitle timeChartDesc">`,
      `<title id="timeChartTitle">Per-stage time breakdown</title>`,
      `<desc id="timeChartDesc">Real per-stage GPU kernel time (seconds) for five acceleration configs; total shrinks from 252.2 s to 80.8 s.</desc>`,
      `<text x="${margin.left}" y="32" fill="#e6e9ef" font-family="ui-sans-serif, system-ui, sans-serif" font-size="20" font-weight="700">Per-stage time breakdown</text>`,
      `<text x="${margin.left}" y="54" fill="#a3adba" font-family="ui-monospace, Menlo, monospace" font-size="12">GPU-busy kernel time, summed over a full generation (wall-clock is slightly higher due to launch/CPU overhead)</text>`,
      grid,
      `<line x1="${margin.left}" y1="${height - 60}" x2="${margin.left + plotWidth}" y2="${height - 60}" stroke="rgba(255,255,255,0.18)"/>`,
      bars,
      `<text x="${margin.left + plotWidth / 2}" y="${height - 34}" text-anchor="middle" fill="#717b88" font-family="ui-monospace, Menlo, monospace" font-size="12">per-stage GPU kernel time (s)</text>`,
      `<text x="24" y="${margin.top + plotHeight / 2}" transform="rotate(-90 24 ${margin.top + plotHeight / 2})" text-anchor="middle" fill="#717b88" font-family="ui-monospace, Menlo, monospace" font-size="12">config</text>`,
      legend,
      "</svg>"
    ].join("");
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function round(value) {
    return Math.round(value * 100) / 100;
  }

  window.cosmosT2V = {
    buildVideoUrl,
    renderVideos
  };
})();
