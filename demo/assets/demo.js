/* ProReviewer demo.
 *
 * Governing principle: once a session is rendered, no node is created or
 * destroyed. Moving through the run writes a few dozen attributes and toggles
 * some [hidden] flags; CSS derives everything else. That is what lets scroll
 * position, open <details>, text selection and focus survive every scrub.
 *
 * Entry *content* is not constant over time — an answer written at step 29 is a
 * different paragraph from the one that replaced it at step 31 — so each entry
 * renders one .rev block per revision and we toggle which is visible. Only four
 * entries in the reference session have more than one.
 *
 * Classes name things; data-attributes name state.
 */
(function () {
  "use strict";

  var DATA = JSON.parse(document.getElementById("pr-data").textContent);
  var SESSIONS = DATA.sessions || [];
  if (!SESSIONS.length) return;

  var GLYPH = {
    to_be_verified: "?", supported: "✓", weak: "~", invalid: "✗",
    open: "?", partially_answered: "~", resolved: "✓"
  };
  var WORD = {
    to_be_verified: "to be verified", supported: "supported", weak: "weak",
    invalid: "invalid", open: "open", partially_answered: "partly answered",
    resolved: "resolved"
  };
  var CLAIM_STATUSES = ["to_be_verified", "supported", "weak", "invalid"];
  var QUESTION_STATUSES = ["open", "partially_answered", "resolved"];
  var SECTION_LABEL = { strengths: "Strength", weaknesses: "Weakness", questions: "Question for authors" };

  var reduceMotion = false;
  try {
    reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch (e) { /* older browsers */ }

  var PROJECT = DATA.project || {};
  var state = { session: 0, view: "overview", step: 0, focus: "", playing: false, src: "paper" };
  var VIEWS = ["overview", "watch"];
  var dom = {};          // per-session cached DOM, so switching back is free
  var timer = null;
  var frame = null;

  // ── helpers ──────────────────────────────────────────────────────

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }
  function $(id) { return document.getElementById(id); }
  function cur() { return SESSIONS[state.session]; }
  function clamp(n, lo, hi) { return n < lo ? lo : (n > hi ? hi : n); }

  /* Inline markdown only: the narration uses **bold** and `code`, nothing more.
     Everything is escaped first so no document text can inject markup. */
  function inlineMd(text) {
    var out = String(text == null ? "" : text)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
    out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    out = out.replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>");
    return out;
  }

  /* Where an entry says its evidence lives. The references that resolved to a
     real line at build time become buttons into the paper; the rest stay as
     the plain text the agent wrote, because a dead link that silently goes
     nowhere is worse than an un-clickable string. */
  function srcRef(text, links) {
    var host = el("p", "rev-srcs");
    if (!links || !links.length) {
      host.textContent = text;
      return host;
    }
    links.forEach(function (link, i) {
      if (i) host.appendChild(document.createTextNode(" "));
      var btn = el("button", "srclink", link.t);
      btn.type = "button";
      btn.setAttribute("data-line", link.n);
      btn.title = "Go to " + link.t + " in the paper (line " + link.n + ")";
      host.appendChild(btn);
    });
    return host;
  }

  function statusChip(status, extraClass) {
    var chip = el("span", "chip" + (extraClass ? " " + extraClass : ""));
    chip.setAttribute("data-status", status);
    chip.appendChild(el("span", "g", GLYPH[status] || "·"));
    // The glyph alone is not an accessible name: "~" is read as "tilde".
    chip.appendChild(el("span", "w", WORD[status] || status));
    return chip;
  }

  // ── rendering a session ──────────────────────────────────────────

  function renderEntry(entry, opts) {
    var node = el("article", "entry");
    node.id = (opts.prefix || "e") + "-" + entry.id;
    node.setAttribute("data-id", entry.id);
    node.setAttribute("data-kind", entry.kind);
    node.setAttribute("data-born", entry.born);
    node.setAttribute("data-present", "1");
    node.setAttribute("data-sel", "0");
    node.setAttribute("data-rel", "0");
    node.setAttribute("data-touch", "0");

    var head = el("div", "entry-head");
    head.appendChild(el("span", "eid", entry.id));
    var chipHost = el("span", "chip-host");
    head.appendChild(chipHost);
    if (entry.section) head.appendChild(srcRef(entry.section, entry.links));
    if (entry.type) head.appendChild(el("span", "entry-type", entry.type));

    if (opts.showOrphan && opts.orphans.indexOf(entry.id) >= 0) {
      head.appendChild(el("span", "orphan-badge", "not cited in the review"));
    }

    if (opts.spine && entry.revs.length > 1) {
      var spine = el("ol", "spine");
      spine.setAttribute("aria-label", "History: " + entry.revs.length + " revisions");
      entry.revs.forEach(function (rev, i) {
        var li = el("li");
        var btn = el("button", null, rev.change === "retraction" ? "↩" :
                     (rev.change === "reinforce" ? "↑" : (GLYPH[rev.status] || "·")));
        btn.type = "button";
        btn.setAttribute("data-status", rev.status);
        btn.setAttribute("data-goto", rev.step);
        btn.setAttribute("data-rev", i);
        btn.setAttribute("aria-label",
          (i === 0 ? "Logged" : (rev.change ? rev.changeLabel : "Updated")) +
          " at step " + (rev.step + 1));
        li.appendChild(btn);
        spine.appendChild(li);
      });
      head.appendChild(spine);
    }
    node.appendChild(head);
    node.appendChild(el("p", "entry-text", entry.text));

    if (entry.issues && entry.issues.length) {
      var ul = el("ul", "issues");
      entry.issues.forEach(function (issue) { ul.appendChild(el("li", null, issue)); });
      node.appendChild(ul);
    }
    if (entry.tag && entry.tag.length) {
      node.appendChild(el("p", "rev-srcs", entry.tag.join(" · ")));
    }

    entry.revs.forEach(function (rev, i) {
      if (i === 0 && !rev.body) return;          // birth carries no prose
      var box = el("div", "rev");
      box.setAttribute("data-rev", i);
      if (i > 0) box.hidden = true;
      if (rev.tag) box.appendChild(el("span", "tagbadge", rev.tag));
      if (rev.change) {
        var mark = el("span", "mark", (rev.change === "retraction" ? "↩ " : "↑ ") + rev.changeLabel);
        mark.setAttribute("data-change", rev.change);
        box.appendChild(mark);
      }
      if (rev.body) box.appendChild(el("p", "rev-body", rev.body));
      if (rev.srcs && rev.srcs.length) {
        box.appendChild(srcRef(rev.srcs.join(" · "), rev.links));
      }

      if (rev.supersedes != null) {
        var prev = entry.revs[rev.supersedes];
        var det = el("details", "superseded");
        var sum = el("summary", null,
          "What this replaced — written at step " + (prev.step + 1) +
          ", withdrawn at step " + (rev.step + 1));
        det.appendChild(sum);
        det.appendChild(el("div", "old", prev.body || ("status: " + WORD[prev.status])));
        box.appendChild(det);
      }
      node.appendChild(box);
    });
    return node;
  }

  function renderPoint(point, clamped, ordinal) {
    var node = el("article", "point");
    node.id = (ordinal ? "rp-" : "p-") + point.id;
    node.setAttribute("data-id", point.id);
    node.setAttribute("data-born", point.born);
    node.setAttribute("data-present", "1");
    node.setAttribute("data-sel", "0");
    node.setAttribute("data-rel", "0");

    var head = el("div", "point-h");
    head.appendChild(el("span", "eid", ordinal ? String(ordinal) : point.id));
    head.appendChild(el("span", "point-sec", SECTION_LABEL[point.section] || point.section));
    node.appendChild(head);
    node.appendChild(el("p", "point-text", point.text));

    if (point.cites.length) {
      var cites = el("div", "cites");
      point.cites.forEach(function (ref) {
        var chip = el("button", "cite", ref);
        chip.type = "button";
        chip.setAttribute("data-cite", ref);
        cites.appendChild(chip);
      });
      node.appendChild(cites);
    }
    if (clamped) node.classList.add("clamped");
    return node;
  }

  function buildSession(session) {
    var built = { entries: {}, points: {} };

    var logBody = el("div");
    session.entries.forEach(function (entry) {
      var node = renderEntry(entry, {
        prefix: "e", spine: session.hasRun, showOrphan: true,
        orphans: session.evidence.orphans
      });
      if (!session.hasRun) {
        node.querySelectorAll(".rev").forEach(function (rev) { rev.hidden = false; });
      }
      built.entries[entry.id] = node;
      logBody.appendChild(node);
    });
    built.logBody = logBody;

    /* The review shares the right column with the paper: both are things the
       log points *at*, and having it beside the log makes clicking a point to
       light up its entries work without a separate view. */
    var runReview = el("div");
    var ordinals = { strengths: 0, weaknesses: 0, questions: 0 };
    ["strengths", "weaknesses", "questions"].forEach(function (name) {
      var points = session.review.points.filter(function (p) { return p.section === name; });
      if (!points.length) return;
      var section = el("section", "rv-section");
      section.setAttribute("data-section", name);
      section.appendChild(el("h3", null,
        name === "questions" ? "Questions for authors"
          : name.charAt(0).toUpperCase() + name.slice(1)));
      points.forEach(function (point) {
        ordinals[name] += 1;
        var node = renderPoint(point, false, ordinals[name]);
        built.points[point.id] = node;
        section.appendChild(node);
      });
      runReview.appendChild(section);
    });
    if (session.review.summary) {
      var sum = el("section", "rv-section");
      sum.appendChild(el("h3", null, "Summary"));
      sum.appendChild(el("p", "summary-text", session.review.summary));
      runReview.insertBefore(sum, runReview.firstChild);
    }
    built.runReviewBody = runReview;

    return built;
  }

  // ── the run: folding state at a step ─────────────────────────────

  function revIndexAt(entry, step) {
    var idx = -1;
    for (var i = 0; i < entry.revs.length; i++) {
      if (entry.revs[i].step <= step) idx = i; else break;
    }
    return idx;
  }

  function applyStep(n) {
    var session = cur();
    var built = dom[state.session];
    if (!session.hasRun) return;

    var step = session.steps[n] || session.steps[session.steps.length - 1];
    var touched = {};
    (step ? step.touched : []).forEach(function (id) { touched[id] = true; });
    var aims = {};
    aimedAt(step, n).forEach(function (id) { aims[id] = true; });

    var counts = {};
    var shown = 0;
    var openNow = 0;
    session.entries.forEach(function (entry) {
      var node = built.entries[entry.id];
      var idx = revIndexAt(entry, n);
      var present = entry.born <= n && idx >= 0;
      node.setAttribute("data-present", present ? "1" : "0");
      node.setAttribute("data-touch", touched[entry.id] ? "1" : "0");
      node.setAttribute("data-aim", aims[entry.id] ? "1" : "0");
      if (!present) return;
      shown++;

      var rev = entry.revs[idx];
      counts[rev.status] = (counts[rev.status] || 0) + 1;

      var host = node.querySelector(".chip-host");
      host.textContent = "";
      var changedNow = touched[entry.id] && idx > 0 && entry.revs[idx].step === n;
      var prevStatus = idx > 0 ? entry.revs[idx - 1].status : null;
      host.appendChild(statusChip(rev.status));
      // Show the transition in place for exactly one step: with both states on
      // screen at once, a status flip needs no animation to be legible.
      if (changedNow && prevStatus && prevStatus !== rev.status) {
        host.appendChild(el("span", "chip-arrow", "←"));
        host.appendChild(statusChip(prevStatus, "chip-was"));
      }

      node.querySelectorAll(".rev").forEach(function (box) {
        box.hidden = Number(box.getAttribute("data-rev")) !== idx;
      });
      node.querySelectorAll(".spine button").forEach(function (btn) {
        btn.setAttribute("data-on", Number(btn.getAttribute("data-rev")) === idx ? "1" : "0");
      });

      if (changedNow && rev.supersedes != null) {
        var det = node.querySelector('.rev[data-rev="' + idx + '"] .superseded');
        if (det && det.dataset.autoOpened !== "1") {
          det.open = true;
          det.dataset.autoOpened = "1";   // once only; the user's toggling wins after
        }
      }
    });

    var pointsShown = 0;
    session.review.points.forEach(function (point) {
      var node = built.points[point.id];
      var present = point.born >= 0 && point.born <= n;
      node.setAttribute("data-present", present ? "1" : "0");
      if (present) pointsShown++;
    });

    renderAction(step, n);
    renderPaper(step, n);
    markCited(built);
    renderStatusBar(counts);
    // The unsettled count is the loop's control flow, so it sits in the log's
    // own header rather than in a panel of its own: watching it drain to zero
    // is watching the agent earn the right to write the review.
    var left = step ? step.left : 0;
    $("log-count").textContent = shown + " entr" + (shown === 1 ? "y" : "ies") +
      (left ? "  ·  " + left + " unsettled" : (shown ? "  ·  all settled" : ""));
    $("review-note").textContent = pointsShown
      ? pointsShown + " of " + session.review.points.length + " points written"
      : "not written yet";
    if (!pointsShown) {
      built.runReviewBody.parentNode && showEmptyReview(session, n);
    } else {
      var empty = $("run-review-body").querySelector(".empty");
      if (empty) empty.remove();
    }
    markSrcDot(pointsShown);

    $("stepnum").textContent = "step " + (n + 1) + " / " + session.steps.length;
    var rail = $("rail");
    rail.setAttribute("aria-valuenow", n);
    rail.setAttribute("aria-valuetext", stepSentence(step, n));
    $("ticks").querySelectorAll(".tick").forEach(function (tick, i) {
      tick.setAttribute("data-on", i === n ? "1" : "0");
      tick.setAttribute("data-past", i <= n ? "1" : "0");
    });

    if (!state.playing) announce(stepSentence(step, n));
    frameTouched(step);
    // After the scrolls above have been issued, so the wire is measured against
    // where things are landing rather than where they were.
    scheduleWires(step, n);
  }

  /* Wires are geometry, so they are redrawn on anything that moves an endpoint:
     a step change, either column scrolling, or a resize. Coalesced into one
     frame so a scroll cannot queue hundreds of redraws. */
  var wireFrame = null;
  function scheduleWires(step, n) {
    if (wireFrame) return;
    wireFrame = requestAnimationFrame(function () {
      wireFrame = null;
      var session = cur();
      if (!session.hasRun) return;
      var s = step || session.steps[state.step];
      drawWires(s, n == null ? state.step : n);
    });
  }

  /* The entries that sent the agent where it went. It names them itself in the
     narration before reading ("targeting **Q1**, **C1/C3/C4**"), and the writes
     that answer them land one or more steps later — so the aim is carried
     forward to the steps that act on it, exactly as far as the read that
     produced them. Without that it would flash for one step and be missed. */
  function aimedAt(step, n) {
    if (!step) return [];
    if (step.targets && step.targets.length) return step.targets;
    var session = cur();
    if (step.src != null && step.touched.length && session.steps[step.src]) {
      return session.steps[step.src].targets || [];
    }
    return [];
  }

  function showEmptyReview(session, n) {
    var body = $("run-review-body");
    if (body.querySelector(".empty")) return;
    var first = session.review.points.reduce(function (min, p) {
      return p.born >= 0 && (min < 0 || p.born < min) ? p.born : min;
    }, -1);
    var note = el("p", "empty",
      "Nothing yet. The agent writes the review only after every claim has a verdict and " +
      "every question is settled — the first point appears at step " + (first + 1) + ".");
    body.insertBefore(note, body.firstChild);
  }

  function stepSentence(step, n) {
    if (!step) return "Step " + (n + 1);
    var parts = ["Step " + (n + 1) + " of " + cur().steps.length + ". " + step.label + "."];
    if (step.status !== "ok") parts.push("This call did not run.");
    if (step.touched.length === 0) parts.push("Nothing changed.");
    else if (step.touched.length <= 3) parts.push("Changed " + step.touched.join(", ") + ".");
    else parts.push(step.touched.length + " entries changed.");
    return parts.join(" ");
  }

  function renderStatusBar(counts) {
    var bar = $("statusbar");
    bar.textContent = "";
    CLAIM_STATUSES.concat(QUESTION_STATUSES).forEach(function (status) {
      // Zero-count statuses stay visible but dimmed: this run never produced an
      // `invalid` claim, and hiding it would hide that the verdict exists.
      var n = counts[status] || 0;
      var item = el("span", "sb-item");
      item.setAttribute("data-zero", n ? "0" : "1");
      var chip = statusChip(status);
      chip.appendChild(el("span", "n", " " + n));
      item.appendChild(chip);
      bar.appendChild(item);
    });
  }

  /* Which way this step moved. Every step is one of two things — the agent
     reaches right into the paper for evidence, or writes back left into the
     log — and the bar names the direction, lights the column that changed, and
     lists the commands that did it. */
  function renderAction(step, n) {
    var bar = $("actionbar");
    if (!step) return;
    var writes = step.writes || [];
    /* Three directions, not two. The outline phase also writes, but it writes
       the *review* — which lives in the right column — so calling it "wrote
       back to the log" would light the wrong side and misdescribe the one
       transition the whole method turns on. */
    var toReview = writes.length && writes.every(function (w) {
      return w.v === "point" || w.v === "outline";
    });
    var dir = step.read ? "fetch" : (writes.length ? (toReview ? "compose" : "write") : "none");

    bar.setAttribute("data-dir", dir);
    $("col-log").setAttribute("data-active", dir === "write" ? "1" : "0");
    $("col-src").setAttribute("data-active", dir === "fetch" || dir === "compose" ? "1" : "0");

    // The glyph points at the column that is about to change; the words say the
    // same thing. Two channels, because this is the one fact on the page a
    // reader should never have to work out.
    $("act-arrow").textContent = dir === "write" ? "←"
      : (dir === "fetch" || dir === "compose" ? "→" : "·");
    $("act-dir").textContent = dir === "fetch" ? "fetched from the paper"
      : (dir === "write" ? "wrote back to the log"
      : (dir === "compose" ? "wrote the review, out of the log" : "nothing moved"));

    var what = $("act-what");
    what.textContent = "";
    if (step.status !== "ok") {
      var kind = el("span", "step-kind", step.status);
      kind.setAttribute("data-status", step.status);
      what.appendChild(kind);
    }
    what.appendChild(document.createTextNode(step.label));

    /* The entries the agent said it was going after. Shown here as well as in
       the log because the bar is where the reader is looking, and the two
       markings are the same fact: this is what sent it. */
    var forHost = $("act-for");
    forHost.textContent = "";
    var aims = aimedAt(step, n);
    if (aims.length) {
      forHost.appendChild(document.createTextNode(
        dir === "fetch" ? "for " : "answering "));
      aims.forEach(function (id, i) {
        if (i) forHost.appendChild(document.createTextNode(" "));
        var chip = el("button", "aim", id);
        chip.type = "button";
        chip.setAttribute("data-cite", id);
        forHost.appendChild(chip);
      });
    }

    renderActs(step, writes);

    var say = $("step-say");
    say.hidden = !step.say;
    say.innerHTML = step.say ? inlineMd(step.say) : "";

    var err = $("step-err");
    err.hidden = !step.err;
    err.textContent = step.err || "";
  }

  /* The commands themselves, one chip each, in the order the CLI ran them.
     What each one *did* is visible in the log beside it, so the chip only has
     to say which command and to what — and clicking it jumps there. */
  function renderActs(step, writes) {
    var host = $("acts");
    host.textContent = "";

    if (!writes.length) {
      if (step.status !== "ok" && step.ops && step.ops.length) {
        host.appendChild(el("span", "caption",
          "The " + step.ops.length + " command(s) in this block never ran, so nothing was "
          + "logged. They were re-issued a moment later."));
      }
      return;
    }

    writes.forEach(function (write) {
      var chip = el("button", "act");
      chip.type = "button";
      chip.setAttribute("data-v", write.v);
      chip.setAttribute("data-cite", write.id);
      chip.appendChild(el("span", "a-op", write.op));
      chip.appendChild(el("span", "a-id", write.id));
      if (write.from && write.to && write.from !== write.to) {
        chip.appendChild(el("span", "a-to",
          (GLYPH[write.from] || "·") + " → " + (GLYPH[write.to] || "·") + " " +
          (WORD[write.to] || write.to)));
      } else if (write.to) {
        chip.appendChild(el("span", "a-to", (GLYPH[write.to] || "·") + " " + (WORD[write.to] || write.to)));
      }
      chip.title = write.text || write.op;
      host.appendChild(chip);
    });

    if (step.cmd) {
      var raw = el("details", "rawcmd");
      raw.appendChild(el("summary", null, "the shell command it actually ran"));
      raw.appendChild(el("pre", "cmd", step.cmd));
      host.appendChild(raw);
    }
  }

  /* The right column is the whole paper, standing still. The run moves through
     it, so a step only re-marks the lines entering and leaving the current
     range — never the 838 rows, and never a rebuild. */
  function renderPaper(step, n) {
    var session = cur();
    var built = dom[state.session];
    if (!session.paper) { $("src-note").textContent = "paper.md not archived"; return; }

    var read = step && step.read ? step.read : null;
    var from = n;
    /* A step that writes almost never reads in the same call. Keep the passage
       that produced it on screen rather than clearing the column, but only
       where the claim holds: while writing the review the agent is working from
       the log, not the paper. */
    if (!read && step && step.src != null && step.touched.length && session.steps[step.src]) {
      read = session.steps[step.src].read;
      from = step.src;
    }

    if (!read) {
      markRange(built, null);
      $("src-note").textContent = step && (step.points || step.outline)
        ? "not open — working from the log" : "not open at this step";
      setCoverageNow(null);
      return;
    }

    var end = read.start + Math.max(read.lines, 1) - 1;
    $("src-note").textContent = read.start + "–" + end +
      (read.heading ? "  ·  " + read.heading : "") +
      (from === n ? "" : "  ·  opened at step " + (from + 1));
    markRange(built, [read.start, end], from !== n);
    setCoverageNow([read.start, end]);
    if (state.src === "paper") frameRange(built, read.start, end);
  }

  /* ── The wire ──────────────────────────────────────────────────────
     The state on the left and the environment on the right, joined across the
     gutter. Which entries get wired, and to where, is never invented: either
     the step's own read range (the passage it is acting on right now), or the
     sections the entry itself cites, resolved to real lines at build time.

     Endpoints are measured against the .split box, so a wire whose target is
     scrolled out of view is clipped to the edge it left through and dashed
     rather than drawn off into space. */
  function drawWires(step, n) {
    var g = $("wire-g");
    if (!g) return;
    g.textContent = "";
    var session = cur();
    var built = dom[state.session];
    if (!session.hasRun || !session.paper || state.src !== "paper") return;

    var box = rect($("split"));
    if (!box || !box.height) return;
    $("wires").setAttribute("viewBox", "0 0 " + box.width + " " + box.height);

    var read = step && step.read ? step.read : null;
    var from = n;
    if (!read && step && step.src != null && step.touched.length && session.steps[step.src]) {
      read = session.steps[step.src].read;
      from = step.src;
    }
    if (!read) return;

    var dir = step.read ? "fetch" : "write";
    var end = read.start + Math.max(read.lines, 1) - 1;
    var paper = paperAnchor(built, read.start, end, box);
    var hub = hubAnchor(box);
    if (!paper || !hub) return;

    // Which entries this step is about: the ones it wrote, or failing that the
    // ones it said it was going after. Both are the log talking to the paper.
    var ids = (step.touched && step.touched.length) ? step.touched : aimedAt(step, n);
    var sides = [];
    ids.forEach(function (id) {
      if (sides.length >= 6) return;            // past this it is a hairball
      var node = built.entries[id];
      if (!node || node.getAttribute("data-present") !== "1") return;
      var anchor = entryAnchor(node, box);
      if (anchor) sides.push(anchor);
    });
    if (!sides.length) return;

    /* Routed through the channel rather than straight across it, so the wires
       say what the middle column says: the entries went in, the action happened,
       the passage came out. The arrowhead is only ever on the arriving end. */
    if (dir === "fetch") {
      sides.forEach(function (a) { wire(g, a, hub.left, false); });
      wire(g, hub.right, paper, true);
    } else {
      wire(g, paper, hub.right, false);
      sides.forEach(function (b) { wire(g, hub.left, b, true); });
    }
  }

  /* The two edges of the channel, at the height of its arrow: every wire enters
     one side and leaves the other. */
  function hubAnchor(box) {
    var channel = rect($("actionbar"));
    var arrow = rect($("act-arrow"));
    if (!channel || !channel.width) return null;
    var y = (arrow && arrow.height ? arrow.top + arrow.height / 2
                                   : channel.top + 28) - box.top;
    y = clamp(y, 6, box.height - 6);
    return {
      left: { x: channel.left - box.left, y: y, clip: false },
      right: { x: channel.right - box.left, y: y, clip: false }
    };
  }

  function rect(node) {
    if (!node || !node.getBoundingClientRect) return null;
    return node.getBoundingClientRect();
  }

  /* An endpoint on the log side: the right edge of the entry card, clamped to
     the column it lives in so a scrolled-away entry does not drag the wire out
     of the picture. */
  function entryAnchor(node, box) {
    var r = rect(node), col = rect($("log-pane"));
    if (!r || !col || !r.height) return null;
    var y = r.top + r.height / 2;
    var clip = y < col.top || y > col.bottom;
    return {
      x: Math.min(r.right, col.right) - box.left,
      y: clamp(y, col.top + 4, col.bottom - 4) - box.top,
      clip: clip
    };
  }

  function paperAnchor(built, start, end, box) {
    var rows = built.paperRows, col = rect($("paper-pane"));
    if (!rows || !col) return null;
    var first = rows[start], last = rows[Math.min(end, rows.length - 1)];
    var a = rect(first), b = rect(last) || a;
    if (!a) return null;
    var y = (a.top + (b ? b.bottom : a.bottom)) / 2;
    var clip = y < col.top || y > col.bottom;
    return {
      x: a.left - box.left,
      y: clamp(y, col.top + 4, col.bottom - 4) - box.top,
      clip: clip
    };
  }

  /* One segment, always drawn in the direction of flow — so the arrowhead is
     always marker-end, and there is no orientation to get backwards. */
  function wire(g, a, b, head) {
    var svg = "http://www.w3.org/2000/svg";
    var path = document.createElementNS(svg, "path");
    var mid = (a.x + b.x) / 2;
    path.setAttribute("d", "M" + a.x + "," + a.y +
      " C" + mid + "," + a.y + " " + mid + "," + b.y + " " + b.x + "," + b.y);
    path.setAttribute("class", "wire");
    path.setAttribute("data-clip", a.clip || b.clip ? "1" : "0");
    if (head) path.setAttribute("marker-end", "url(#wire-head)");
    g.appendChild(path);

    // A dot where the wire leaves, so the source end is not just a bare stop.
    var dot = document.createElementNS(svg, "circle");
    dot.setAttribute("cx", a.x);
    dot.setAttribute("cy", a.y);
    dot.setAttribute("r", "2.5");
    dot.setAttribute("class", "wire-dot");
    g.appendChild(dot);
  }

  /* Following a citation is a state change, not a side effect: `cite` holds the
     line, commit() renders the mark, and the scroll happens after. Doing the
     marking inline instead would have it wiped by the very commit the click
     triggers — which is exactly what happened when it was written that way. */
  function gotoLine(line) {
    var rows = dom[state.session].paperRows;
    if (!rows || !rows[line]) return;
    setState({ view: "watch", src: "paper", cite: line });
    var pane = $("paper-pane");
    pane.scrollTo({
      top: Math.max(0, rows[line].offsetTop - 24),
      behavior: reduceMotion ? "auto" : "smooth"
    });
    announce("Jumped to line " + line + " in the paper.");
  }

  /* A heading names the block under it, so a citation of "§3.2" marks the
     section, not the one line the heading sits on. */
  function markCited(built) {
    var rows = built.paperRows;
    if (!rows) return;
    (built.cited || []).forEach(function (i) {
      if (rows[i]) rows[i].setAttribute("data-cited", "0");
    });
    built.cited = [];
    var line = state.cite;
    if (!line || !rows[line]) return;
    var end = line;
    var limit = Math.min(rows.length - 1, line + 40);
    while (end + 1 <= limit && rows[end + 1].getAttribute("data-h") !== "1") end++;
    for (var i = line; i <= end; i++) {
      rows[i].setAttribute("data-cited", "1");
      built.cited.push(i);
    }
  }

  function markRange(built, span, carried) {
    var rows = built.paperRows;
    if (!rows) return;
    (built.lit || []).forEach(function (i) {
      if (rows[i]) rows[i].setAttribute("data-now", "0");
    });
    var lit = [];
    if (span) {
      for (var i = span[0]; i <= span[1] && i < rows.length; i++) {
        if (!rows[i]) continue;
        rows[i].setAttribute("data-now",
          i === span[0] ? "first" : (i === span[1] ? "last" : "1"));
        lit.push(i);
      }
    }
    built.lit = lit;
    $("paper-pane").setAttribute("data-carried", carried ? "1" : "0");
  }

  /* scrollIntoView scrolls every scrollable ancestor and cannot frame a range,
     so the arithmetic is done by hand: show the passage from its first line. */
  function frameRange(built, start, end) {
    var pane = $("paper-pane");
    var first = built.paperRows[start];
    var last = built.paperRows[Math.min(end, built.paperRows.length - 1)];
    if (!first) return;
    var top = first.offsetTop;
    var height = last ? (last.offsetTop + last.offsetHeight - top) : 0;
    var target = height && height < pane.clientHeight
      ? top - (pane.clientHeight - height) / 2
      : top - 24;
    pane.scrollTo({ top: Math.max(0, target), behavior: reduceMotion ? "auto" : "smooth" });
  }

  /* Built once per session. Rendering all 838 lines up front is what makes the
     never-opened regions showable at all: they are the faint lines still
     sitting between the passages the run did visit. */
  function buildPaper(session) {
    var host = $("paper-body");
    host.textContent = "";
    var rows = [null];
    if (!session.paper) {
      host.appendChild(el("p", "paper-empty",
        "paper.md was not archived with this session, so the text it read cannot be shown."));
      return rows;
    }

    var opened = {};
    session.paper.read.forEach(function (span) {
      for (var i = span[0]; i <= span[1]; i++) opened[i] = true;
    });

    var lines = session.paper.text.split("\n");
    lines.forEach(function (text, i) {
      var number = i + 1;
      var row = el("div", "pline");
      row.setAttribute("data-n", number);
      row.setAttribute("data-cov", opened[number] ? "1" : "0");
      row.setAttribute("data-now", "0");
      if (/^#{1,6}\s/.test(text)) row.setAttribute("data-h", "1");
      row.appendChild(el("span", "n", String(number)));
      row.appendChild(el("span", "tx", text));
      host.appendChild(row);
      rows.push(row);
    });
    return rows;
  }


  /* The ribbon is the paper end to end: filled where the run opened it, empty
     where it never did. What the agent never read is as much a fact about the
     review as what it did, and it is only showable because the whole paper is
     inlined rather than just the excerpts.

     Built once per session; stepping only moves the marker, so the ribbon is
     never rebuilt underneath a reader who is looking at it. */
  function buildCoverage(session) {
    var bar = $("cov-bar");
    var marker = $("cov-now");
    bar.textContent = "";
    bar.appendChild(marker);
    if (!session.paper) { marker.hidden = true; return; }

    var paper = session.paper;
    var cursor = 1;
    paper.read.forEach(function (span) {
      if (span[0] > cursor) bar.appendChild(seg(cursor, span[0] - 1, 0));
      bar.appendChild(seg(span[0], span[1], 1));
      cursor = span[1] + 1;
    });
    if (cursor <= paper.totalLines) bar.appendChild(seg(cursor, paper.totalLines, 0));

    bar.title = Math.round(paper.coverage * 100) + "% of " + paper.totalLines +
      " lines were opened during this run";

    function seg(from, to, read) {
      var s = el("span", "cov-seg");
      s.setAttribute("data-read", read);
      s.style.flex = (to - from + 1) + " 0 0";
      return s;
    }
  }

  function setCoverageNow(span) {
    var marker = $("cov-now");
    var paper = cur().paper;
    if (!span || !paper || !paper.totalLines) { marker.hidden = true; return; }
    marker.hidden = false;
    marker.style.left = (100 * (span[0] - 1) / paper.totalLines) + "%";
    marker.style.width = (100 * (span[1] - span[0] + 1) / paper.totalLines) + "%";
  }

  /* scrollIntoView scrolls every scrollable ancestor and cannot frame a range,
     so the arithmetic is done by hand: centre the touched block if it fits. */
  function frameTouched(step) {
    if (!step || !step.touched.length) return;
    var pane = $("log-pane");
    var built = dom[state.session];
    var tops = [], bottoms = [];
    step.touched.forEach(function (id) {
      var node = built.entries[id];
      if (!node || node.getAttribute("data-present") !== "1") return;
      tops.push(node.offsetTop);
      bottoms.push(node.offsetTop + node.offsetHeight);
    });
    if (!tops.length) return;
    var top = Math.min.apply(null, tops), bottom = Math.max.apply(null, bottoms);
    var height = bottom - top;
    var target = height <= pane.clientHeight
      ? top - (pane.clientHeight - height) / 2
      : top - pane.clientHeight * 0.25;
    pane.scrollTo({ top: Math.max(0, target - 40), behavior: reduceMotion ? "auto" : "smooth" });
  }

  function announce(text) {
    var node = $("announcer");
    if (node) node.textContent = text;
  }

  // ── rail ─────────────────────────────────────────────────────────

  /* The replay starts at the first step that opens the paper, so the setup —
     venv, dependency install, `init`, and any call the operator declined — is
     not shown. Say so under the rail: a trimmed run presented as the whole
     transcript would misstate what was recorded. */
  function railHelp(session) {
    var help = $("rail-help");
    help.textContent = "";
    var hidden = session.counts.hiddenSteps;
    if (hidden) {
      var note = el("span", "rail-trim",
        hidden + " setup step" + (hidden === 1 ? "" : "s") + " hidden of " +
        session.counts.recordedSteps + " recorded · ");
      note.title = "Environment setup and calls that never ran. None of them " +
        "changed the review log.";
      help.appendChild(note);
    }
    help.appendChild(document.createTextNode("Arrow keys step · "));
    help.appendChild(el("kbd", null, "J"));
    help.appendChild(document.createTextNode("/"));
    help.appendChild(el("kbd", null, "K"));
    help.appendChild(document.createTextNode(" jump to where the agent changed its mind · "));
    help.appendChild(el("kbd", null, "Home"));
    help.appendChild(document.createTextNode("/"));
    help.appendChild(el("kbd", null, "End"));
  }

  function buildRail(session) {
    var ticks = $("ticks");
    ticks.textContent = "";
    var notableBy = {};
    session.notable.forEach(function (item) { notableBy[item.step] = item; });

    session.steps.forEach(function (step, i) {
      var li = el("li", "tick");
      li.setAttribute("data-n", i);
      li.style.setProperty("--mut", Math.min(step.mut, 8));
      var kind = step.phase === "setup" ? "setup"
        : (step.read ? "read" : (step.phase === "outline" ? "review" : "log"));
      li.setAttribute("data-kind", kind);
      if (notableBy[i]) {
        li.setAttribute("data-mark", notableBy[i].kind);
        li.setAttribute("data-glyph", notableBy[i].kind === "retraction" ? "↩" : "↑");
      }
      ticks.appendChild(li);
    });

    $("rail").setAttribute("aria-valuemax", Math.max(session.steps.length - 1, 0));
    buildChapters(session);
    buildNotableStrip(session);
    railHelp(session);
  }

  function buildChapters(session) {
    var host = $("chapters");
    host.textContent = "";
    var groups = [];
    session.steps.forEach(function (step, i) {
      var name = step.phase;
      if (!groups.length || groups[groups.length - 1].name !== name) {
        groups.push({ name: name, from: i, to: i });
      } else groups[groups.length - 1].to = i;
    });
    var titles = { setup: "Setting up", investigate: "Reading & logging", outline: "Writing the review" };
    groups.forEach(function (group) {
      var span = el("span", "chapter", titles[group.name] || group.name);
      span.style.flex = (group.to - group.from + 1) + " 0 0";
      host.appendChild(span);
    });
  }

  function buildNotableStrip(session) {
    var host = $("notable-strip");
    host.textContent = "";
    if (!session.notable.length) return;
    var retractions = session.notable.filter(function (n) { return n.kind === "retraction"; });
    host.appendChild(el("span", "notable-lead",
      retractions.length
        ? "This run withdrew " + retractions.length + " of its own findings:"
        : "Moments where the agent revised itself:"));
    session.notable.forEach(function (item) {
      var btn = el("button", "notable-btn",
        (item.kind === "retraction" ? "↩ " : "↑ ") + item.id + " · step " + (item.step + 1));
      btn.type = "button";
      btn.title = item.label;
      btn.setAttribute("data-goto", item.step);
      host.appendChild(btn);
    });
  }

  function nextNotable(from, dir) {
    var list = cur().notable.map(function (n) { return n.step; }).sort(function (a, b) { return a - b; });
    var found = null;
    for (var i = 0; i < list.length; i++) {
      if (dir > 0 && list[i] > from) { found = list[i]; break; }
      if (dir < 0 && list[i] < from) found = list[i];
    }
    return found;
  }

  // ── state ────────────────────────────────────────────────────────

  function setState(patch) {
    Object.keys(patch).forEach(function (key) { state[key] = patch[key]; });
    if (frame) return;
    frame = requestAnimationFrame(function () { frame = null; commit(); });
  }

  function commit() {
    var session = cur();
    document.body.setAttribute("data-view", state.view);
    document.body.setAttribute("data-focus", state.focus);
    document.body.setAttribute("data-playing", state.playing ? "1" : "0");

    document.querySelectorAll("[data-view-btn]").forEach(function (btn) {
      btn.setAttribute("aria-selected", btn.getAttribute("data-view-btn") === state.view ? "true" : "false");
    });
    VIEWS.forEach(function (name) {
      $("view-" + name).hidden = name !== state.view;
    });

    // The right column holds two things the log points at; only one at a time.
    $("col-src").setAttribute("data-src", state.src);
    $("paper-pane").hidden = state.src !== "paper";
    $("review-pane").hidden = state.src !== "review";
    document.querySelectorAll("[data-src-btn]").forEach(function (btn) {
      btn.setAttribute("aria-selected",
        btn.getAttribute("data-src-btn") === state.src ? "true" : "false");
    });

    if (session.hasRun) applyStep(state.step);
    applyFocus();
  }

  /* The review filling up while the reader is looking at the paper is worth
     flagging; switching the tab out from under them is not. */
  function markSrcDot(pointsShown) {
    var btn = document.querySelector('[data-src-btn="review"]');
    if (btn) btn.setAttribute("data-dot", pointsShown && state.src !== "review" ? "1" : "0");
  }

  function applyFocus() {
    var session = cur();
    var built = dom[state.session];
    var id = state.focus;
    var related = {};
    if (id) {
      if (session.evidence.entryToPoints[id]) {
        session.evidence.entryToPoints[id].forEach(function (p) { related[p] = true; });
      }
      if (session.evidence.pointToEntries[id]) {
        session.evidence.pointToEntries[id].forEach(function (e) { related[e] = true; });
      }
    }
    function mark(map) {
      Object.keys(map).forEach(function (key) {
        var node = map[key];
        node.setAttribute("data-sel", key === id ? "1" : "0");
        node.setAttribute("data-rel", related[key] ? "1" : "0");
      });
    }
    mark(built.entries); mark(built.points);
  }

  function gotoStep(n) {
    var session = cur();
    // Moving the run drops the citation mark: it answered "where does C6 say
    // its evidence is", which is no longer the question being asked.
    setState({ step: clamp(n, 0, session.steps.length - 1), cite: 0 });
  }

  function play() {
    if (state.playing) return;
    setState({ playing: true });
    timer = setInterval(function () {
      if (state.step >= cur().steps.length - 1) { pause(); return; }
      setState({ step: state.step + 1 });
    }, reduceMotion ? 1400 : 900);
  }
  function pause() {
    if (timer) { clearInterval(timer); timer = null; }
    setState({ playing: false });
  }

  // ── session mounting ─────────────────────────────────────────────

  function mountSession(index) {
    state.session = index;
    var session = SESSIONS[index];
    if (!dom[index]) dom[index] = buildSession(session);
    var built = dom[index];

    swap($("log-body"), built.logBody);
    swap($("run-review-body"), built.runReviewBody);

    $("run-title").textContent = session.hasRun
      ? "replay_trajectory"
      : "review_and_log";
    $("run-byline").textContent = session.authors && session.authors.length
      ? session.authors.join(", ") + (session.venue ? " · " + session.venue : "")
      : (session.venue || "");
    buildRunStats(session);
    buildUncitedNote(session, built);

    // The score belongs with the review it summarises, not in the chrome.
    var scoreBox = $("score-box");
    scoreBox.hidden = !session.scoreLine;
    if (session.scoreLine) $("score-line").textContent = session.scoreLine;

    /* Only sessions with an archived transcript can be replayed. The rest keep
       their log and review; the scrubber is removed rather than faked. */
    document.body.setAttribute("data-hasrun", session.hasRun ? "1" : "0");
    if (session.hasRun) buildRail(session);
    built.paperRows = buildPaper(session);
    built.lit = [];
    buildCoverage(session);
    buildMethod(session);
    buildOverview(session);
    buildWarnings();

    buildSelectors();

    /* Land on the finished artifact; replay is a choice. A session with no
       transcript has no paper passages to show, so it opens on its review. */
    setState({
      step: session.hasRun ? session.steps.length - 1 : 0,
      focus: "",
      src: session.hasRun ? "paper" : "review"
    });
  }

  function swap(host, node) {
    host.textContent = "";
    host.appendChild(node);
  }

  /* Entries the review never cites are a real finding, and an absence is
     invisible by construction — so state the count and make the ids clickable
     rather than relying on the reader noticing nothing is highlighted. */
  function buildUncitedNote(session, built) {
    var host = $("uncited-note");
    host.textContent = "";
    var orphans = session.evidence.orphans;
    if (!orphans.length) return;
    host.appendChild(document.createTextNode(
      orphans.length + " of the " + session.entries.length +
      " entries were investigated but never cited in the review: "));
    orphans.forEach(function (id, i) {
      if (i) host.appendChild(document.createTextNode(", "));
      var chip = el("button", "cite", id);
      chip.type = "button";
      chip.setAttribute("data-cite", id);
      host.appendChild(chip);
    });
  }

  function buildMethod(session) {
    // The legend appears twice: in the overview explainer, and inside the log
    // pane where you are actually reading statuses. Both are filled from here.
    var absent = CLAIM_STATUSES.concat(QUESTION_STATUSES).filter(function (status) {
      return !session.statusCounts[status];
    });
    ["status-legend", "status-legend-pane"].forEach(function (id) {
      var legend = $(id);
      if (!legend) return;
      legend.textContent = "";
      CLAIM_STATUSES.concat(QUESTION_STATUSES).forEach(function (status) {
        var chip = statusChip(status);
        chip.setAttribute("data-zero", session.statusCounts[status] ? "0" : "1");
        legend.appendChild(chip);
      });
    });
    // Statuses this run never produced stay in the legend, greyed: otherwise a
    // reader never learns the verdict exists.
    var caption = absent.length
      ? "Greyed out above: verdicts this run never reached (" +
        absent.map(function (s) { return WORD[s]; }).join(", ") + ")."
      : "";
    ["status-caption", "status-caption-pane"].forEach(function (id) {
      if ($(id)) $(id).textContent = caption;
    });
  }

  function buildWarnings() {
    var host = $("build-warnings");
    host.textContent = "";
    (DATA.build && DATA.build.sessions || []).forEach(function (record) {
      (record.warnings || []).forEach(function (text) {
        host.appendChild(el("p", "warn", record.slug + ": " + text));
      });
    });
  }

  /* The overview is the project page: what this is, who made it, where the
     paper and the code live. Everything here comes from demo/project.json, so
     no link is invented by the builder. */
  function buildOverview(session) {
    $("brand-name").textContent = PROJECT.name || "";
    $("hero-name").textContent = PROJECT.name || "";
    $("hero-sub").innerHTML = inlineMd(PROJECT.subtitle || "");
    // Through inlineMd so project.json can mark the load-bearing phrases with
    // **bold**. It escapes first, so nothing in that file can inject markup.
    $("hero-tagline").innerHTML = inlineMd(PROJECT.tagline || "");
    $("idea-blurb").innerHTML = inlineMd(PROJECT.blurb || "");
    $("bibtex").textContent = PROJECT.bibtex || "";

    $("hero-affil").textContent = PROJECT.affiliation || "";
    $("hero-authors").textContent = (PROJECT.authors || []).join(" · ");

    // The builder wrote a data URI into data-src, or left it empty when the
    // asset was missing. An empty <img> is worse than no figure at all.
    var fig = $("methodfig"), img = $("method-img");
    var src = img.getAttribute("data-src");
    if (src) { img.src = src; fig.hidden = false; } else { fig.hidden = true; }

    var repo = PROJECT.repo || "";
    var navRepo = $("nav-repo");
    if (repo) navRepo.href = repo; else navRepo.hidden = true;
    // Joined rather than concatenated, so an absent field does not leave a
    // dangling separator behind it.
    $("footer-meta").textContent = [PROJECT.license, PROJECT.affiliation]
      .filter(Boolean).join(" · ");

    var links = $("hero-links");
    links.textContent = "";
    (PROJECT.links || []).forEach(function (link) {
      var a = el("a", "btn" + (link.primary ? " btn-primary" : ""), "> " + link.label);
      a.href = link.url;
      a.target = "_blank";
      a.rel = "noopener";
      links.appendChild(a);
    });

    /* No counts here. The tallies this page used to lead with — steps, claims
       verified, questions resolved — describe one recorded session, and they
       are on screen in the run view where each one can be clicked and checked.
       Up front they were a scoreboard for a claim the page had not yet made. */
    $("run-blurb").textContent = session.hasRun
      ? "A real review of a real submission, replayed from the archived transcript. "
        + "Step through it and watch the log decide where to read, the paper answer "
        + "back, and the review get assembled from what was settled — including the "
        + "moments the agent went back and withdrew its own findings."
      : "The review of a real submission, beside the log of claims and questions it "
        + "was built from. Every point traces back to the entries behind it.";

    var install = $("install-blocks");
    install.textContent = "";
    (PROJECT.install || []).forEach(function (item) {
      var card = el("div", "install-card");
      card.appendChild(el("h3", null, item.title));
      card.appendChild(el("p", "caption", item.note));
      card.appendChild(el("pre", "cmd", item.code));
      install.appendChild(card);
    });
  }

  /* Two selectors, after the AutoLab live-lab pattern: pick the paper, then
     pick which model reviewed it. Sessions are the cross product, so the model
     list is rebuilt for whichever paper is chosen. */
  function papersInOrder() {
    var seen = {}, out = [];
    SESSIONS.forEach(function (s) {
      if (!seen[s.title]) { seen[s.title] = true; out.push(s.title); }
    });
    return out;
  }

  function modelsFor(title) {
    return SESSIONS.filter(function (s) { return s.title === title; })
                   .map(function (s) { return s.model; });
  }

  function sessionIndex(title, model) {
    for (var i = 0; i < SESSIONS.length; i++) {
      if (SESSIONS[i].title === title && SESSIONS[i].model === model) return i;
    }
    for (var j = 0; j < SESSIONS.length; j++) {
      if (SESSIONS[j].title === title) return j;
    }
    return 0;
  }

  function fill(select, values, chosen) {
    select.textContent = "";
    values.forEach(function (value) {
      var option = el("option", null, value);
      option.setAttribute("value", value);
      if (value === chosen) option.setAttribute("selected", "selected");
      select.appendChild(option);
    });
    select.value = chosen;
    // A single choice is still shown, so the reader can see what produced the
    // run; it is just not a decision.
    select.disabled = values.length < 2;
  }

  function buildSelectors() {
    var session = cur();
    fill($("sel-paper"), papersInOrder(), session.title);
    fill($("sel-model"), modelsFor(session.title), session.model);
  }

  function buildRunStats(session) {
    var host = $("run-stats");
    host.textContent = "";
    var stats = [
      [session.counts.claims, "claims"],
      [session.counts.questions, "questions"],
      [session.counts.notes, "notes"],
      [session.counts.points, "review points"]
    ];
    if (session.hasRun) {
      stats.unshift([session.counts.steps, "steps"]);
      if (session.counts.retractions) {
        stats.push([session.counts.retractions, "retractions"]);
      }
    }
    stats.forEach(function (pair) {
      var group = el("div", "runstat");
      group.appendChild(el("dt", null, String(pair[0])));
      group.appendChild(el("dd", null, pair[1]));
      host.appendChild(group);
    });
  }

  // ── events ───────────────────────────────────────────────────────

  document.addEventListener("click", function (ev) {
    var target = ev.target;

    var viewBtn = target.closest("[data-view-btn]");
    if (viewBtn && !viewBtn.disabled) {
      if (viewBtn.tagName === "A") ev.preventDefault();
      setState({ view: viewBtn.getAttribute("data-view-btn") });
      var panel = $("view-" + viewBtn.getAttribute("data-view-btn"));
      var heading = panel.querySelector("h2[tabindex], .doc h2");
      if (heading) heading.focus();
      return;
    }

    var srcBtn = target.closest("[data-src-btn]");
    if (srcBtn) { setState({ src: srcBtn.getAttribute("data-src-btn") }); return; }

    /* Following a citation into the paper. This is the interaction the whole
       two-column layout exists for: the entry says "§3.2", and §3.2 is right
       there to go and look at. */
    var lineBtn = target.closest("[data-line]");
    if (lineBtn) {
      var card = lineBtn.closest(".entry");
      if (card) state.focus = card.getAttribute("data-id");
      gotoLine(Number(lineBtn.getAttribute("data-line")));
      return;
    }

    var sessionBtn = target.closest("[data-session]");
    if (sessionBtn) { pause(); mountSession(Number(sessionBtn.getAttribute("data-session"))); return; }

    var goto_ = target.closest("[data-goto]");
    if (goto_) {
      pause();
      setState({ view: "watch", step: Number(goto_.getAttribute("data-goto")) });
      $("rail").focus();
      return;
    }

    /* Citation chips, the command chips, and the aim chips all resolve here.
       Log entries live left and review points live right, so following a
       reference has to be able to switch the right column to find one. */
    var cite = target.closest("[data-cite]");
    if (cite) {
      var ref = cite.getAttribute("data-cite");
      var built = dom[state.session];
      var node = built.entries[ref] || built.points[ref];
      if (!node) return;
      var patch = { view: "watch", focus: state.focus === ref ? "" : ref };
      if (built.points[ref] && !built.entries[ref]) patch.src = "review";
      setState(patch);
      node.scrollIntoView({ block: "center", behavior: reduceMotion ? "auto" : "smooth" });
      return;
    }

    var card = target.closest(".entry, .point");
    if (card && !target.closest("summary")) {
      var id = card.getAttribute("data-id");
      setState({ focus: state.focus === id ? "" : id });
      return;
    }

    if (target.closest("#replay-btn")) { setState({ step: 0 }); play(); return; }
    if (target.closest("#prev-btn")) { pause(); gotoStep(state.step - 1); return; }
    if (target.closest("#next-btn")) { pause(); gotoStep(state.step + 1); return; }
    if (target.closest("#theme-btn")) { cycleTheme(); return; }

    var tick = target.closest(".tick");
    if (tick) { pause(); gotoStep(Number(tick.getAttribute("data-n"))); return; }
  });

  document.addEventListener("change", function (ev) {
    var target = ev.target;
    if (target.id === "sel-paper") {
      pause();
      mountSession(sessionIndex(target.value, modelsFor(target.value)[0]));
      return;
    }
    if (target.id === "sel-model") {
      pause();
      mountSession(sessionIndex($("sel-paper").value, target.value));
    }
  });

  $("rail").addEventListener("keydown", function (ev) {
    var session = cur();
    var last = session.steps.length - 1;
    var key = ev.key;
    var handled = true;
    if (key === "ArrowRight" || key === "ArrowUp") gotoStep(state.step + 1);
    else if (key === "ArrowLeft" || key === "ArrowDown") gotoStep(state.step - 1);
    else if (key === "PageUp") gotoStep(state.step + 5);
    else if (key === "PageDown") gotoStep(state.step - 5);
    else if (key === "Home") gotoStep(0);
    else if (key === "End") gotoStep(last);
    else if (key === "j" || key === "J") { var n = nextNotable(state.step, 1); if (n != null) gotoStep(n); }
    else if (key === "k" || key === "K") { var p = nextNotable(state.step, -1); if (p != null) gotoStep(p); }
    else if (key === " ") { state.playing ? pause() : play(); }
    else handled = false;
    if (handled) { ev.preventDefault(); if (key !== " ") pause(); }
  });

  document.addEventListener("keydown", function (ev) {
    if (ev.key === "Escape" && state.focus) setState({ focus: "" });
  });

  ["log-pane", "paper-pane"].forEach(function (id) {
    var node = $(id);
    if (node) node.addEventListener("scroll", function () { scheduleWires(); }, { passive: true });
  });
  window.addEventListener("resize", function () { scheduleWires(); });

  function cycleTheme() {
    var order = ["auto", "light", "dark"];
    var now = document.documentElement.getAttribute("data-theme") || "auto";
    var next = order[(order.indexOf(now) + 1) % order.length];
    document.documentElement.setAttribute("data-theme", next);
    $("theme-btn").setAttribute("aria-label", "Colour theme: " + next);
    try { localStorage.setItem("pr-theme", next); } catch (e) { /* file:// may deny */ }
  }

  // ── boot ─────────────────────────────────────────────────────────

  buildSelectors();
  mountSession(0);
  commit();
})();
