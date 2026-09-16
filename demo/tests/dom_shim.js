/* A minimal DOM good enough to boot demo.js under JavaScriptCore (osascript).
 *
 * The point is not to emulate a browser — it is to run the page's real boot
 * path (buildSelectors -> mountSession -> commit -> applyStep) against the
 * real payload, so that a typo, a null dereference or a wrong method name fails
 * here rather than in front of a reader. Layout is stubbed; structure is not.
 *
 * Usage: osascript -l JavaScript demo/tests/dom_shim.js
 */

ObjC.import('Foundation');

function readFile(path) {
  return $.NSString.stringWithContentsOfFileEncodingError(path, $.NSUTF8StringEncoding, null).js;
}

var REPO = '/Users/haishuo/arxiv2026-ProReviewer';

// ── selector matching ────────────────────────────────────────────────

function parseSimple(sel) {
  var out = { tag: null, id: null, classes: [], attrs: [] };
  var re = /([.#]?[\w-]+)|\[([\w-]+)(?:="([^"]*)")?\]/g, m;
  while ((m = re.exec(sel))) {
    if (m[2]) { out.attrs.push([m[2], m[3]]); continue; }
    var tok = m[1];
    if (tok[0] === '.') out.classes.push(tok.slice(1));
    else if (tok[0] === '#') out.id = tok.slice(1);
    else out.tag = tok.toLowerCase();
  }
  return out;
}

function matchesSimple(node, part) {
  if (part.tag && node.tagName !== part.tag) return false;
  if (part.id && node.id !== part.id) return false;
  for (var i = 0; i < part.classes.length; i++) {
    if (node._classes().indexOf(part.classes[i]) < 0) return false;
  }
  for (var j = 0; j < part.attrs.length; j++) {
    var name = part.attrs[j][0], want = part.attrs[j][1];
    var have = node.getAttribute(name);
    if (have == null) return false;
    if (want != null && String(have) !== want) return false;
  }
  return true;
}

function matches(node, selector) {
  return selector.split(',').some(function (group) {
    var parts = group.trim().split(/\s+/).map(parseSimple);
    if (!matchesSimple(node, parts[parts.length - 1])) return false;
    var cursor = node.parentNode;
    for (var i = parts.length - 2; i >= 0; i--) {
      while (cursor && !matchesSimple(cursor, parts[i])) cursor = cursor.parentNode;
      if (!cursor) return false;
      cursor = cursor.parentNode;
    }
    return true;
  });
}

// ── nodes ────────────────────────────────────────────────────────────

function Element(tag) {
  this.tagName = String(tag).toLowerCase();
  this.attributes = {};
  this.children = [];
  this.parentNode = null;
  this._text = '';
  this.style = { setProperty: function () {} };
  this.dataset = {};
  this.hidden = false;
  this.classList = {
    _self: this,
    add: function (c) { this._self.className = (this._self.className ? this._self.className + ' ' : '') + c; }
  };
  // Layout is not modelled; the values only have to be numbers.
  this.offsetTop = 0; this.offsetHeight = 10; this.clientHeight = 500;
}

Object.defineProperty(Element.prototype, 'className', {
  get: function () { return this.attributes['class'] || ''; },
  set: function (v) { this.attributes['class'] = v; }
});
Object.defineProperty(Element.prototype, 'id', {
  get: function () { return this.attributes.id || ''; },
  set: function (v) { this.attributes.id = v; DOC._byId[v] = this; }
});
Object.defineProperty(Element.prototype, 'textContent', {
  get: function () {
    return this.children.length
      ? this.children.map(function (c) { return c.textContent; }).join('')
      : this._text;
  },
  set: function (v) { this._text = String(v); this.children = []; }
});
/* innerHTML also updates the text, so textContent reads back what a browser
   would show. The hero tagline and the step narration are both set this way —
   without it they assert as empty and any breakage in them is invisible. */
Object.defineProperty(Element.prototype, 'innerHTML', {
  get: function () { return this._html || ''; },
  set: function (v) {
    this._html = String(v);
    this._text = this._html.replace(/<[^>]*>/g, '');
    this.children = [];
  }
});
Object.defineProperty(Element.prototype, 'firstChild', {
  get: function () { return this.children[0] || null; }
});
/* href/target/rel reflect to attributes in a real browser. Without this the
   "no link was invented by the builder" check would pass vacuously, having
   found no links at all. */
['href', 'src', 'target', 'rel', 'title', 'type'].forEach(function (name) {
  Object.defineProperty(Element.prototype, name, {
    get: function () { return this.attributes[name] || null; },
    set: function (v) { this.attributes[name] = String(v); }
  });
});

Element.prototype._classes = function () {
  return (this.attributes['class'] || '').split(/\s+/).filter(Boolean);
};
Element.prototype.setAttribute = function (k, v) {
  this.attributes[k] = String(v);
  if (k === 'id') DOC._byId[v] = this;
};
Element.prototype.getAttribute = function (k) {
  return Object.prototype.hasOwnProperty.call(this.attributes, k) ? this.attributes[k] : null;
};
Element.prototype.removeAttribute = function (k) { delete this.attributes[k]; };
Element.prototype.appendChild = function (child) {
  child.parentNode = this; this.children.push(child); return child;
};
Element.prototype.insertBefore = function (child, ref) {
  var i = this.children.indexOf(ref);
  child.parentNode = this;
  this.children.splice(i < 0 ? this.children.length : i, 0, child);
  return child;
};
Element.prototype.remove = function () {
  if (!this.parentNode) return;
  var i = this.parentNode.children.indexOf(this);
  if (i >= 0) this.parentNode.children.splice(i, 1);
};
Element.prototype._walk = function (out) {
  for (var i = 0; i < this.children.length; i++) {
    out.push(this.children[i]);
    this.children[i]._walk(out);
  }
  return out;
};
Element.prototype.querySelectorAll = function (sel) {
  return this._walk([]).filter(function (n) { return matches(n, sel); });
};
Element.prototype.querySelector = function (sel) {
  return this.querySelectorAll(sel)[0] || null;
};
Element.prototype.closest = function (sel) {
  var node = this;
  while (node) { if (matches(node, sel)) return node; node = node.parentNode; }
  return null;
};
Element.prototype.addEventListener = function () {};
Element.prototype.focus = function () {};
Element.prototype.scrollTo = function () {};
Element.prototype.scrollIntoView = function () {};
/* Geometry is faked but must be *coherent*: the wire code divides by height and
   clamps against column bounds, so returning zeros everywhere would make every
   wire test pass vacuously. Give each node a distinct, plausible box. */
Element.prototype.getBoundingClientRect = function () {
  // The containers are the frame everything else is clamped against, so they
  // get real viewport-sized boxes; giving them 24px like a row would make every
  // wire report itself clipped for a reason that exists only in this shim.
  if (this.id === 'split') return { left: 0, right: 1280, width: 1280, top: 100, bottom: 800, height: 700 };
  if (this.id === 'log-pane') return { left: 40, right: 520, width: 480, top: 160, bottom: 800, height: 640 };
  if (this.id === 'actionbar') return { left: 530, right: 750, width: 220, top: 110, bottom: 800, height: 690 };
  if (this.id === 'act-arrow') return { left: 530, right: 750, width: 220, top: 120, bottom: 152, height: 32 };
  if (this.id === 'paper-pane') return { left: 760, right: 1240, width: 480, top: 160, bottom: 800, height: 640 };
  var seq = (this._seq != null) ? this._seq : (this._seq = (RECT_SEQ += 37) % 560);
  var left = this.closest('#col-src') ? 760 : (this.closest('#actionbar') ? 530 : 40);
  var top = 180 + seq;
  return {
    left: left, right: left + 540, width: 540,
    top: top, bottom: top + 24, height: 24
  };
};
var RECT_SEQ = 0;

// ── document assembled from the real template ────────────────────────

var DOC = {
  _byId: {},
  documentElement: new Element('html'),
  body: new Element('body'),
  createElement: function (tag) { return new Element(tag); },
  createElementNS: function (ns, tag) { return new Element(tag); },
  createTextNode: function (text) {
    var node = new Element('#text');
    node.textContent = String(text);
    return node;
  },
  getElementById: function (id) { return this._byId[id] || null; },
  querySelector: function (s) { return this.body.querySelector(s); },
  querySelectorAll: function (s) { return this.body.querySelectorAll(s); },
  // Captured so the harness can drive the page's real interaction paths.
  _handlers: {},
  addEventListener: function (type, fn) { this._handlers[type] = fn; }
};

function click(node) {
  if (!node) throw new Error('click target does not exist');
  DOC._handlers.click({ target: node, preventDefault: function () {} });
  drain();
}

/* Build the element tree from template.html. A full HTML parser is overkill:
   the template is generated and well-formed, so a tag scanner is sufficient and
   preserves the nesting the JS relies on.

   Literal text between tags is captured too. Much of the page — the loop
   figure, the entry-kind cards, the contrast columns — is hand-written HTML
   rather than generated, and without this every assertion about that copy would
   pass while reading an empty string. */
function buildFromTemplate(html) {
  var body = html.slice(html.indexOf('<body'), html.indexOf('</body>'));
  var stack = [DOC.body];
  var re = /<(\/)?([a-zA-Z0-9-]+)((?:\s+[\w-]+(?:="[^"]*")?)*)\s*(\/)?>/g, m;
  var VOID = ['meta', 'link', 'br', 'hr', 'img', 'input'];
  var cursor = 0;

  function flushText(upto) {
    var raw = body.slice(cursor, upto);
    if (!raw || !raw.replace(/\s+/g, '')) return;
    var text = raw.replace(/<!--[\s\S]*?-->/g, '')
                  .replace(/&amp;/g, '&').replace(/&lt;/g, '<')
                  .replace(/&gt;/g, '>').replace(/&nbsp;/g, ' ')
                  .replace(/\s+/g, ' ');
    if (text.replace(/\s/g, '')) {
      stack[stack.length - 1].appendChild(DOC.createTextNode(text));
    }
  }

  while ((m = re.exec(body))) {
    var closing = m[1], tag = m[2].toLowerCase(), attrs = m[3] || '', self = m[4];
    if (tag !== 'script' && tag !== 'style') flushText(m.index);
    cursor = re.lastIndex;
    if (tag === 'body') continue;
    if (closing) {
      if (stack.length > 1) stack.pop();
      continue;
    }
    var node = new Element(tag);
    var ar = /([\w-]+)(?:="([^"]*)")?/g, a;
    while ((a = ar.exec(attrs))) node.setAttribute(a[1], a[2] == null ? '' : a[2]);
    if (node.attributes.id) DOC._byId[node.attributes.id] = node;
    stack[stack.length - 1].appendChild(node);
    if (!self && VOID.indexOf(tag) < 0 && tag !== 'script') stack.push(node);
  }
}

buildFromTemplate(readFile(REPO + '/demo/assets/template.html'));

// The data island: demo.js reads it through getElementById.
var island = new Element('script');
island.id = 'pr-data';
island.textContent = readFile(REPO + '/demo/index.html')
  .match(/<script type="application\/json" id="pr-data">([\s\S]*?)<\/script>/)[1];
DOC._byId['pr-data'] = island;

// ── globals demo.js touches ──────────────────────────────────────────

var document = DOC;                                    // eslint-disable-line
var window = {                                          // eslint-disable-line
  matchMedia: function () { return { matches: false }; },
  addEventListener: function () {}
};
var localStorage = {                                    // eslint-disable-line
  getItem: function () { return null; },
  setItem: function () {}
};
/* rAF must stay asynchronous. setState() does `frame = requestAnimationFrame(cb)`
   and cb clears `frame`; running cb inline would assign the id *after* it was
   cleared, wedging every later update. Queue and drain instead. */
var _raf = [];
var requestAnimationFrame = function (fn) { _raf.push(fn); return _raf.length; };
var cancelAnimationFrame = function () {};
function drain() {
  var guard = 0;
  while (_raf.length && guard++ < 100) _raf.shift()();
}
var setInterval = function () { return 1; };
var clearInterval = function () {};

// ── run it ───────────────────────────────────────────────────────────

var report = [];
try {
  eval(readFile(REPO + '/demo/assets/demo.js'));
  drain();
  report.push('BOOT OK');
} catch (e) {
  report.push('BOOT FAILED: ' + (e && e.message) + '\n' + (e && e.stack ? e.stack : ''));
}

function count(sel) { return DOC.querySelectorAll(sel).length; }

if (report[0] === 'BOOT OK') {
  var data = JSON.parse(island.textContent).sessions[0];
  var visible = DOC.querySelectorAll('#log-body .entry').filter(function (n) {
    return n.getAttribute('data-present') === '1';
  });
  var points = DOC.querySelectorAll('#run-review-body .point').filter(function (n) {
    return n.getAttribute('data-present') === '1';
  });
  report.push('entries rendered      : ' + count('#log-body .entry') + ' (expect ' + data.entries.length + ')');
  report.push('entries visible at end: ' + visible.length + ' (expect ' + data.entries.length + ')');
  report.push('review points visible : ' + points.length + ' (expect ' + data.review.points.length + ')');
  report.push('rail ticks            : ' + count('#ticks .tick') + ' (expect ' + data.steps.length + ')');
  report.push('steps shown / recorded : ' + data.counts.steps + ' / ' +
    data.counts.recordedSteps + ' (' + data.counts.hiddenSteps + ' hidden)');
  report.push('trim disclosed on page : ' +
    (/hidden of \d+ recorded/.test(DOC.getElementById('rail-help').textContent)
      ? DOC.getElementById('rail-help').textContent.slice(0, 52) : 'NOT DISCLOSED'));
  report.push('first visible step     : ' + data.steps[0].label);
  report.push('notable buttons       : ' + count('#notable-strip button') + ' (expect ' + data.notable.length + ')');
  report.push('select_paper          : ' + DOC.querySelectorAll('#sel-paper option')
    .map(function (o) { return o.textContent; }).join(' | ') +
    (DOC.getElementById('sel-paper').disabled ? '  [single]' : ''));
  report.push('sessions shipped      : ' + JSON.parse(island.textContent).sessions.length);
  report.push('select_model          : ' + DOC.querySelectorAll('#sel-model option')
    .map(function (o) { return o.textContent; }).join(' | ') +
    (DOC.getElementById('sel-model').disabled ? '  [single]' : ''));
  report.push('section label         : ' + DOC.getElementById('run-title').textContent);
  report.push('run stats             : ' + DOC.querySelectorAll('#run-stats .runstat')
    .map(function (x) { return x.textContent; }).join(' · '));
  report.push('status legend chips   : ' + count('#status-legend .chip'));
  report.push('superseded blocks     : ' + count('.superseded') + ' (expect ' + data.notable.length + ')');
  report.push('title                 : ' + data.title);
  report.push('step label            : ' + (DOC.getElementById('stepnum').textContent || '(empty)'));
  report.push('log count             : ' + (DOC.getElementById('log-count').textContent || '(empty)'));
  report.push('review note           : ' + (DOC.getElementById('review-note').textContent || '(empty)'));
  report.push('status caption        : ' + DOC.getElementById('status-caption').textContent.slice(0, 96) + '…');
  var chrome = DOC.querySelector('.topbar').textContent + ' ' +
               DOC.querySelector('.selectors').textContent;
  report.push('score in chrome       : ' + (/\d+\/10/.test(chrome) ? 'LEAKED' : 'absent'));

  // ── scrubbing: the states a reader will actually land on ──────────
  function gotoTick(n) { click(DOC.querySelector('.tick[data-n="' + n + '"]')); }
  function present(sel) {
    return DOC.querySelectorAll(sel).filter(function (x) {
      return x.getAttribute('data-present') === '1';
    });
  }

  report.push('');
  report.push('--- scrubbing ---');

  gotoTick(0);
  report.push('step 1  : ' + present('#log-body .entry').length + ' entries, ' +
    present('#run-review-body .point').length + ' points  (expect 0, 0)');

  var firstAdd = data.steps.filter(function (s) { return s.touched.length; })[0];
  gotoTick(firstAdd.n);
  report.push('step ' + (firstAdd.n + 1) + ' : ' + present('#log-body .entry').length +
    ' entries  (first logging step, expect ' + firstAdd.touched.length + ')');

  // ── left column / right column, and which one moved ───────────────
  report.push('');
  report.push('--- the two columns ---');
  function look() {
    return {
      dir: DOC.getElementById('act-arrow').textContent + ' ' +
           DOC.getElementById('act-dir').textContent,
      lit: (DOC.getElementById('col-log').getAttribute('data-active') === '1' ? 'LEFT/log' : '') +
           (DOC.getElementById('col-src').getAttribute('data-active') === '1' ? 'RIGHT/src' : '') || 'neither',
      tab: DOC.getElementById('col-src').getAttribute('data-src'),
      what: DOC.getElementById('act-what').textContent,
      aims: DOC.querySelectorAll('#act-for .aim').map(function (a) { return a.textContent; }),
      acts: DOC.querySelectorAll('#acts .act').map(function (a) {
        return a.querySelector('.a-id').textContent;
      }),
      src: DOC.getElementById('src-note').textContent,
      now: DOC.querySelectorAll('#paper-body .pline').filter(function (r) {
        return r.getAttribute('data-now') !== '0';
      }).length,
      logcount: DOC.getElementById('log-count').textContent,
      aimed: DOC.querySelectorAll('#log-body .entry').filter(function (e) {
        return e.getAttribute('data-aim') === '1';
      }).map(function (e) { return e.getAttribute('data-id'); })
    };
  }

  report.push('paper lines rendered : ' + count('#paper-body .pline') +
    ' (expect ' + data.paper.totalLines + ')');
  var faint = DOC.querySelectorAll('#paper-body .pline').filter(function (r) {
    return r.getAttribute('data-cov') === '0';
  }).length;
  report.push('never-opened lines   : ' + faint + ' of ' + data.paper.totalLines +
    '  (coverage ' + Math.round(data.paper.coverage * 100) + '%)');

  var readStep = data.steps.filter(function (s) { return s.read && !s.writes; })[0];
  if (readStep) {
    gotoTick(readStep.n);
    var b = look();
    report.push('step ' + (readStep.n + 1) + ' : ' + b.dir + '  lit=' + b.lit +
      '  (expect RIGHT/src — it reached into the paper)');
    report.push('           ' + b.now + ' paper lines marked · ' + b.src);
    report.push('           aims: ' + (b.aims.join(' ') || '(none)') +
      ' · log rows flagged: ' + (b.aimed.join(' ') || '(none)'));
  }

  var writeStep = data.steps.filter(function (s) { return s.writes && s.writes.length > 1 && !s.read; })[0];
  if (writeStep) {
    gotoTick(writeStep.n);
    var w = look();
    report.push('step ' + (writeStep.n + 1) + ' : ' + w.dir + '  lit=' + w.lit +
      '  (expect LEFT/log — it wrote entries back)');
    report.push('           commands: ' + w.acts.join(' ') +
      '  (expect ' + writeStep.writes.length + ')');
    report.push('           paper still showing ' + w.now + ' lines · ' + w.src);
  }

  // The log steering the step: the agent names entries, then acts on them.
  report.push('');
  report.push('--- the log driving the step ---');
  var aimed = data.steps.filter(function (s) { return s.targets && s.targets.length; })[0];
  if (aimed) {
    gotoTick(aimed.n);
    var a = look();
    report.push('step ' + (aimed.n + 1) + ' (aims) : ' + a.dir + ' ' + a.aims.join(' '));
    report.push('           flagged in the log: ' + a.aimed.join(' '));
    var after = data.steps.filter(function (s) { return s.n > aimed.n && s.settles; })[0];
    if (after) {
      gotoTick(after.n);
      var f = look();
      report.push('step ' + (after.n + 1) + ' (acts) : ' + f.dir + ' ' + f.aims.join(' ') +
        '  — carried across the gap');
      report.push('           commands: ' + f.acts.join(' '));
    }
  }
  report.push('unsettled across the run : ' +
    data.steps.map(function (s) { return s.left; }).join(' '));
  var firstPoint = Math.min.apply(null, data.review.points.map(function (p) { return p.born; }));
  report.push('review starts at step ' + (firstPoint + 1) + ', agenda empty from step ' +
    (data.steps.filter(function (s) { return s.left === 0 && s.n > 12; })[0].n + 1));

  // Writing the review is not reading the paper.
  var outlineStep = data.steps.filter(function (s) { return s.points && s.points.length; })[0];
  if (outlineStep) {
    gotoTick(outlineStep.n);
    var o = look();
    report.push('step ' + (outlineStep.n + 1) + ' (review): ' + o.dir + ' lit=' + o.lit +
      ' tab=' + o.tab + ' · paper says "' + o.src + '"');
    report.push('           (expect RIGHT/src — the review it writes lives there, not in the log)');
    report.push('           ' + o.logcount);
  }

  // ── wires and citation links: log <-> paper ───────────────────────
  report.push('');
  report.push('--- the wire between state and environment ---');
  function wires() {
    return DOC.querySelectorAll('#wire-g .wire').map(function (w) {
      var d = w.getAttribute('d') || '';
      var m = d.match(/^M([\d.]+),[\d.]+ C.* ([\d.]+),[\d.]+$/);
      var from = m ? Number(m[1]) : -1, to = m ? Number(m[2]) : -1;
      function side(x) { return x < 530 ? 'log' : (x > 750 ? 'paper' : 'hub'); }
      return side(from) + (w.getAttribute('marker-end') ? '=▶' : '—') + side(to) +
             (w.getAttribute('data-clip') === '1' ? '(clipped)' : '');
    });
  }
  // A read taken while the log already has entries: an empty log has nothing
  // to wire from, so the first read of the run is not the interesting case.
  var rs = data.steps.filter(function (s) { return s.read && s.targets; })[0];
  gotoTick(rs.n);
  report.push('step ' + (rs.n + 1) + ' (fetch): ' + (wires().join(' ') || 'none'));
  report.push('           (expect N log—hub, one hub=\u25B6paper: the entries sent it there)');
  var ws = data.steps.filter(function (s) { return s.writes && !s.read && s.touched.length; })[0];
  gotoTick(ws.n);
  report.push('step ' + (ws.n + 1) + ' (write): ' + (wires().join(' ') || 'none'));
  report.push('           (expect one paper\u2014hub, ' + ws.touched.length +
    ' hub=\u25B6log: the passage produced them)');
  gotoTick(outlineStep ? outlineStep.n : data.steps.length - 1);
  report.push('step ' + ((outlineStep ? outlineStep.n : 0) + 1) +
    ' (compose): ' + (wires().join(' ') || 'none') + '  (expect none — not reading the paper)');

  // Citations resolved at build time, and every one must land on a real line.
  var linked = 0, targets = [];
  data.entries.forEach(function (e) {
    (e.links || []).forEach(function (l) { linked++; targets.push(l); });
    (e.revs || []).forEach(function (r) {
      (r.links || []).forEach(function (l) { linked++; targets.push(l); });
    });
  });
  report.push('');
  report.push('citation links resolved : ' + linked + ' (payload says ' + data.counts.links + ')');
  var offPaper = targets.filter(function (l) { return l.n < 1 || l.n > data.paper.totalLines; });
  report.push('links off the paper     : ' + (offPaper.length ? JSON.stringify(offPaper) : 'none'));
  report.push('rendered link buttons   : ' + count('#log-body .srclink'));
  // Follow one and check the paper actually moved and marked the block.
  var link = DOC.querySelector('#log-body .srclink');
  if (link) {
    click(link);
    var marked = DOC.querySelectorAll('#paper-body .pline').filter(function (r) {
      return r.getAttribute('data-cited') === '1';
    });
    report.push('clicked "' + link.textContent.replace(' ↗', '') + '" -> line ' +
      link.getAttribute('data-line') + ', ' + marked.length + ' lines marked, tab=' +
      DOC.getElementById('col-src').getAttribute('data-src'));
    report.push('  first marked line     : ' +
      (marked[0] ? marked[0].getAttribute('data-n') + ' ' +
        marked[0].querySelector('.tx').textContent.slice(0, 56) : '(none)'));
  }

  // Every command chip must name something the log or the review can show.
  var unresolvable = [];
  data.steps.forEach(function (s) {
    (s.writes || []).forEach(function (x) {
      if (x.v === 'outline') return;                       // summary / overall_score
      var known = data.entries.some(function (e) { return e.id === x.id; }) ||
                  data.review.points.some(function (p) { return p.id === x.id; });
      if (!known) unresolvable.push(s.n + ':' + x.id);
    });
  });
  report.push('command chips naming nothing : ' + (unresolvable.length ? unresolvable.join(', ') : 'none'));

  // Q4: the answer is replaced but the status stays `resolved`, so exactly one
  // chip — the transition pair only appears when the status itself moves.
  var retraction = data.notable.filter(function (n) { return n.kind === 'retraction'; })[0];
  gotoTick(retraction.step);
  var node = DOC.getElementById('e-' + retraction.id);
  var openDetails = node.querySelectorAll('.superseded').filter(function (d) { return d.open; });
  report.push('step ' + (retraction.step + 1) + ' : ' + retraction.id + ' touch=' +
    node.getAttribute('data-touch') + ' superseded-open=' + openDetails.length +
    ' chips=' + node.querySelectorAll('.chip').length + '  (expect 1, 1, 1)');
  report.push('           announcer: ' + DOC.getElementById('announcer').textContent.slice(0, 88));

  // C5 does move status (weak -> supported), so both states must be on screen
  // at once: that co-presence is what makes a flip legible without animation.
  var flip = data.entries.filter(function (e) {
    return e.revs.some(function (r, i) {
      return i > 0 && r.status !== e.revs[i - 1].status && r.change;
    });
  })[0];
  if (flip) {
    var at = flip.revs.filter(function (r) { return r.change; })[0].step;
    gotoTick(at);
    var fnode = DOC.getElementById('e-' + flip.id);
    var chips = fnode.querySelectorAll('.chip-host .chip');
    report.push('step ' + (at + 1) + ' : ' + flip.id + ' shows ' + chips.length + ' chips — ' +
      chips.map(function (c) { return c.getAttribute('data-status'); }).join(' ← ') +
      '  (expect 2: supported ← weak)');
    // One step later the superseded state is gone again.
    gotoTick(at + 1);
    report.push('step ' + (at + 2) + ' : ' + flip.id + ' shows ' +
      fnode.querySelectorAll('.chip-host .chip').length + ' chip  (expect 1)');
  }

  var beforeReview = data.review.points.reduce(function (m, p) {
    return Math.min(m, p.born);
  }, 1e9) - 1;
  gotoTick(beforeReview);
  report.push('step ' + (beforeReview + 1) + ' : ' + present('#run-review-body .point').length +
    ' points just before the review starts  (expect 0)');

  // ── evidence explorer, both directions ────────────────────────────
  report.push('');
  report.push('--- review pane, beside the log ---');
  var multi = Object.keys(data.evidence.entryToPoints).filter(function (k) {
    return data.evidence.entryToPoints[k].length > 2;
  })[0];
  click(DOC.getElementById('e-' + multi));
  var lit = DOC.querySelectorAll('#run-review-body .point').filter(function (p) {
    return p.getAttribute('data-rel') === '1';
  });
  report.push('select ' + multi + '  -> ' + lit.length + ' review points lit  (expect ' +
    data.evidence.entryToPoints[multi].length + ')');

  var point = data.review.points.filter(function (p) { return p.cites.length > 2; })[0];
  click(DOC.getElementById('rp-' + point.id));
  var litEntries = DOC.querySelectorAll('#log-body .entry').filter(function (e) {
    return e.getAttribute('data-rel') === '1';
  });
  report.push('select ' + point.id + '  -> ' + litEntries.length + ' entries lit  (expect ' +
    point.cites.length + ')');

  var badged = DOC.querySelectorAll('#log-body .orphan-badge').map(function (b) {
    return b.closest('.entry').getAttribute('data-id');
  });
  report.push('"not cited" badges -> ' + badged.join(', ') +
    '  (expect ' + data.evidence.orphans.join(', ') + ')');
  report.push('uncited note       -> ' +
    DOC.getElementById('uncited-note').textContent.slice(0, 110));
  report.push('views               : ' + count('[data-view-btn]') + ' (brand + 2 tabs + 1 CTA)');
  report.push('score in review pane: ' +
    (DOC.getElementById('score-box').hidden ? 'HIDDEN' : DOC.getElementById('score-line').textContent));

  report.push('');
  report.push('--- switching papers ---');
  var all = JSON.parse(island.textContent).sessions;
  if (all.length > 1) {
    var sel = DOC.getElementById('sel-paper');
    all.forEach(function (sess, i) {
      sel.value = sess.title;
      DOC._handlers.change({ target: sel });
      drain();
      var shown = DOC.querySelectorAll('#log-body .entry').filter(function (e) {
        return e.getAttribute('data-present') === '1';
      }).length;
      var pts = DOC.querySelectorAll('#run-review-body .point').filter(function (p) {
        return p.getAttribute('data-present') === '1';
      }).length;
      report.push('  ' + sess.title.slice(0, 40) + '  ticks=' +
        count('#ticks .tick') + '/' + sess.counts.steps +
        ' entries=' + shown + '/' + sess.entries.length +
        ' points=' + pts + '/' + sess.review.points.length +
        ' paper=' + count('#paper-body .pline') + '/' + (sess.paper ? sess.paper.totalLines : 0) +
        ' links=' + count('#log-body .srclink'));
    });
    sel.value = all[0].title; DOC._handlers.change({ target: sel }); drain();
  }

  report.push('');
  report.push('--- overview / landing ---');
  var proj = JSON.parse(island.textContent).project || {};
  report.push('hero title    : ' + DOC.getElementById('hero-name').textContent +
              ' — ' + DOC.getElementById('hero-sub').textContent);
  report.push('tagline       : ' + DOC.getElementById('hero-tagline').textContent.slice(0, 70) + '…');
  report.push('authors       : ' + DOC.getElementById('hero-authors').textContent);
  report.push('cta links     : ' + DOC.querySelectorAll('#hero-links a').map(function (a) {
    return a.textContent + ' -> ' + a.getAttribute('href');
  }).join('  |  '));
  report.push('affiliation   : ' + DOC.getElementById('hero-affil').textContent);
  var mimg = DOC.getElementById('method-img');
  var msrc = mimg.getAttribute('src') || '';
  // The shim boots template.html, so data-src is still the placeholder here;
  // that the real data URI lands is checked against the built page instead.
  report.push('method figure : src=' + (msrc || '(none)') +
    ' hidden=' + DOC.getElementById('methodfig').hidden);
  report.push('figure alt    : ' + (mimg.getAttribute('alt') || '').slice(0, 64) + '…');
  report.push('method points : ' + DOC.querySelectorAll('.points .point-card h3')
    .map(function (h) { return h.textContent; }).join(' | '));
  report.push('hero loop     : ' + DOC.querySelectorAll('.hero .loopfig .loop-box strong')
    .map(function (b) { return b.textContent; }).join(' -> '));
  report.push('figure lives in : ' + (DOC.querySelector('.prose .methodfig')
    ? 'the How it works section' : 'NOT in the prose'));
  report.push('figure caption: "' + DOC.querySelector('.methodfig figcaption').textContent.trim() + '"');
  // Two paths to emphasis: literal <strong> in the template, and **markers**
  // in project.json run through inlineMd. Check both — innerHTML is not parsed
  // into elements here, so the generated one is inspected as markup.
  report.push('bold, static  : ' + DOC.querySelectorAll('.prose strong').length + ' phrase(s)');
  ['hero-tagline', 'idea-blurb'].forEach(function (id) {
    var html = DOC.getElementById(id).innerHTML;
    report.push('bold, ' + id.slice(0, 8) + ': ' +
      (html.match(/<strong>/g) || []).length + ' phrase(s)' +
      (html.indexOf('**') >= 0 ? '  ** LEFT LITERAL **' : ''));
  });
  report.push('  e.g. ' + DOC.querySelectorAll('.prose strong')
    .slice(0, 4).map(function (b) { return '"' + b.textContent.trim().slice(0, 34) + '"'; }).join(', '));
  report.push('pull quote    : "' +
    DOC.querySelector('.highlight p').textContent.trim().slice(0, 74) + '…"');
  report.push('card glyphs   : ' + DOC.querySelectorAll('.pc-glyph')
    .map(function (g) { return g.textContent; }).join(' '));
  report.push('subtitle html : ' + DOC.getElementById('hero-sub').innerHTML);
  report.push('entry kinds   : ' + DOC.querySelectorAll('.kinds .kind h3')
    .map(function (h) { return h.textContent; }).join(' '));
  // The run-specific tallies were moved into the run view; a digit reappearing
  // in the landing copy means that decision quietly regressed.
  var landing = DOC.querySelector('.hero').textContent + ' ' +
                DOC.getElementById('run-blurb').textContent;
  report.push('numbers in landing copy : ' +
    (landing.match(/\b\d+\b/g) || ['none']).join(' '));
  report.push('install cards : ' + count('#install-blocks .install-card') +
              ' (expect ' + (proj.install || []).length + ')');
  report.push('bibtex        : ' + (DOC.getElementById('bibtex').textContent.indexOf('@article') === 0
              ? 'present' : 'MISSING'));
  report.push('github nav    : ' + DOC.getElementById('nav-repo').getAttribute('href'));
  report.push('footer        : ' + DOC.getElementById('footer-meta').textContent);
  report.push('run blurb     : ' + DOC.getElementById('run-blurb').textContent.slice(0, 80) + '…');
  var invented = DOC.querySelectorAll('a').map(function (a) { return a.getAttribute('href') || ''; })
    .filter(function (h) { return /^https?:/.test(h); })
    .filter(function (h) { return JSON.stringify(proj).indexOf(h) < 0; });
  report.push('links not in project.json : ' + (invented.length ? invented.join(', ') : 'none'));
}

report.join('\n');
