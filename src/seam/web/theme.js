/**
 * Theme resolution, persistence, and the toggle every page shares.
 *
 * Precedence: an explicit choice wins, then the reader's system preference, then dark.
 * The explicit choice is stored, so a reader who picks light in a dark room does not get
 * overridden by their OS at the next load — a toggle that loses to the system is a bug, not
 * a preference.
 *
 * The default is *not* a category choice. This is a research instrument read at a desk, so
 * it follows the reader. What it must never do is surprise: `apply()` runs from an inline
 * copy in each page's `<head>` before first paint, so there is no flash of the wrong theme.
 */

const KEY = 'seam.theme';
const ORDER = ['dark', 'light'];
const QUERY = '(prefers-color-scheme: light)';

/** The theme in force right now. */
export function current() {
  const attr = document.documentElement.dataset.theme;
  if (attr === 'light' || attr === 'dark') return attr;
  return window.matchMedia?.(QUERY).matches ? 'light' : 'dark';
}

/**
 * `?theme=light` / `?theme=dark` — an explicit override that wins over everything.
 *
 * Not test scaffolding. The M7 preference study runs several raters against the same
 * stimuli, and a rater whose OS is in light mode seeing a different stage background and a
 * different light rig from the next rater is a confounded experiment. The study harness pins
 * the theme with this parameter so every session is identical, and it is also what makes a
 * screenshot of "the light theme" reproducible.
 */
function override() {
  try {
    const q = new URLSearchParams(location.search).get('theme');
    return q === 'light' || q === 'dark' ? q : null;
  } catch {
    return null;
  }
}

/** Resolve the theme a first-time visitor should get. */
export function preferred() {
  const forced = override();
  if (forced) return forced;
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === 'light' || saved === 'dark') return saved;
  } catch {
    /* Private mode, or storage disabled. Not worth failing a page render over. */
  }
  return window.matchMedia?.(QUERY).matches ? 'light' : 'dark';
}

/** Write the theme to the document. Cheap and idempotent, so it is safe to call often. */
export function apply(theme) {
  document.documentElement.dataset.theme = theme;
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) {
    const bg = getComputedStyle(document.documentElement)
      .getPropertyValue('--bg')
      .trim();
    if (bg) meta.setAttribute('content', bg);
  }
  for (const el of document.querySelectorAll('[data-theme-toggle]')) {
    el.textContent = theme === 'dark' ? 'Light' : 'Dark';
    el.setAttribute(
      'aria-label',
      theme === 'dark' ? 'Switch to the light theme' : 'Switch to the dark theme',
    );
  }
  document.dispatchEvent(new CustomEvent('seam:theme', { detail: { theme } }));
}

/** Set and remember. */
export function set(theme) {
  try {
    localStorage.setItem(KEY, theme);
  } catch {
    /* Non-fatal: the theme still applies for this page view. */
  }
  apply(theme);
}

export function toggle() {
  set(current() === 'dark' ? 'light' : 'dark');
}

/** Follow the system while the reader has expressed no preference of their own. */
function watchSystem() {
  const mq = window.matchMedia?.(QUERY);
  if (!mq) return;
  const onChange = () => {
    let saved = null;
    try {
      saved = localStorage.getItem(KEY);
    } catch {
      /* ignore */
    }
    if (saved !== 'light' && saved !== 'dark') apply(current());
  };
  mq.addEventListener?.('change', onChange);
}

/**
 * Read a token from the active theme.
 *
 * The 3D stage needs this: `scene.background` and the light colours are numbers handed to
 * three.js, so they cannot come from a CSS variable automatically. Without it the canvas
 * stays dark in the light theme and the page looks broken in exactly one of two themes.
 */
export function token(name, fallback = '') {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

/** Numeric token for three.js — `#rrggbb` or `#rrggbbaa` to a 0xRRGGBB int. */
export function tokenHex(name, fallback = 0x000000) {
  const v = token(name);
  const m = /^#([0-9a-f]{6})/i.exec(v);
  return m ? parseInt(m[1], 16) : fallback;
}

let mounted = false;

export function mount() {
  if (mounted) return;
  mounted = true;
  for (const el of document.querySelectorAll('[data-theme-toggle]')) {
    el.addEventListener('click', toggle);
  }
  watchSystem();
  apply(current());
}

/* Auto-mount. `nav.js` does the same for itself; both are idempotent. */
if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }
}

/**
 * The inline snippet every page puts in <head>.
 *
 * Kept as a string rather than a file because it must run synchronously before first paint,
 * and an external module would arrive after the document has already painted. Duplicated
 * into four pages on purpose: it is five lines and a wrong theme flash is worse than the
 * duplication.
 */
export const BOOT = `(function(){try{var k='seam.theme',s=localStorage.getItem(k);
var q=new URLSearchParams(location.search).get('theme');
var t=(q==='light'||q==='dark')?q:((s==='light'||s==='dark')?s:(matchMedia('(prefers-color-scheme: light)').matches?'light':'dark'));
document.documentElement.dataset.theme=t;}catch(e){}})();`;