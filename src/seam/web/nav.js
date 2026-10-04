/**
 * One navigation definition for every SEAM page, injected at runtime.
 *
 * Why a module and not copy-pasted HTML: `/avatar` shipped without being linked from `/`,
 * without appearing in the `make serve` banner, and without being covered by
 * `make serve-check` - and stayed broken for its whole life partly because nothing pointed
 * at it. Three copies of a nav list drift; one list cannot.
 *
 * Every entry is also probed on the `/routes` page, so a link that 404s is visible rather
 * than merely unfortunate.
 */

export const ROUTES = [
  {
    path: '/',
    title: 'Live demo',
    what: 'Webcam. MediaPipe Tasks runs in the browser; the page posts landmark numbers only, so no video reaches the server.',
    kind: 'page',
  },
  {
    path: '/avatar',
    title: 'SMPL-X avatar',
    what: 'Animated skinned glTF from SMPLer-X, beside the per-clip pipeline measurements. Needs `make demo-avatar` first.',
    kind: 'page',
  },
  {
    path: '/api/coverage',
    title: 'What is and is not claimed',
    what: 'Every cue-to-feature expectation with its rationale, including the cues that have no feature behind them.',
    kind: 'page',
  },
  {
    path: '/routes',
    title: 'Route index',
    what: 'This list, probed live: every route with its current status and size.',
    kind: 'page',
  },
  {
    path: '/api/health',
    title: 'Request contract',
    what: 'The shape /api/analyse expects, and the server version.',
    kind: 'json',
  },
  {
    path: '/api/demo/manifest',
    title: 'Rendered demo manifest',
    what: 'What `make demo-avatar` produced, including a `status` while a run is in progress.',
    kind: 'json',
  },
  {
    path: '/api/cue_expectations',
    title: 'Cue-to-feature map',
    what: 'The declared expectations and their rationales, plus the cues with no feature behind them. Backs the coverage page.',
    kind: 'json',
  },
  {
    path: '/api/analyse',
    title: 'Analyse one window',
    what: 'POST only. 52 blendshapes + head pose for >= 12 frames -> marker readings with per-cue reliability.',
    kind: 'post',
  },
  {
    path: '/api/demo/{name}',
    title: 'One rendered artefact',
    what: 'A .glb or .mp4 from the demo directory. Path traversal is rejected.',
    kind: 'pattern',
  },
  {
    path: '/static/vendor/{path}',
    title: 'Vendored three.js',
    what: 'Pinned locally under src/seam/web/vendor/three. The page needs no CDN and no network.',
    kind: 'pattern',
  },
];

/** The four pages a person actually navigates between. */
const NAV = ['/', '/avatar', '/api/coverage', '/routes'];

export function navHtml(current) {
  const items = NAV.map((p) => {
    const r = ROUTES.find((x) => x.path === p);
    const here = p === current;
    const style = here
      ? 'color:var(--fg);border-color:var(--acc);background:rgba(88,166,255,.10)'
      : 'color:var(--dim);border-color:var(--line)';
    return `<a class="seamnav-link" href="${p}" style="${style}"${
      here ? ' aria-current="page"' : ''
    }>${r ? r.title : p}</a>`;
  }).join('');
  return `<nav class="seamnav" aria-label="SEAM pages">${items}</nav>`;
}

/**
 * Render the nav into every `[data-seamnav]` placeholder.
 *
 * Falls back to plain links if this module fails to load, because a navigation bar that
 * disappears when the network hiccups is the same failure this project keeps hitting.
 */
export function mountNav(current = location.pathname) {
  const targets = document.querySelectorAll('[data-seamnav]');
  if (!targets.length) return;
  let html;
  try {
    html = navHtml(current);
  } catch (e) {
    html = NAV.map((p) => `<a class="seamnav-link" href="${p}">${p}</a>`).join('');
  }
  for (const t of targets) t.outerHTML = html;
}

/** CSS the pages need for the nav, injected once so no page has to repeat it. */
export const NAV_CSS = `
.seamnav{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.seamnav-link{font-size:12px;text-decoration:none;padding:4px 10px;border-radius:6px;
  border:1px solid var(--line);white-space:nowrap}
.seamnav-link:hover{background:rgba(255,255,255,.06);color:var(--fg)}
`;

// Auto-mount. `document.readyState` is checked because the module is deferred, but the
// avatar page injects its nav placeholder from a template built at parse time.
if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => mountNav());
  } else {
    mountNav();
  }
  const style = document.createElement('style');
  style.textContent = NAV_CSS;
  document.head.appendChild(style);
}