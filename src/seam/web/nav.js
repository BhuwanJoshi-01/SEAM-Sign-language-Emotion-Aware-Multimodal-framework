/**
 * One navigation and one theme control for every SEAM page.
 *
 * Why a module and not copy-pasted HTML: `/avatar` shipped without being linked from `/`,
 * without appearing in the `make serve` banner, and without being covered by
 * `make serve-check` — and stayed broken for its whole life partly because nothing pointed
 * at it. Three copies of a nav list drift; one list cannot.
 *
 * Also the single place that knows how a page chrome is *composed*, so the header, the nav
 * and the theme control cannot disagree about spacing or focus order.
 */

export const ROUTES = [
  {
    path: '/live',
    title: 'Live HUD',
    what: 'The standalone real-time demo: face mesh, hands and body tracked in the browser, with the grammar and affect channels read out live. Needs no server; this is the page hosted on GitHub Pages.',
    kind: 'page',
  },
  {
    path: '/',
    title: 'Server demo',
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

/** The pages a person navigates between, in reading order. */
const NAV = ['/', '/avatar', '/api/coverage', '/routes'];

const BRAND = 'SEAM';
const TAGLINE = 'Sign-language non-manual analysis';

function navHtml(current) {
  const links = NAV.map((p) => {
    const r = ROUTES.find((x) => x.path === p);
    const here = p === current;
    return `<a class="seam-navlink" href="${p}"${here ? ' aria-current="page"' : ''}>${
      r ? r.title : p
    }</a>`;
  }).join('');

  return `<header class="seam-header">
  <a class="seam-brand" href="/"><b>${BRAND}</b><span>${TAGLINE}</span></a>
  <nav class="seam-nav" aria-label="Pages">${links}</nav>
  <div class="seam-headeractions">
    <button type="button" class="btn" data-theme-toggle aria-label="Switch theme">Light</button>
  </div>
</header>`;
}

/**
 * Render the header into every `[data-seam-header]` placeholder.
 *
 * Falls back to plain links if this module fails to load, because a navigation bar that
 * disappears when the network hiccups is the same failure this project keeps hitting.
 */
export function mountNav(current = location.pathname) {
  const targets = document.querySelectorAll('[data-seam-header]');
  if (!targets.length) return;
  let html;
  try {
    html = navHtml(current);
  } catch (e) {
    html = `<header class="seam-header"><a class="seam-brand" href="/"><b>${BRAND}</b></a>
      <nav class="seam-nav" aria-label="Pages">${NAV.map(
        (p) => `<a class="seam-navlink" href="${p}">${p}</a>`,
      ).join('')}</nav></header>`;
  }
  for (const t of targets) t.outerHTML = html;
}

/* Auto-mount. Deferred module scripts run after parsing, so `document.body` exists. */
if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => mountNav());
  } else {
    mountNav();
  }
  // theme.js binds [data-theme-toggle] on DOMContentLoaded too; dispatch once it is up so
  // the button label reflects the applied theme rather than the initial 'Light' default.
  import('./theme.js').then((t) => t.mount());
}