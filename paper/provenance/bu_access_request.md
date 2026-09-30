# Provenance: ASLLRP non-manual annotation access

**Status: prepared, not sent.** Compiled 2026-09-30.

## 0. Outage log — the portal is down

**The download portal is unavailable.** This is an external outage on Rutgers' server,
not a project problem, and it blocks the one step that would lift M3's instrument
limitation.

| Date (UTC) | Host | Result |
|---|---|---|
| 2026-09-30 | `dai.cs.rutgers.edu/dai/s/dai` | HTTP 503, Apache/2.4.58 (Ubuntu) — "maintenance downtime or capacity problems" |
| 2026-09-30 | `dai.cs.rutgers.edu/dai/s/continuoussigndownload` | connection timeout (25 s, no response) |
| 2026-09-30 | `dai.cs.rutgers.edu/dai/s/runningstats` | connection timeout (25 s, no response) |
| 2026-09-30 | `dai.cs.rutgers.edu/` (root) | **HTTP 302 in 0.9 s — host is alive** |
| 2026-09-30 | `www.bu.edu/asllrp/` and `/SignStream/3/download-newSS.html` | HTTP 200 — **unaffected** |
| 2026-09-30 09:12 UTC | all of the above, re-checked | **unchanged**: root 302, every `/dai/s/` path times out |
| 2026-09-30 09:25 UTC | re-checked via `scripts/check_dai_portal.py` | **unchanged**, verdict `APP_DOWN` |

Re-checks are now machine-logged to `artifacts/m7a/dai_portal_status.json` by
`scripts/check_dai_portal.py`, so the waiting period leaves a dated record rather than
depending on anyone's memory.

**The handoff for the person doing this task is
[`STEP3_HANDOFF.md`](STEP3_HANDOFF.md)** — self-contained, with the email draft, the
verification command, and the report-back table.

### What the pattern tells us

The host answers in under a second but **every `/dai/s/*` path times out**. Apache is
running and serving; the application behind it is not responding. That distinguishes
"the site is down" from "the application is down", and it is worth telling the contact
because it narrows what is wrong and suggests the outage is not a whole-machine failure.

Independently confirmed by a second person hitting the same 503 on the same date, so
this is not a single-user network problem.

### Consequences

- **M3's instrument limitation stands** until the portal returns. It is currently
  written up honestly as a limitation; nothing needs correcting.
- **No ETA is available to us** and none should be invented. The log records observations
  only, and each new observation gets its own dated row.
- **`defCodingScheme.xml` is not on the critical path.** It is only needed to interpret
  an XML file we do not yet have, so obtaining the SignStream software is second-order
  and can wait. It is also form-gated (name, email, employer) and therefore not
  something to submit on someone's behalf.

### What is NOT blocked

The 1.17 GB DWPose pose corpus on Hugging Face is a **different host and is up** — the
`resolve` endpoint returns HTTP 200 with `x-linked-size: 1169520640`. That is step 4, it
unblocks the scaled M5a test, and it is not waiting on Rutgers. It can start now.

---

> **I did not send the email and cannot.** This record exists so the request is
> reproducible and auditable, and so the research behind it is not lost. The draft in
> §4 must be sent by a person, from an account the project owns, by the person who will
> be named as the requester. Sending an institutional request under someone's identity
> is not something an assistant should do unprompted.

---

## 1. Why this exists

Every non-manual feature in this project — brow raise, brow furrow, head shake, mouth
morpheme — currently comes from **heuristic pseudo-labels** derived from MediaPipe
blendshapes. The artifacts say so explicitly:

```
"marker_provenance": "heuristic (pseudo-labels); ASLLRP SignStream XML not available"
```

This is the largest single instrument limitation in the project, and the reason M3's
gate reads "met on labels, blocked on the visual instrument."

---

## 2. The finding that changes the task

The original plan in `guide.md` step 3 assumed an email request was the only route.
**It is not.** Research on 2026-09-30 found the data is behind a **free self-service
account**, with no documented approval step.

From ASLLRP Report 18 (Neidle & Opoku, May 2020), §8.1:

> "If you are interested in downloading data from DAI 2, you should request a (free)
> account by clicking on the 'login' link. […] The purpose of the account is to help
> keep track of prior downloads and downloads in progress."

> "No account is needed to browse and search all of the data."

No PI letter, no IRB approval, and no data-use agreement is described in any ASLLRP
document.

### So the task is two things, not one

1. **Register a free account** at the DAI 2 portal and download the SignStream XML.
   This is the part that actually unblocks the work, and it needs no one's permission.
2. **Send the email** to confirm the right corpus and to be a good citizen about a
   long-running academic dataset. This is optional for access, but worth doing.

### Two different downloads — this is the key distinction

| Download | URL | Contains non-manuals? |
|---|---|---|
| ASLLRP **SignStream®3 Corpus** | `dai.cs.rutgers.edu/dai/s/dai` | **YES** — XML export has a `<NON_MANUALS>` block |
| **Sign Bank**, signs segmented from continuous signing | `dai.cs.rutgers.edu/dai/s/continuoussigndownload` | **NO** — gloss/handshape/sign-type columns only |

**This explains the project's original obstacle precisely.** The `17,522 sign tokens;
4 signers` figure is the *Sign Bank sign-clip* download, and ASLLRP Report 24 §1.1 shows
its columns are: video ID, glosses, start/end frames, handshapes, sign type — **no
non-manual columns**. The Hugging Face mirror this project already uses was built from
that download. So the mirror is not missing anything; it is simply the wrong artifact.

**The non-manuals were never in what we already have.** They are a different download.

### The XML we need contains, per ASLLRP Report 18 §8.3

```xml
<NON_MANUALS>
  <NON_MANUAL ID='28963' START_FRAME='6504498' END_FRAME='6533527'>
    <LABEL>'head pos: tilt fr/bk'</LABEL>
    <VALUE>'slightly back'</VALUE>
  </NON_MANUAL>
</NON_MANUALS>
```

Time-aligned to frames, which means it can be joined to the landmarks already on disk.

---

## 3. Confirmed facts, and what is still unconfirmed

### Confirmed

| Fact | Source |
|---|---|
| DAI 2 access is a free self-service account | Report 18 §8.1, verbatim above |
| `dai/s/dai` XML export includes non-manuals | Report 18 §8.3, verbatim XML above |
| `continuoussigndownload` (17,522) is manual-only | Report 24 §1.1 column list |
| **Carol Neidle, carol@bu.edu** — Director ASLLRP, Professor Emerita of Linguistics, BU | `bu.edu/asllrp/people.html`, `about-datasets.pdf` (Nov 2025), `about-dai2.html`, `signbank-terms.pdf` |
| Augustine Opoku, augustine.opoku@gmail.com — DAI technical support | Report 18 §9; `people.html` |
| Rutgers hosts the server; the project is directed from BU | Report 18 §9: "The site is hosted by LCSR at Rutgers University" |
| No formal data-request form exists | Searched; only the SignStream *software* download form |

### Unconfirmed — do not assume these

- **The live DAI site was unreachable during this research** (HTTP 503, "maintenance
  downtime or capacity problems", and most requests hung). Every DAI page detail above
  is quoted from ASLLRP's own documentation, not from the live site. The account flow
  needs checking by hand.
- Whether the registration form requires email verification or manual admin approval.
  The form is a screenshot in Report 18, not text.
- The exact login/registration URL — Report 18 only says "clicking on the 'login' link".
- `about-datasets.pdf` (Nov 2025) is **internally inconsistent**: it claims "84
  SignStream® files" then itemises 37 + 10 + 4 = 51. The live count is at
  `dai.cs.rutgers.edu/dai/s/runningstats`, which was down.

### Non-manual field identifiers — partial only

Most useful IDs that could be confirmed without guessing:

| ID | Field | Kind |
|---|---|---|
| 10 | `eye brows` | categorical annotation field |
| 5 | the `-raised` value under `eye brows` | categorical value |
| 40001 | `eye brows` | continuous, normalised to (−1, 1) |
| 40002 | `eye aperture` | continuous |
| 50001 / 50002 / 50003 | `yaw` / `pitch` / `roll` | continuous |

The remaining categorical IDs are defined only in `defCodingScheme.xml`, which ships
inside the SignStream 3 macOS app and is not on the public web. **Do not guess them.**
Download SignStream 3.5.1 from `bu.edu/asllrp/SignStream/3/download-newSS.html`
(free, self-service, MIT-licensed software) to obtain it.

Note that categorical field `10` and continuous field `40001` share the name
`eye brows` and are **separate namespaces**. Do not conflate them.

Field *names* that exist (ASLLRP Report 17, Appendix II), useful for designing the
label mapping:

- **Head:** `head pos: tilt fr/bk`, `head pos: turn`, `head pos: tilt side`,
  `head pos: jut`, `head mvmt: nod`, `head mvmt: shake`, `head mvmt: side to side`
- **Body:** `body lean`, `shoulders`
- **Face:** `eye brows`, `eye gaze`, `eye aperture`, `nose`, `mouth`, `cheeks`
- **Grammatical:** `negative`, `wh question`, `yes-no question`, `rhetorical question`,
  `topic/focus`, `conditional/when`, `role shift`, `adverbial`

This list maps onto the project's existing markers: `eye brows` → brow raise/furrow,
`head mvmt: shake` → head shake, `mouth` → mouth morpheme. **The project's marker
definitions predate this and have never been checked against the real labels** — that
mapping is a task to do, not an assumption to make.

---

## 4. Draft email — for a person to review, edit and send

**To:** `carol@bu.edu`
**CC:** `augustine.opoku@gmail.com` (DAI technical support)
**Subject:** ASLLRP SignStream 3 XML — non-manual annotations, access and field IDs

> Edit before sending. Put your own name, affiliation and contact details in. Do not
> send it verbatim if any detail below is wrong for you.

Dear Professor Neidle,

I am writing about the ASLLRP SignStream®3 corpus, available through the DAI 2 data
access interface.

**First, something practical that may be useful to you.** The DAI 2 interface has been
unavailable for me since 2026-09-30. `dai.cs.rutgers.edu` itself responds — the root
returns a redirect in under a second — but every path under `/dai/s/` times out with no
response, including `/dai/s/dai`, `/dai/s/continuoussigndownload` and
`/dai/s/runningstats`. An Apache 2.4.58 error page with "maintenance downtime or
capacity problems" is returned for at least one of those paths. I mention it in case it
is not already known; I am happy to send the exact responses if useful, and I realise
this may be entirely routine.

My research involves measuring how facial non-manuals and affect markers interact in
American Sign Language, and I am trying to do that with your annotations rather than
with heuristic labels derived from a face tracker. I have built a reproducible
analysis pipeline and I am at the stage where I need the linguistic annotations.

I am not yet able to register a DAI 2 account, because the login flow sits behind the
same unavailable interface. Once it is reachable I intend to do so, but two things I
would be grateful for confirmation on in the meantime:

1. **Is the account route the correct and complete route for the non-manual
   annotations?** The Report 18 documentation describes the XML export as a per-
   collection download, and I want to confirm I am not missing a separate request
   process or a restricted subset before I download a large number of collections. I
   also want to be sure I am downloading the SignStream®3 corpus at
   `/dai/s/dai` rather than the Sign Bank sign-clip download, since Report 24's column
   list for the latter has no non-manual fields and I would like the former.

2. **Field identifiers.** The SignStream 3 XML documentation lists the manual field IDs
   and gives `eye brows` as field 10, plus the continuous IDs 40001 (`eye brows`),
   40002 (`eye aperture`) and 50001–50003 (`yaw`/`pitch`/`roll`). I understand the
   remaining categorical IDs are defined in `defCodingScheme.xml` inside the
   SignStream application, and that I can obtain that by downloading the software. If
   there is a more direct route to that file, or if any non-manual fields are not
   included in the standard XML export, I would rather know now than discover it after
   a long download.

I understand from the terms that the data may be used for research and education but
not redistributed, and that commercial use requires permission. I will not redistribute
it, and I will cite the ASLLRP corpus with the required URLs in any resulting
publication.

For context on the project, I am happy to share a short description of the analysis
and the intended outputs if that would help.

Thank you for your time.

Kind regards,

[Your name]
[Affiliation, or "independent researcher"]
[Contact details]

---

## 5. If access is granted — do not commit the data

The terms are explicit and stricter than they first look:

> "The data available from these pages can be used for research and education purposes,
> but cannot be redistributed without permission. Commercial use, without explicit
> permission, is not allowed, nor are any patents and copyrights based on this
> material."

> "**No Redistribution of Data** — The video data provided through this website may not
> be redistributed, in whole or in part. If you know of others who wish to use the
> data, please refer them to this website."

This covers **the video and the XML annotations both.** Consequences for this repo:

- Store under `data/`, which is already gitignored. **Never commit it.**
- Do not upload it to Hugging Face or any public mirror. The mirror this project reads
  was built from the manual-only download, which is why it has no non-manuals.
- You may **publish analyses**, and may share derived data online provided it is not
  downloadable and attribution is visible.
- The code may read the data and the pipeline may write derived features; those derived
  artifacts are a different question and should be checked before publishing.

### Required citation

> Carol Neidle and Dimitris Metaxas, American Sign Language Linguistic Research Project
> (ASLLRP) Sign Bank © 2022, Boston and Rutgers Universities:
> https://dai.cs.rutgers.edu/dai/s/signbank

plus the URLs: `http://dai.cs.rutgers.edu/`,
`http://dai.cs.rutgers.edu/dai/s/signbank`, `http://www.bu.edu/asllrp/`

Term 5 also requires checking for superseded versions before use.

---

## 6. Record of outcome

Fill in when known, and update `implimentation.md` Track B at the same time.

| Field | Entry |
|---|---|
| Account registered? | date |
| Email sent? | date |
| Reply received? | date |
| Verdict | granted / declined / no reply after 2 weeks |
| Collections downloaded | |
| Did the XML contain `<NON_MANUALS>`? | |
| Defeats the M3 instrument limitation? | |

**If this is declined or goes unanswered for two weeks, that is the answer.** The M3
limitation then stands as final and must be written up as a limitation in the paper,
not left as an open action. M3's gate stays "met on labels, blocked on the visual
instrument" and the heuristic provenance line stays in the artifacts.
