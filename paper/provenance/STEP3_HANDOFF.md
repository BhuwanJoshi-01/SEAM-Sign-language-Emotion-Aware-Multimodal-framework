# Step 3 handoff — ASLLRP non-manual XML

**Self-contained.** You do not need to read `guide.md`, `plan.md` or the code to do this
task. Everything you need is here.

**For:** whoever on the team owns this.
**Blocks:** M3's instrument limitation. Nothing else.
**Time:** ~20 minutes of work, then it is a waiting game.
**Status as of 2026-09-30 09:25 UTC:** `BLOCKED — EXTERNAL`. The Rutgers portal is down.

---

## 1. What the task is, and why it is worth doing

Every non-manual feature in this project — brow raise, brow furrow, head shake, mouth
morpheme — is currently a **heuristic guess** derived from a face tracker's blendshape
scores. The artifacts say so in as many words:

```
"marker_provenance": "heuristic (pseudo-labels); ASLLRP SignStream XML not available"
```

The authoritative human annotations exist. They are distributed by Boston University
through a Rutgers data portal. Getting them turns M3's labels from guesses into
measurements, and would let us check whether our marker definitions are even correct.

## 2. The single most important thing to understand

**The data we already have does not contain the non-manuals, and that is not a mirror
problem.** There are two different downloads:

| Download | URL | Has non-manuals? |
|---|---|---|
| ASLLRP **SignStream®3 Corpus** | `dai.cs.rutgers.edu/dai/s/dai` | **YES** |
| **Sign Bank** sign clips (the "17,522 / 4 signers") | `dai.cs.rutgers.edu/dai/s/continuoussigndownload` | **NO** — glosses and handshapes only |

The Hugging Face mirror this project already reads was built from the **second** one.
ASLLRP Report 24 lists that download's columns as gloss, start/end frames, handshape and
sign type — no non-manual columns. So the mirror is not missing anything. **You need the
first download, which is a different thing.**

If you download the wrong one you will get files that look plausible and contain no
non-manuals. Step 5 below is the check that catches that.

## 3. Current state of the blocker

The portal is down. Measured repeatedly on 2026-09-30:

| Path | Result |
|---|---|
| `dai.cs.rutgers.edu/` | **302 in ~1.3 s — the host is alive** |
| `dai.cs.rutgers.edu/dai/s/dai` | timeout, no response |
| `dai.cs.rutgers.edu/dai/s/continuoussigndownload` | timeout, no response |
| `dai.cs.rutgers.edu/dai/s/runningstats` | timeout, no response |
| `www.bu.edu/asllrp/` | **200 — Boston's site is fine** |

So the web server is running and the **application behind it is not responding**. That is
not our problem to fix, and it is not blocked on anyone on our team. Confirmed
independently by two people on the same date.

### Check it yourself (one command)

```bash
cd /mnt/Volume2/Sign_Language_EmotionAware
/home/bhuwan/miniconda3/envs/slr/bin/python scripts/check_dai_portal.py
```

It prints a verdict and appends a dated row to `artifacts/m7a/dai_portal_status.json`,
so the waiting period leaves a record instead of being remembered. To poll until it
comes back:

```bash
/home/bhuwan/miniconda3/envs/slr/bin/python scripts/check_dai_portal.py --watch
```

**`APP_DOWN` → keep waiting. `PORTAL_UP` → go to Part A.**

---

# Part A — when the portal returns

## A1. Get a free account

1. Open <https://dai.cs.rutgers.edu/dai/s/dai>
2. Click **login**, then request a free account.

Per ASLLRP Report 18 §8.1, the account exists "to help keep track of prior downloads and
downloads in progress." **No PI letter, no IRB, no data-use agreement.** You do not need
permission from anyone.

## A2. Download the SignStream XML

In the portal, per SignStream collection, choose the **XML annotations** download. From
Report 18 §8.2.1 you can pick the SignStream file, the XML export, or both — **choose the
XML.** That is the point of the whole task.

Save under `data/` (which is gitignored). **Do not put it in the repository.**

## A3. Verify you got the right thing — do not skip this

```bash
cd /mnt/Volume2/Sign_Language_EmotionAware
grep -l "NON_MANUALS" data/<wherever-you-downloaded>/*.xml | head
```

**This must print at least one file.** The XML looks like this (Report 18 §8.3):

```xml
<NON_MANUALS>
  <NON_MANUAL ID='28963' START_FRAME='6504498' END_FRAME='6533527'>
    <LABEL>'head pos: tilt fr/bk'</LABEL>
    <VALUE>'slightly back'</VALUE>
  </NON_MANUAL>
</NON_MANUALS>
```

Note it is **time-aligned to frames**, which means it can be joined to the video
landmarks already on disk.

> **If `grep` prints nothing, you have the wrong download.** Stop and report it. Do not
> proceed as though you have labels — that would put a false claim into M3's write-up,
> and the whole reason this task exists is to stop claiming things we cannot support.

## A4. Optional: the field IDs

Most non-manual field IDs are defined only in `defCodingScheme.xml`, which ships inside
the SignStream 3 desktop app (macOS, free, self-service, MIT-licensed):
<https://www.bu.edu/asllrp/SignStream/3/download-newSS.html>

Already confirmed without guessing: `10` = `eye brows` (categorical), `40001` =
`eye brows`, `40002` = `eye aperture`, `50001/50002/50003` = `yaw`/`pitch`/`roll`
(continuous). **Note `10` and `40001` share a name and are separate namespaces** — do not
conflate them.

**Do not guess the remaining IDs.** This is not on the critical path; it only matters once
you hold an XML file to interpret.

---

# Part B — send the email (do this now, it does not need the portal)

The email is worth sending even though the portal is down, for two reasons: reporting a
downed portal is genuinely useful to them, and **the field-ID question can be answered
without the download ever happening.**

**To:** `carol@bu.edu` — Carol Neidle, Director of ASLLRP, Professor Emerita of Linguistics,
Boston University. Confirmed as the current contact across several November 2025 sources.

**CC:** `augustine.opoku@gmail.com` — DAI technical support.

**Do not email the Rutgers developers.** Gregory Dimitriadis and Douglas Motto built the
SignStream software; they do not handle data requests.

**Subject:** ASLLRP SignStream 3 XML — non-manual annotations, access and field IDs

### Draft — edit the bracketed parts, then send

> Dear Professor Neidle,
>
> I am writing about the ASLLRP SignStream®3 corpus, available through the DAI 2 data
> access interface.
>
> **First, something practical that may be useful to you.** The DAI 2 interface has been
> unavailable for me since 2026-09-30. `dai.cs.rutgers.edu` itself responds — the root
> returns a 302 redirect in about a second — but every path under `/dai/s/` times out with
> no response, including `/dai/s/dai`, `/dai/s/continuoussigndownload` and
> `/dai/s/runningstats`. An Apache 2.4.58 page with "maintenance downtime or capacity
> problems" is returned for at least one of those paths. I mention it in case it is not
> already known; I am happy to send the exact responses if useful, and I realise this may
> be entirely routine.
>
> My research involves measuring how facial non-manuals and affect markers interact in
> American Sign Language, and I am trying to do that with your annotations rather than
> with heuristic labels derived from a face tracker. I have built a reproducible analysis
> pipeline and I am at the stage where I need the linguistic annotations.
>
> I am not yet able to register a DAI 2 account, because the login flow sits behind the
> same unavailable interface. Once it is reachable I intend to do so, but two things I
> would be grateful for confirmation on in the meantime:
>
> 1. **Is the account route the correct and complete route for the non-manual
>    annotations?** The Report 18 documentation describes the XML export as a
>    per-collection download, and I want to confirm I am not missing a separate request
>    process or a restricted subset before I download a large number of collections. I
>    also want to be sure I am downloading the SignStream®3 corpus at `/dai/s/dai` rather
>    than the Sign Bank sign-clip download, since Report 24's column list for the latter
>    has no non-manual fields and I would like the former.
>
> 2. **Field identifiers.** The SignStream 3 XML documentation lists the manual field IDs
>    and gives `eye brows` as field 10, plus the continuous IDs 40001 (`eye brows`),
>    40002 (`eye aperture`) and 50001–50003 (`yaw`/`pitch`/`roll`). I understand the
>    remaining categorical IDs are defined in `defCodingScheme.xml` inside the SignStream
>    application, and that I can obtain that by downloading the software. If there is a
>    more direct route to that file, or if any non-manual fields are not included in the
>    standard XML export, I would rather know now than discover it after a long download.
>
> I understand from the terms that the data may be used for research and education but not
> redistributed, and that commercial use requires permission. I will not redistribute it,
> and I will cite the ASLLRP corpus with the required URLs in any resulting publication.
>
> For context on the project, I am happy to share a short description of the analysis and
> the intended outputs if that would help.
>
> Thank you for your time.
>
> Kind regards,
>
> [Your name]
> [Affiliation, or "independent researcher"]
> [Contact details]

---

## Do not commit the data

The terms are explicit and stricter than they look:

> "The data available from these pages can be used for research and education purposes,
> but cannot be redistributed without permission."

> "**No Redistribution of Data** — The video data provided through this website may not be
> redistributed, in whole or in part."

That covers **the video and the XML annotations both.** So:

- Keep it under `data/` (already gitignored). **Never `git add` it.**
- **Never upload it to Hugging Face or any public mirror.** The existing mirror is public
  only because it was built from the manual-only download — which is exactly why it has
  no non-manuals. Putting these files there would breach the licence.
- You may **publish analyses**, and may share derived data online provided it is not
  downloadable and attribution is visible.

**Required citation:**

> Carol Neidle and Dimitris Metaxas, American Sign Language Linguistic Research Project
> (ASLLRP) Sign Bank © 2022, Boston and Rutgers Universities:
> https://dai.cs.rutgers.edu/dai/s/signbank

plus `http://dai.cs.rutgers.edu/`, `http://dai.cs.rutgers.edu/dai/s/signbank`,
`http://www.bu.edu/asllrp/`. The terms also require checking for superseded versions.

---

# Report back

Fill this in and put it in `report.md` under **Step 3**.

| Field | Value |
|---|---|
| Owner | |
| Date portal first seen up | |
| Account registered? | date |
| Email sent? | date |
| Reply? | date |
| Collections downloaded | |
| **`grep -l NON_MANUALS` found files?** | yes / **no — wrong download** |
| Any field IDs obtained? | |
| Files stored at | `data/...` (must be gitignored) |
| `git status` shows nothing data-related? | |

**A "no" in the `NON_MANUALS` row is a genuinely useful result** — it means the download
route in this document is wrong, and someone needs to know before more time goes into it.

## If it is declined, or silent for two weeks

**That is an answer.** Do not keep the task open. Record it as declined, and M3's
limitation becomes final and must be written up as a limitation in the paper rather than
left as an open action. The current text —
"met on labels, blocked on the visual instrument" — stays accurate and honest.
