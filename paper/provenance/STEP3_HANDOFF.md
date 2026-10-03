# Step 3 handoff — ASLLRP non-manual XML

**Self-contained.** You do not need to read `guide.md`, `plan.md` or the code to do this
task. Everything you need is here.

**For:** whoever on the team owns this.
**Blocks:** M3's instrument limitation. Nothing else.
**Time:** ~30 minutes, including waiting for the account email.
**Status as of 2026-09-30 ~09:35 UTC:** **PORTAL IS UP.** The outage is over — do this now
while it is reachable.

---

## 0. Read this first: the obvious URL is the wrong one

The page you will probably land on first is:

- `dai.cs.rutgers.edu/dai/s/signbank` — **this is the wrong page.** It is the ASL Sign
  Bank (lexical signs). Verified 2026-09-30: its HTML does **not** mention XML or
  annotations anywhere. Downloading from here gets you videos and handshapes, and no
  non-manuals.

Use these instead:

| Page                             | URL                                                     | What it is                                                                                                     |
| -------------------------------- | ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **SignStream® 3 Corpora** | `https://dai.cs.rutgers.edu/dai/s/dai`                | **← the one you want.** Search, Download Cart, XML annotations                                          |
| **Utterance search**       | `https://dai.cs.rutgers.edu/dai/s/utterancesearch`    | Browse by utterance — non-manuals are annotated at**utterance** level, so this is the right browse view |
| Statistics                       | `https://dai.cs.rutgers.edu/dai/s/runningstats`       | Confirms counts (see below)                                                                                    |
| Login / register                 | `https://dai.cs.rutgers.edu/dai/s/index?redirect=dai` | Has a**"Request new account"** button                                                                    |

### This is the right corpus

The statistics page lists, for **"ASLLRP SignStream® 3 Corpus: Continuous Signing"**:

- **17,522** sign tokens
- 1,898 distinct sign entries/variants

**17,522 is exactly the token count in this project's ASLLRP token table.** So this is the
same corpus we have been working with — we have the tokens and need the annotations that
sit alongside them. The Signers are the same four.

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

| Download                                                  | URL                                                 | Has non-manuals?                            |
| --------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------- |
| ASLLRP**SignStream®3 Corpus**                      | `dai.cs.rutgers.edu/dai/s/dai`                    | **YES**                               |
| **Sign Bank** sign clips (the "17,522 / 4 signers") | `dai.cs.rutgers.edu/dai/s/continuoussigndownload` | **NO** — glosses and handshapes only |

The Hugging Face mirror this project already reads was built from the **second** one.
ASLLRP Report 24 lists that download's columns as gloss, start/end frames, handshape and
sign type — no non-manual columns. So the mirror is not missing anything. **You need the
first download, which is a different thing.**

If you download the wrong one you will get files that look plausible and contain no
non-manuals. Step 5 below is the check that catches that.

## 3. Current state of the blocker — RESOLVED

The portal was down through 2026-09-30 morning (three separate checks). **It came back the
same morning**, verified 2026-09-30 ~09:35 UTC:

| Path                                         | Result           |
| -------------------------------------------- | ---------------- |
| `dai.cs.rutgers.edu/dai/s/dai`             | **200** ✅ |
| `dai.cs.rutgers.edu/dai/s/utterancesearch` | **200** ✅ |
| `dai.cs.rutgers.edu/dai/s/runningstats`    | **200** ✅ |
| `www.bu.edu/asllrp/`                       | **200** ✅ |

The history is in `artifacts/m7a/dai_portal_status.json`, written by
`scripts/check_dai_portal.py`. Keep that script — if the site goes down again mid-task,
run `scripts/check_dai_portal.py --watch` and you have dated evidence rather than a
hunch.

> The outage was real and it was worth reporting: the host answered with a 302 while every
> `/dai/s/` path timed out, which points at the application behind Apache rather than the
> machine. Part B's email still includes that report, and it is still worth sending.

---

# Part A — when the portal returns

## A1. Get a free account

1. Open [https://dai.cs.rutgers.edu/dai/s/index?redirect=dai](https://dai.cs.rutgers.edu/dai/s/index?redirect=dai)

   The page says verbatim: *"You must log in to access the feature you have requested.
   You can request a free account if you don't already have one."*
2. Fill in email and password, then click **"Request new account"** (it is a button on
   that same form, next to Login and Reset password).
3. Check your email for the temporary password, then log in and change it. Per Report 18,
   the temporary password is sent by email and you should log in promptly.

**Per ASLLRP Report 18 §8.1, the account exists "to help keep track of prior downloads and
downloads in progress." No PI letter, no IRB, no data-use agreement.**

> Use a real inbox you actually read — the account email carries the temporary password.
> Do not use a throwaway address; if you lose access to the inbox you lose the account.

## A2. Download the SignStream XML

1. Go to [https://dai.cs.rutgers.edu/dai/s/dai](https://dai.cs.rutgers.edu/dai/s/dai) and log in.
2. Prefer [https://dai.cs.rutgers.edu/dai/s/utterancesearch](https://dai.cs.rutgers.edu/dai/s/utterancesearch) to browse — the non-manuals
   are annotated per **utterance**, not per sign token, so the utterance view is where
   they will be.
3. Search, then add the collections you want to the **Download Cart**.
4. From Report 18 §8.2.1, per collection you can choose the SignStream file, **the
   annotations in XML export format**, or both — plus a separate "Video" button.
   **Choose the XML annotations.** Video is optional and is a much larger download; the
   task needs the annotations.

   The four signers in this corpus are Ben, Cory, Jonathan and Rachel — the same four our
   token table resolves. Prefer those four collections if the portal lets you choose.
5. Save under `data/` (which is gitignored). **Do not put it in the repository.**

## A3. Verify you got the right thing — do not skip this

```bash
grep -rl "NON_MANUALS" /path/to/downloaded --include=*.xml | head
```

**Or, better, run the parser — it is already written, tested, and waiting:**

```bash
cd /mnt/Volume2/Sign_Language_EmotionAware
/home/bhuwan/miniconda3/envs/slr/bin/python scripts/parse_signstream.py \
  --path /path/to/downloaded --min-utterances 100
```

`src/seam/data/signstream.py` parses the documented schema and reports what the corpus
*actually* contains: utterances, participants, non-manual events, how many map onto the
project's existing markers, and — importantly — **every label it could not map**. It
writes `artifacts/m3/signstream_report.json`.

> **Expect the mapped fraction to be well under 100%, and that is not a failure.** The
> first label in ASLLRP Report 18's own example, `'head pos: tilt fr/bk'`, has no marker
> in this project, because the vocabulary covers head *movement* (shake, nod) and not head
> *position*. The unmapped list is the honest measure of how far the marker vocabulary is
> from the annotations, and it is what tells you whether M3 can use these labels at all
> or whether the vocabulary needs extending first.

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
[https://www.bu.edu/asllrp/SignStream/3/download-newSS.html](https://www.bu.edu/asllrp/SignStream/3/download-newSS.html)

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

| Field                                          | Value                               |
| ---------------------------------------------- | ----------------------------------- |
| Owner                                          |                                     |
| Date portal first seen up                      |                                     |
| Account registered?                            | date                                |
| Email sent?                                    | date                                |
| Reply?                                         | date                                |
| Collections downloaded                         |                                     |
| **`grep -l NON_MANUALS` found files?** | yes /**no — wrong download** |
| Any field IDs obtained?                        |                                     |
| Files stored at                                | `data/...` (must be gitignored)   |
| `git status` shows nothing data-related?     |                                     |

**A "no" in the `NON_MANUALS` row is a genuinely useful result** — it means the download
route in this document is wrong, and someone needs to know before more time goes into it.

## If it is declined, or silent for two weeks

**That is an answer.** Do not keep the task open. Record it as declined, and M3's
limitation becomes final and must be written up as a limitation in the paper rather than
left as an open action. The current text —
"met on labels, blocked on the visual instrument" — stays accurate and honest.
