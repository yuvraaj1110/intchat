"""Generate app/static/colleges.json: top US colleges with a campus photo + demonym.

Image source: Wikidata's "image" property (P18), which is a representative
campus photo (not a logo), served from Wikimedia Commons. Run occasionally to
refresh:  python3 scripts/fetch_colleges.py
"""

import json
import sys
import time
import urllib.parse
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent.parent / "app" / "static" / "colleges.json"

# (Wikipedia article title, demonym shown as "Oh, so you're a ___!")
COLLEGES = [
    ("Princeton University", "Tiger"),
    ("Massachusetts Institute of Technology", "Engineer"),
    ("Harvard University", "Harvard Crimson"),
    ("Stanford University", "Cardinal"),
    ("Yale University", "Bulldog"),
    ("University of Pennsylvania", "Quaker"),
    ("California Institute of Technology", "Beaver"),
    ("Duke University", "Blue Devil"),
    ("Johns Hopkins University", "Blue Jay"),
    ("Northwestern University", "Wildcat"),
    ("Dartmouth College", "Big Green"),
    ("Brown University", "Bear"),
    ("Vanderbilt University", "Commodore"),
    ("Rice University", "Owl"),
    ("Washington University in St. Louis", "Bear"),
    ("Cornell University", "Big Red"),
    ("Columbia University", "Lion"),
    ("University of Chicago", "Maroon"),
    ("University of California, Berkeley", "Golden Bear"),
    ("University of California, Los Angeles", "Bruin"),
    ("University of Michigan", "Wolverine"),
    ("University of Southern California", "Trojan"),
    ("Carnegie Mellon University", "Tartan"),
    ("University of Virginia", "Cavalier"),
    ("University of North Carolina at Chapel Hill", "Tar Heel"),
    ("Georgetown University", "Hoya"),
    ("University of Notre Dame", "Fighting Irish"),
    ("New York University", "Bobcat"),
    ("University of Florida", "Gator"),
    ("University of Texas at Austin", "Longhorn"),
    ("Georgia Institute of Technology", "Yellow Jacket"),
    ("University of California, San Diego", "Triton"),
    ("University of California, Davis", "Aggie"),
    ("University of California, Irvine", "Anteater"),
    ("University of Illinois Urbana-Champaign", "Fighting Illini"),
    ("University of Wisconsin–Madison", "Badger"),
    ("Boston College", "Eagle"),
    ("Tufts University", "Jumbo"),
    ("Ohio State University", "Buckeye"),
    ("Purdue University", "Boilermaker"),
    ("University of Maryland, College Park", "Terrapin"),
    ("Boston University", "Terrier"),
    ("University of Washington", "Husky"),
    ("Pennsylvania State University", "Nittany Lion"),
    ("University of Georgia", "Bulldog"),
    ("Texas A&M University", "Aggie"),
    ("Emory University", "Eagle"),
    ("Wake Forest University", "Demon Deacon"),
    ("Michigan State University", "Spartan"),
    ("Indiana University Bloomington", "Hoosier"),
]

WIKIDATA = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "intchat-college-bg/1.0 (educational project)"}


def p18_filename(title: str, session: requests.Session) -> str | None:
    """Return the Wikidata P18 (image) filename for a Wikipedia article title.

    Retries with exponential backoff on HTTP 429 (rate limit).
    """
    params = {
        "action": "wbgetentities",
        "sites": "enwiki",
        "titles": title,
        "props": "claims",
        "format": "json",
    }
    for attempt in range(5):
        r = session.get(WIKIDATA, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 429:
            wait = 2 ** attempt
            print(f"    …429, backing off {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        r.raise_for_status()
        entities = r.json().get("entities", {})
        ent = next(iter(entities.values()), {})
        try:
            return ent["claims"]["P18"][0]["mainsnak"]["datavalue"]["value"]
        except (KeyError, IndexError):
            return None
    return None


def commons_url(filename: str, width: int = 1920) -> str:
    return (
        "https://commons.wikimedia.org/wiki/Special:FilePath/"
        + urllib.parse.quote(filename)
        + f"?width={width}"
    )


def main() -> None:
    session = requests.Session()
    out = []
    for title, demonym in COLLEGES:
        name = title.split(" (")[0]
        try:
            fn = p18_filename(title, session)
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠️  {name}: error {exc}", file=sys.stderr)
            fn = None
        if not fn:
            print(f"  ⚠️  {name}: no image, skipped", file=sys.stderr)
            continue
        if fn.lower().endswith(".svg"):
            print(f"  ⚠️  {name}: image is a logo SVG, skipped", file=sys.stderr)
            continue
        out.append({"name": name, "demonym": demonym, "image": commons_url(fn)})
        print(f"  ✓ {name} → {fn}")
        time.sleep(1.0)  # be polite to the API

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(out)} colleges to {OUT}")


if __name__ == "__main__":
    main()
