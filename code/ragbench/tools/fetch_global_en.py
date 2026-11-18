"""Corpus D (natural English): download the global companies' annual reports and add them to the manifest.

    .venv/Scripts/python.exe -m ragbench.tools.fetch_global_en

Sector-matched to the Persian corpus where possible (e-commerce <-> Digikala, telecom <-> Irancell/MCI, delivery <->
Snappfood, and sectors of the development corpus), with three multi-year series for
timeline questions. Files go to corpus/natural/<company>/ (as every other source); each gets a manifest row (lane global_en,
language en) with its sha256. Idempotent: a file already present with a manifest row is skipped.
"""
import csv
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "corpus" / "natural" / "manifest.csv"
DEST = ROOT / "corpus" / "natural" / "global_en"

REPORTS = [  # (company, stem, year, title, url, note)
    ("mowi", "integrated-annual-report", 2022, "Mowi Integrated Annual Report 2022",
     "https://mowi.com/wp-content/uploads/2023/03/Mowi-Integrated-Annual-Report-2022.pdf", "aquaculture"),
    ("mowi", "integrated-annual-report", 2023, "Mowi Integrated Annual Report 2023",
     "https://mowi.com/wp-content/uploads/2024/03/Mowi_Integrated_Annual_Report_2023.pdf", "aquaculture"),
    ("mowi", "integrated-annual-report", 2024, "Mowi Integrated Annual Report 2024",
     "https://mowi.com/wp-content/uploads/2025/03/Mowi-Integrated-Annual-Report-2024.pdf", "aquaculture"),
    ("zalando", "annual-report", 2023, "Zalando Annual Report 2023",
     "https://corporate.zalando.com/sites/default/files/media-download/Zalando-SE_AGM-2024_Annual%20report%202023%20"
     "including%20the%20consolidated%20financial%20statements%20and%20the%20combined%20management%20report%20of%20"
     "Zalando%20SE%20and%20Zalando%20group.pdf", "e-commerce"),
    ("zalando", "annual-report", 2024, "Zalando Annual Report 2024",
     "https://corporate.zalando.com/sites/default/files/media-download/Zalando-SE_AGM-2025_1_Annual-report_2024.pdf",
     "e-commerce"),
    ("arcelormittal", "annual-report", 2023, "ArcelorMittal Annual Report 2023",
     "https://cdn.arcelormittal.com/media/cncdyomr/annual-report-2023_combined.pdf", "steel"),
    ("arcelormittal", "annual-report", 2024, "ArcelorMittal Annual Report 2024",
     "https://cdn.arcelormittal.com/media/dwop2owk/annual-report-2024.pdf", "steel"),
    ("riotinto", "annual-report", 2024, "Rio Tinto Annual Report 2024",
     "https://www.riotinto.com/-/media/content/documents/invest/reports/annual-reports/2024-annual-report.pdf",
     "iron ore / mining"),
    ("turkcell", "form-20f", 2024, "Turkcell Annual Report on Form 20-F 2024",
     "https://s.turkcell.com.tr/SiteAssets/Hakkimizda/yatirimci-iliskileri/documents/pdf/20F2024.pdf", "telecom"),
    ("deliveryhero", "annual-report", 2024, "Delivery Hero Annual Report 2024",
     "https://www.lobbyregister.bundestag.de/media/d6/a7/560286/Delivery-Hero-Annual-Report-2024.pdf",
     "food delivery; copy of the company PDF filed with the German lobby register"),
    # 2026-09-23, second opinion: more COMPANIES before more years (independent families for the holdout)
    ("bhp", "annual-report", 2024, "BHP Annual Report 2024",
     "https://www.bhp.com/-/media/documents/investors/annual-reports/2024/240827_bhpannualreport2024.pdf", "mining"),
    ("vodafone", "annual-report", 2024, "Vodafone Group Plc Annual Report 2024",
     "https://data.fca.org.uk/artefacts/NSM/Portal/NI-000097578/NI-000097578.pdf",
     "telecom; the copy filed with the FCA National Storage Mechanism"),
    ("wise", "annual-report", 2024, "Wise plc Annual Report and Accounts 2024",
     "https://wise.com/imaginary-v2/images/3f1628373b212ca54c1ac73c68d69b72-WISE-2024-Annual-Report-and-Accounts.pdf",
     "payments"),
    ("leroy", "annual-report", 2024, "Leroy Seafood Group Annual Report 2024",
     "https://www.leroyseafood.com/globalassets/02--documents/english/annual-reports/lsg-annual-report-2024.pdf",
     "aquaculture"),
    ("maersk", "annual-report", 2024, "A.P. Moller - Maersk Annual Report 2024",
     "https://www.maersk.com/~/media_sc9/maersk/corporate/sustainability/files/resources/2024/"
     "maersk-annual-report-2024.pdf", "logistics"),
    ("tatasteel", "integrated-report", 2024, "Tata Steel Integrated Report 2023-24",
     "https://www.tatasteel.com/investors/integrated-report-2023-24/pdf/tata-steel-limited-ir-2024.pdf", "steel"),
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    with open(MANIFEST, encoding="utf-8-sig", newline="") as fh:
        rd = csv.DictReader(fh)
        fields, rows = rd.fieldnames, list(rd)
    have = {r["local_path"] for r in rows}
    added = 0
    for company, stem, year, title, url, note in REPORTS:
        rel = "%s\\%s_%s_%d_en.pdf" % (company, company, stem, year)      # the manifest's own form
        p = ROOT / "corpus" / "natural" / company / rel.split("\\")[1]
        if rel in have and p.exists():
            print("have  ", rel)
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research corpus; ragbench)"})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=180) as r, open(str(p) + ".part", "wb") as out:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        out.write(b)
                Path(str(p) + ".part").replace(p)
            except Exception as e:  # noqa: BLE001
                print("FAIL  ", rel, type(e).__name__, str(e)[:120])
                continue
            print("got   ", rel, "%.1f MB" % (p.stat().st_size / 1e6), "%.0fs" % (time.time() - t0))
        with open(p, "rb") as fh:
            if fh.read(5) != b"%PDF-":
                print("NOTPDF", rel)
                continue
        row = {k: "" for k in fields}
        row.update({"lane": "global_en", "company": company, "title_en": title, "year_gregorian": str(year),
                    "language": "en", "format": "pdf", "size_kb": str(p.stat().st_size // 1024), "source_url": url,
                    "local_path": rel, "sha256": sha256(p), "notes": "corpus D (natural English), " + note})
        if rel not in have:
            rows.append(row)
            added += 1
    with open(MANIFEST, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("manifest rows added:", added)


if __name__ == "__main__":
    main()
