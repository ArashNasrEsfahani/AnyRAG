"""Map each Overfull hbox warning in main.log to the last section/table/figure file opened before it."""
import re

s = open("main.log", encoding="utf-8", errors="ignore").read()
for m in re.finditer(r"Overfull \\hbox \(([\d.]+)pt too wide\) in (\w+) at lines (\d+)--(\d+)", s):
    files = re.findall(r"\((?:\./)?((?:sections|tables|figures)/[\w/]+\.tex)", s[: m.start()])
    print(m.group(1), "pt", m.group(2), "lines", m.group(3), "-", m.group(4), "|", files[-1] if files else "?")
