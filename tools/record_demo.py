"""Records one run of chat.py and renders it as a GIF.

Example:
  .venv\\Scripts\\python.exe tools\\record_demo.py

This is a replay, not a screen capture: the script notes when each piece of output arrives and ffmpeg
(which must be on PATH, built with libass) draws the text at those times. Two things differ from the
recorded run: the startup wait before the first generated text is shortened (--startup), and the last frame
is held longer so that it can be read. The timing from the first generated text on is unchanged.
Run it once before recording: the first run after a reboot or a build is much slower than later ones.
"""

import argparse
import codecs
import os
import subprocess
import sys
import tempfile
import textwrap
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WIDTH, FONT_SIZE, LINE_HEIGHT, MARGIN = 800, 18, 18, 16
COLUMNS = 76
FPS = 25  # a GIF stores frame delays in 1/100 s, so 25 fps (4/100 s) is exact


def record(cmd):
    """Runs cmd and returns [(seconds since start, all output so far), ...]."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    events, text = [], ""
    while True:
        chunk = os.read(proc.stdout.fileno(), 4096)  # returns as soon as any data is there
        if not chunk:
            break
        text += decoder.decode(chunk).replace("\r", "")
        events.append((time.perf_counter() - start, text))
    if proc.wait() != 0:
        sys.exit(f"the recorded command failed:\n{text}")
    return events


def shorten_startup(events, prompt, startup):
    """Moves the first generated text to `startup` seconds. Later gaps stay as recorded.

    events[0] must be the prompt alone: chat.py prints it before it starts generate.exe.
    """
    if len(events) < 2 or events[0][1] != prompt:
        sys.exit(f"expected the prompt alone as the first output, got:\n{events[0][1]}")
    cut = max(events[1][0] - startup, 0.0)
    return [(startup / 2, events[0][1])] + [(t - cut, text) for t, text in events[1:]]


def wrap(text):
    lines = []
    for line in text.split("\n"):
        lines += textwrap.wrap(line, COLUMNS) or [""]
    return lines


def ass_time(seconds):
    cs = round(seconds * 100)
    return f"{cs // 360000}:{cs // 6000 % 60:02}:{cs // 100 % 60:02}.{cs % 100:02}"


def make_ass(screens, end, height):
    """screens: [(start time, lines)]; each screen stays until the next one starts."""
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {WIDTH}
PlayResY: {height}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV
Style: Term,Consolas,{FONT_SIZE},&H00E6E6E6,1,0,0,7,{MARGIN},{MARGIN},{MARGIN}

[Events]
Format: Layer, Start, End, Style, Text
"""
    rows = []
    for i, (begin, lines) in enumerate(screens):
        stop = screens[i + 1][0] if i + 1 < len(screens) else end
        body = "\\N".join(line.replace("{", "(").replace("}", ")") for line in lines)
        rows.append(f"Dialogue: 0,{ass_time(begin)},{ass_time(stop)},Term,{body}")
    return head + "\n".join(rows) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max", type=int, default=120, help="number of new tokens")
    ap.add_argument("--startup", type=float, default=1.0, help="wait before the first generated text, s")
    ap.add_argument("--hold", type=float, default=8.0, help="how long the last frame stays, s")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "img", "demo.gif"))
    args = ap.parse_args()

    events = record([sys.executable, os.path.join("tools", "chat.py"), args.prompt, "--max", str(args.max)])
    wait = events[1][0] if len(events) > 1 else 0.0
    events = shorten_startup(events, args.prompt, args.startup)
    shown = f'> python tools/chat.py "{args.prompt}" --max {args.max}'
    screens = [(0.0, [shown])] + [(t, [shown, ""] + wrap(text.rstrip("\n"))) for t, text in events]
    height = max(len(lines) for _, lines in screens) * LINE_HEIGHT + 2 * MARGIN
    height += height % 2
    end = events[-1][0] + args.hold

    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "demo.ass"), "w", encoding="utf-8") as f:
            f.write(make_ass(screens, end, height))
        # The subtitle file is given by a relative name: a drive letter would need escaping in the filter.
        graph = "subtitles=demo.ass,split[a][b];[a]palettegen=max_colors=32[p];[b][p]paletteuse=dither=none"
        source = f"color=c=0x0d1117:s={WIDTH}x{height}:r={FPS}:d={end:.2f}"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", source, "-vf", graph, os.path.abspath(args.out)],
            cwd=tmp,
            check=True,
        )

    print(events[-1][1])
    print(f"recorded wait before the first generated text: {wait:.2f} s, shown as {args.startup:.2f} s")
    print(f"generated text is shown from {events[1][0]:.2f} to {events[-1][0]:.2f} s")
    print(f"{args.out}: {os.path.getsize(args.out) // 1024} KB, {end:.1f} s")


if __name__ == "__main__":
    main()
