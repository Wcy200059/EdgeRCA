"""Bundle the static trace dashboard into one file:// friendly HTML."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web" / "trace"

index = json.loads((WEB / "data" / "index.json").read_text())
payloads = {"data/index.json": index}
for case in index["cases"]:
    payloads["data/" + case["name"]] = json.loads((WEB / "data" / case["name"]).read_text())

style = (WEB / "style.css").read_text()
app = (WEB / "app.js").read_text()
html = (WEB / "index.html").read_text()

inline = "<style>\n" + style + "\n</style>"
html = html.replace('<link rel="stylesheet" href="style.css">', inline)

data_json = (json.dumps(payloads, ensure_ascii=False).replace("<", "\\u003c")
             .replace(">", "\\u003e").replace("&", "\\u0026"))
shim = """<script>
/* Single-file build: serve embedded trace payloads without fetch/CORS. */
window.__TRACE_DATA__ = %s;
window.fetch = async function (url) {
  var key = String(url || "");
  if (window.__TRACE_DATA__[key]) {
    return { ok: true, json: async function () { return window.__TRACE_DATA__[key]; } };
  }
  throw new Error("embedded trace payload not found: " + key);
};
</script>""" % data_json
html = html.replace('<script src="app.js"></script>', shim + "\n<script>\n" + app.replace("</script", "<\\/script") + "\n</script>")
out = WEB / "EdgeRCA_trace.html"
out.write_text(html)
print(out, out.stat().st_size)
