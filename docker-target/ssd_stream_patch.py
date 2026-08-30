# 共存补丁: 非预制模型(无 ssd-stream.json 清单)时跳过 SSD Stream, 正常服务
from pathlib import Path

p = Path("/usr/local/lib/python3.12/dist-packages/sglang_ssd_stream/config.py")
src = p.read_text()
anchor = (
    "    manifest_path, commit = _resolve_artifact(namespace.model_path, namespace.revision)\n"
    "    config = load_manifest(manifest_path)\n"
)
patched = (
    "    manifest_path, commit = _resolve_artifact(namespace.model_path, namespace.revision)\n"
    "    if not manifest_path.exists():\n"
    "        import logging as _logging\n"
    "\n"
    "        _logging.getLogger(\"sglang_ssd_stream\").warning(\n"
    "            \"no ssd-stream.json under %s: SSD Stream disabled, serving normally\",\n"
    "            manifest_path,\n"
    "        )\n"
    "        return args, kwargs\n"
    "    config = load_manifest(manifest_path)\n"
)
assert anchor in src, "anchor not found - plugin layout changed?"
p.write_text(src.replace(anchor, patched))
print("ssd-stream coexistence patch applied")
