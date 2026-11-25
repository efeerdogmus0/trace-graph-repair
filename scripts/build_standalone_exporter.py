"""Write dist/export_flags_standalone.py: export_flags.py with the two helper functions
inlined, for a training machine whose source tree predates bootstrap.py. Prints its sha256."""
import hashlib, inspect, re
from pathlib import Path
from trace_vr import bootstrap

src = Path("scripts/export_flags.py").read_text()
src = src.replace("from trace_vr import arms, bootstrap, dataset", "from trace_vr import arms, dataset, metrics")
src = src.replace("bootstrap.score_flags(", "score_flags(")
# Old VM trees predate arms.run_suffix / arms.select_samples (invariance runs): main set only.
src = src.replace("suffix = arms.run_suffix(split, input_mode, sample_set)",
                  'assert sample_set == "main", "standalone exporter handles the main sample set only"\n'
                  'suffix = split if input_mode == "image" else "%s_%s" % (split, input_mode.replace("+", "_"))')
src = src.replace("rows = arms.select_samples(by_split[split], n_roots, sample_set)",
                  "rows = arms.eval_samples(by_split[split], n_roots)")
# Old VM trees predate the test-split lock; the standalone exporter is used on dev only.
src = src.replace('dataset.check_split_allowed(split, "eval")\n', 'assert not split.startswith("test"), "standalone exporter is for dev splits only"\n')
funcs = inspect.getsource(bootstrap.order_sha) + "\n" + inspect.getsource(bootstrap.score_flags)
funcs = re.sub(r"def (\w+)\(.*?\).*?:\n",
               lambda m: "def %s(%s):\n" % (m.group(1), "samples" if m.group(1) == "order_sha" else "samples, responses"), funcs)
i = src.index("manifest, split")
full = src[:i] + funcs + "\n" + src[i:]
compile(full, "export_flags_standalone.py", "exec")
Path("dist").mkdir(exist_ok=True)
Path("dist/export_flags_standalone.py").write_text(full)
print(hashlib.sha256(full.encode()).hexdigest()[:16])
