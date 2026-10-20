"""External baseline runner; no registry or campaign side effects."""
from __future__ import annotations
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
import numpy as np

def _read(path):
    return json.loads(Path(path).read_text())
def _read_route(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open() as handle:
            return [json.loads(line) for line in handle if line.strip()]
    return _read(path)

def _git_pin(path, expected):
    if not expected: return "not_declared"
    try:
        actual=subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc: raise ValueError(f"author checkout is not a readable git checkout: {path}") from exc
    if actual != expected: raise ValueError(f"author pin mismatch: expected {expected}, found {actual}")
    return actual

def _route_manifest(path):
    manifest=Path(path).parent/"converter_manifest.json"
    if not manifest.exists(): raise ValueError(f"fixed-window route lacks converter_manifest.json: {path}")
    return json.loads(manifest.read_text())

def _validate_route(path, *, horizon, lower=None, upper=None, reject_empty=False):
    path=Path(path); manifest=_route_manifest(path)
    split=path.stem; expected=manifest.get("splits",{}).get(split,{})
    import hashlib
    digest=hashlib.sha256(); windows=events=empties=0
    with path.open("rb") as handle:
        for line_no,line in enumerate(handle,1):
            digest.update(line)
            if not line.strip(): continue
            row=json.loads(line); times=np.asarray(row.get("times",[]),float); loc=np.asarray(row.get("locations",[]),float)
            if loc.size == 0: loc=loc.reshape(0,2)
            if loc.shape != (len(times),2): raise ValueError(f"{path}:{line_no}: locations must be N×2")
            if not np.isfinite(times).all() or not np.isfinite(loc).all() or np.any(times <= 0) or np.any(times >= horizon) or np.any(np.diff(times) <= 0):
                raise ValueError(f"{path}:{line_no}: fixed-horizon route requires finite strictly increasing 0<t<{horizon}")
            if lower is not None and (np.any(loc < lower) or np.any(loc > upper)):
                raise ValueError(f"{path}:{line_no}: location outside declared contract support")
            windows += 1; events += len(times); empties += not bool(len(times))
    if digest.hexdigest() != expected.get("sha256"): raise ValueError(f"route hash mismatch: {path}")
    if reject_empty and empties: raise ValueError(f"WSM launch gate: {path} contains {empties} empty windows")
    return {"windows":windows,"events":events,"empty_windows":empties,"sha256":digest.hexdigest()}

def validate_spec(spec, *, stage="validate"):
    """Validate a frozen external spec without fitting or opening a checkpoint."""
    kind=spec.get("kind"); splits=spec.get("splits")
    if kind not in {"wsm","tritpp","etas"}:
        raise ValueError("unsupported external baseline kind")
    if kind != "etas" and (not isinstance(splits,dict) or not {"train","val","test"}.issubset(splits)):
        raise ValueError("external spec needs kind and train/val/test route paths")
    if kind != "etas":
        selected=("train","val") if stage == "fit" else ("test",) if stage == "evaluate" else ("train","val","test")
        splits={key:splits[key] for key in selected}
        expected=spec.get("converter_manifest_sha256")
        if expected:
            import hashlib
            for folder in {Path(path).parent for path in splits.values()}:
                if hashlib.sha256((folder/"converter_manifest.json").read_bytes()).hexdigest()!=expected:
                    raise ValueError("Frozen converter manifest changed.")
    if kind == "wsm":
        from .wsm_author import WSMContract, WSMAuthorAdapter
        contract=WSMContract(**spec["contract"]); contract.validate()
        _git_pin(spec["authorcheckout"],spec.get("author_commit_required"))
        # Importing author classes catches unrecorded dependency/API drift.
        WSMAuthorAdapter(contract,spec["authorcheckout"])._imports()
        lo=np.asarray(contract.spatial_lower,float); hi=np.asarray(contract.spatial_upper,float)
        checks={name:_validate_route(path,horizon=contract.horizon,lower=lo,upper=hi,reject_empty=True) for name,path in splits.items()}
    elif kind == "tritpp":
        root=spec.get("authorcheckout")
        from .tritpp import TriTPPContract, _author_ttpp
        from .provenance import TRITPP_PROVENANCE
        if not root: raise ValueError("TriTPP spec must declare authorcheckout for pinned-source verification")
        expected=spec.get("author_commit_required", TRITPP_PROVENANCE.commit)
        _git_pin(root, expected)
        if str(root) not in sys.path: sys.path.insert(0,str(root))
        contract=TriTPPContract(**spec["contract"]); contract.validate(); _author_ttpp()
        checks={name:_validate_route(path,horizon=contract.t_max) for name,path in splits.items()}
    else:
        # ETAS consumes its separately frozen CSV/poly route, not fixed windows.
        for key in ("catalog_csv","contract"):
            if key not in spec: raise ValueError(f"ETAS spec missing {key}")
        if not Path(spec["catalog_csv"]).is_file(): raise ValueError("ETAS frozen catalogue is missing")
        if spec.get("catalog_sha256"):
            import hashlib
            if hashlib.sha256(Path(spec["catalog_csv"]).read_bytes()).hexdigest()!=spec["catalog_sha256"]:
                raise ValueError("ETAS catalogue changed after freeze.")
        _git_pin(spec["authorcheckout"], spec.get("author_commit"))
        checks={"catalog_csv":str(spec["catalog_csv"])}
    resource_status = "not_applicable"
    if kind in {"wsm", "tritpp"}:
        steps, seconds = spec.get("max_steps"), spec.get("max_time_seconds")
        if not isinstance(steps, int) or steps < 1 or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("external fit spec needs positive max_steps and max_time_seconds plus an outer watchdog")
        resource_status = "declared_adapter_cap; require compiler outer-watchdog status artifact"
    return {"status":"pass","kind":kind,"checks":checks,"resource_cap_status":resource_status}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--spec",required=True); p.add_argument("--out",required=True); p.add_argument("--checkpoint"); p.add_argument("--stage",choices=("fit","evaluate","validate"),default="fit"); a=p.parse_args(); s=_read(a.spec); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    if a.stage == "validate":
        print(json.dumps(validate_spec(s),indent=2)); return
    if s.get("execution_gate", "ready") != "ready":
        raise ValueError(s.get("execution_gate_reason", s["execution_gate"]))
    validate_spec(s, stage=a.stage)
    if s["kind"] != "etas":
        import torch
        torch.set_num_threads(4)
    if s["kind"]=="wsm":
        import torch
        from .wsm_author import WSMAuthorAdapter, WSMContract
        if "seed" in s: np.random.seed(int(s["seed"])); torch.manual_seed(int(s["seed"]))
        ad=WSMAuthorAdapter(WSMContract(**s["contract"]),s["authorcheckout"])
        if a.stage=="fit":
            splits={k:_read_route(v) if isinstance(v,str) else v for k,v in s["splits"].items() if k in {"train","val"}}
            if not {"train", "val"}.issubset(splits): raise ValueError("fit requires train and val routes")
            model,metrics=ad.fit({"train":splits["train"],"val":splits["val"]},estimator=s["estimator"],epochs=s.get("epochs",1),lr=s.get("lr",1e-3),batch_size=s.get("batch",64),work_dir=out/"prepared",max_steps=s.get("max_steps"),max_time_seconds=s.get("max_time_seconds")); ckpt=ad.save(model,metrics,out,estimator=s["estimator"]); (out/"metrics.json").write_text(json.dumps({**metrics,"checkpoint":str(ckpt)},indent=2))
        else:
            splits={"test":_read_route(s["splits"]["test"])}
            if "test" not in splits: raise ValueError("evaluate requires a test route")
            checkpoint=a.checkpoint or s.get("checkpoint")
            if not checkpoint: raise ValueError("evaluate requires --checkpoint from one completed refit; never evaluate all HPO trials")
            ad,model,payload=WSMAuthorAdapter.load(checkpoint); (out/"score.json").write_text(json.dumps({"test_loglik":ad.score(model,splits["test"],payload)},indent=2))
    elif s["kind"]=="tritpp":
        import torch
        from .tritpp import TriTPPAdapter, TriTPPContract
        if "seed" in s: np.random.seed(int(s["seed"])); torch.manual_seed(int(s["seed"]))
        ad=TriTPPAdapter(TriTPPContract(**s["contract"]))
        if a.stage=="fit":
            splits={k:_read_route(v) if isinstance(v,str) else v for k,v in s["splits"].items() if k in {"train","val"}}
            metrics=ad.fit(splits["train"],validation_sequences=splits["val"],epochs=s.get("epochs",1),lr=s.get("lr",1e-3),batch_size=s.get("batch",64),max_steps=s.get("max_steps"),max_time_seconds=s.get("max_time_seconds")); ckpt=ad.save(out); (out/"metrics.json").write_text(json.dumps({**metrics,"checkpoint":str(ckpt)},indent=2))
        else:
            splits={"test":_read_route(s["splits"]["test"])}
            checkpoint=a.checkpoint or s.get("checkpoint")
            if not checkpoint: raise ValueError("evaluate requires --checkpoint from one completed refit; never evaluate all HPO trials")
            ad=TriTPPAdapter.load(checkpoint); (out/"score.json").write_text(json.dumps({"test_log_prob":ad.log_prob(splits["test"]).tolist()},indent=2))
    elif s["kind"]=="etas":
        import pandas as pd
        from .etas_author import ETASAuthorAdapter, ETASContract
        ad=ETASAuthorAdapter(ETASContract(**s["contract"]))
        if a.stage == "fit":
            catalog = pd.read_csv(s["catalog_csv"])
            ad.fit(catalog, data_path=out)
            results=sorted(out.glob("parameters_*.json"))
            if len(results) != 1: raise RuntimeError("expected exactly one author ETAS parameter result")
            (out/"metrics.json").write_text(json.dumps({"checkpoint":str(results[0]),"fit_catalog":str(s["catalog_csv"])},indent=2))
        else:
            checkpoint=a.checkpoint or s.get("checkpoint")
            if not checkpoint: raise ValueError("evaluate requires --checkpoint from the completed ETAS fit")
            scores=ad.score_author_test_window(ad.load_fitted_metadata(checkpoint),quadrature_n=s.get("quadrature_n",1_000_000))
            (out/"score.json").write_text(json.dumps(scores,indent=2))
    else: raise ValueError("kind must be wsm, tritpp, or etas")
if __name__=="__main__": main()
