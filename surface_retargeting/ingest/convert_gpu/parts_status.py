"""Per extracted HiPHI part: clips, converted by CPU (smplx/hiphi qa.json), by GPU (smplx_gpu/hiphi qa.json), missing.
Writes /home/arenalabs/umr_factory_data/smplx_gpu/hiphi/parts_status.json and parts_done.txt (parts whose every
clip has a qa.json from either converter -> their extracted BVH is no longer needed by the converters)."""
import json
import time
from pathlib import Path

DATA = Path("/home/arenalabs/umr_factory_data")
OUT = DATA / "smplx_gpu/hiphi"


def main():
    st = {}
    for part in sorted((DATA / "raw/hiphi/extracted").glob("part_*")):
        if not (part / ".extracted").exists():
            st[part.name] = {"state": "extracting"}
            continue
        n = cpu = gpu = 0
        miss = []
        for m in part.glob("HiPHI/data/*/*/*/metadata.json"):
            meta = json.loads(m.read_text())
            rel = Path("data") / meta["frame"] / meta["lu"] / meta["motion_id"] / "qa.json"
            n += 1
            c, g = (DATA / "smplx/hiphi" / rel).exists(), (OUT / rel).exists()
            cpu += c; gpu += g
            if not (c or g):
                miss.append(str(m.parent))
        st[part.name] = {"state": "done" if n and not miss else "pending", "clips": n, "cpu_done": cpu, "gpu_done": gpu,
                         "missing": len(miss)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "parts_status.json").write_text(json.dumps({"t": time.strftime("%F %T"), "parts": st}, indent=1))
    (OUT / "parts_done.txt").write_text("".join(f"{p.split('_')[1]}\n" for p, v in st.items() if v.get("state") == "done"))
    print(json.dumps(st))


if __name__ == "__main__":
    main()
