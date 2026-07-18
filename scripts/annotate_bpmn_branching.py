"""
annotate_bpmn_branching.py
===========================
Produces a Signavio-compatible copy of simulation/models/bpic17_process.bpmn
with a bpmn:textAnnotation attached to every exclusive gateway that can be
matched to a decision point mined in simulation/models/dp_branching_probs.json
(scripts/mine_dp_probs.py, Section 1.5). Each annotation shows the branching
probabilities split by first vs. repeat visit, including the data-driven
__END__ (case-terminates-here) outcome where the mined table has one.

Matching a gateway to a mined decision point
---------------------------------------------
pm4py's BPMN -> Petri net conversion (the exact one PetriNetProcessComponent
uses) names the place right after an exclusive gateway's split
"exi_<gateway id>". Seeding a marking with a single token there and running
the same tau-closure (_visible_frontier) used during mining reproduces the
identical decision-point key ("label | label | ...") for genuine standalone
decision points.

Only 4 of the 21 mined decision points turn out to match a single gateway
this way. The rest are joint decisions across BPIC-17's concurrently active
O_/W_/A_ threads (real parallelism in the net) or loop redo-points that
several gateways feed into together -- no single gateway shape represents
them honestly. Rather than silently dropping those (and losing central
1.5 findings like the A_Incomplete/A_Validating loop), they are placed as
floating text annotations (no bpmn:association, so nothing implies false
gateway ownership) in a grid below the diagram, each labelled with its full
decision-point key.

Usage (from repo root):
    python scripts/annotate_bpmn_branching.py
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pm4py.objects.petri_net.obj import Marking  # noqa: E402

from simulation.components.petri_process import PetriNetProcessComponent  # noqa: E402

BPMN_PATH = REPO_ROOT / "simulation" / "models" / "bpic17_process.bpmn"
OUTPUT_PATH = REPO_ROOT / "simulation" / "models" / "bpic17_process_annotated.bpmn"

NS = {
    "bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL",
    "bpmndi": "http://www.omg.org/spec/BPMN/20100524/DI",
    "omgdc": "http://www.omg.org/spec/DD/20100524/DC",
    "omgdi": "http://www.omg.org/spec/DD/20100524/DI",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
    "xsd": "http://www.w3.org/2001/XMLSchema",
}
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


def qn(tag: str) -> str:
    prefix, local = tag.split(":")
    return f"{{{NS[prefix]}}}{local}"


def label_text(label: str) -> str:
    return "case ends here" if label == "__END__" else label


def format_dist(dist: dict) -> list:
    lines = []
    for label, p in sorted(dist.items(), key=lambda kv: -kv[1]):
        pct = round(p * 100, 1)
        if pct <= 0.0:
            continue
        lines.append(f"  {label_text(label)}: {pct:g}%")
    return lines


def average_dists(dists: list) -> dict:
    keys = set()
    for d in dists:
        keys.update(d.keys())
    return {k: sum(d.get(k, 0.0) for d in dists) / len(dists) for k in keys}


def annotation_text(dp_key: str, entry: dict, title: str) -> str:
    lines = [title, f"DP: {dp_key}", ""]

    if "1" in entry:
        lines.append("1st visit:")
        lines.extend(format_dist(entry["1"]))
        header_used = "1"
    else:
        lines.append("Overall (per-visit data too sparse):")
        lines.extend(format_dist(entry.get("all", {})))
        header_used = "all"

    repeat_buckets = [b for b in entry if b not in ("1", "all")]
    if repeat_buckets and header_used == "1":
        lines.append("")
        lines.append("Repeat visits (2+):")
        repeat_dist = average_dists([entry[b] for b in repeat_buckets])
        lines.extend(format_dist(repeat_dist))

    return "\n".join(lines)


def find_place(net, name: str):
    for p in net.places:
        if p.name == name:
            return p
    return None


def main():
    comp = PetriNetProcessComponent(bpmn_path=str(BPMN_PATH), seed=42)
    dp_probs = comp._dp_probs
    if not dp_probs:
        raise SystemExit(f"No dp_probs loaded -- check {BPMN_PATH} / dp_branching_probs.json")
    print(f"[load] {len(dp_probs)} mined decision points available.")

    tree = ET.parse(BPMN_PATH)
    root = tree.getroot()
    process_el = root.find("bpmn:process", NS)
    plane_el = root.find("bpmndi:BPMNDiagram/bpmndi:BPMNPlane", NS)

    shape_bounds = {}
    for shape in plane_el.findall("bpmndi:BPMNShape", NS):
        gid = shape.get("bpmnElement")
        bounds = shape.find("omgdc:Bounds", NS)
        shape_bounds[gid] = {
            "x": float(bounds.get("x")),
            "y": float(bounds.get("y")),
            "width": float(bounds.get("width")),
            "height": float(bounds.get("height")),
        }

    gateways = []
    for gw in process_el.findall("bpmn:exclusiveGateway", NS):
        gid = gw.get("id")
        outgoing = gw.findall("bpmn:outgoing", NS)
        if len(outgoing) >= 2:
            gateways.append(gid)
    print(f"[scan] {len(gateways)} exclusive-gateway splits in the BPMN.")

    matched = skipped = 0
    matched_keys = set()
    for gid in gateways:
        place = find_place(comp.net, f"exi_{gid}")
        if place is None:
            skipped += 1
            continue
        marking = Marking()
        marking[place] = 1
        frontier = comp._visible_frontier(marking)
        if not frontier:
            skipped += 1
            continue
        dp_key = " | ".join(sorted(frontier.keys()))
        entry = dp_probs.get(dp_key)
        if entry is None:
            skipped += 1
            continue
        matched += 1
        matched_keys.add(dp_key)

        text = annotation_text(dp_key, entry, "Branching probabilities (1.5, mined from BPIC17 log)")
        annot_id = f"annot_{gid}"
        assoc_id = f"assoc_{gid}"

        annot_el = ET.SubElement(process_el, qn("bpmn:textAnnotation"), {"id": annot_id})
        text_el = ET.SubElement(annot_el, qn("bpmn:text"))
        text_el.text = text

        ET.SubElement(process_el, qn("bpmn:association"), {
            "id": assoc_id,
            "sourceRef": gid,
            "targetRef": annot_id,
            "associationDirection": "None",
        })

        gw_bounds = shape_bounds.get(gid)
        if gw_bounds is None:
            continue
        n_lines = text.count("\n") + 1
        ann_width, ann_height = 260.0, max(60.0, 16.0 * n_lines + 20.0)
        ann_x = gw_bounds["x"] + gw_bounds["width"] + 40.0
        ann_y = gw_bounds["y"] - ann_height / 2.0 + gw_bounds["height"] / 2.0

        shape_el = ET.SubElement(plane_el, qn("bpmndi:BPMNShape"), {
            "bpmnElement": annot_id, "id": f"{annot_id}_gui",
        })
        ET.SubElement(shape_el, qn("omgdc:Bounds"), {
            "height": str(ann_height), "width": str(ann_width),
            "x": str(ann_x), "y": str(ann_y),
        })

        edge_el = ET.SubElement(plane_el, qn("bpmndi:BPMNEdge"), {
            "bpmnElement": assoc_id, "id": f"{assoc_id}_gui",
        })
        gw_cx = gw_bounds["x"] + gw_bounds["width"]
        gw_cy = gw_bounds["y"] + gw_bounds["height"] / 2.0
        ET.SubElement(edge_el, qn("omgdi:waypoint"), {"x": str(gw_cx), "y": str(gw_cy)})
        ET.SubElement(edge_el, qn("omgdi:waypoint"), {"x": str(ann_x), "y": str(ann_y + ann_height / 2.0)})

    print(f"[match] {matched} gateways annotated, {skipped} skipped (no matching mined decision point).")

    # Decision points that don't correspond to any single gateway (joint
    # decisions across concurrently active branches, or loop redo-points fed
    # by several gateways): still surface them, as floating annotations with
    # no bpmn:association, in a grid below the diagram.
    leftover = [(k, v) for k, v in dp_probs.items() if k not in matched_keys]
    if leftover:
        max_x = max(b["x"] + b["width"] for b in shape_bounds.values())
        min_x = min(b["x"] for b in shape_bounds.values())
        max_y = max(b["y"] + b["height"] for b in shape_bounds.values())

        title_id = "annot_leftover_title"
        title_el = ET.SubElement(process_el, qn("bpmn:textAnnotation"), {"id": title_id})
        title_text_el = ET.SubElement(title_el, qn("bpmn:text"))
        title_text_el.text = (
            "Additional 1.5 branching findings\n"
            "These decision points involve several concurrently active branches "
            "(BPIC17 has parallel O_/W_/A_ threads) or loop redo-points fed by "
            "multiple gateways, so they cannot be pinned to one gateway shape "
            "honestly -- shown here unattached instead."
        )
        title_shape = ET.SubElement(plane_el, qn("bpmndi:BPMNShape"), {
            "bpmnElement": title_id, "id": f"{title_id}_gui",
        })
        grid_y0 = max_y + 120.0
        ET.SubElement(title_shape, qn("omgdc:Bounds"), {
            "height": "80.0", "width": "500.0", "x": str(min_x), "y": str(grid_y0),
        })

        col_width, row_gap, n_cols = 340.0, 40.0, 3
        col_x = [min_x + c * col_width for c in range(n_cols)]
        col_y = [grid_y0 + 120.0] * n_cols  # next free y per column

        for i, (dp_key, entry) in enumerate(leftover):
            text = annotation_text(dp_key, entry, "Joint decision point (1.5, concurrent branches / loop)")
            col = i % n_cols
            annot_id = f"annot_leftover_{i}"
            annot_el = ET.SubElement(process_el, qn("bpmn:textAnnotation"), {"id": annot_id})
            text_el = ET.SubElement(annot_el, qn("bpmn:text"))
            text_el.text = text

            n_lines = text.count("\n") + 1
            ann_width, ann_height = 320.0, max(60.0, 16.0 * n_lines + 20.0)
            ann_x = col_x[col]
            ann_y = col_y[col]

            shape_el = ET.SubElement(plane_el, qn("bpmndi:BPMNShape"), {
                "bpmnElement": annot_id, "id": f"{annot_id}_gui",
            })
            ET.SubElement(shape_el, qn("omgdc:Bounds"), {
                "height": str(ann_height), "width": str(ann_width),
                "x": str(ann_x), "y": str(ann_y),
            })
            col_y[col] += ann_height + row_gap

        print(f"[floating] {len(leftover)} joint/loop decision points added as unattached annotations.")

    tree.write(OUTPUT_PATH, encoding="utf-8", xml_declaration=True)
    print(f"[save] -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
