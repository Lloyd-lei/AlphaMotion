"""Conservative skeleton endpoint candidates for the offline pilot trainer.

``indices`` is a candidate pool. For a confirmed hand pair it contains both
anchors; for generic leaves the caller samples up to two without replacement.
No endpoint is a mesh/contact/tool-frame assertion. Sample validity and FK QC
remain the data loader's responsibility because this function sees no motion.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

_POLICY_PATH = Path(__file__).with_name("unimate_endpoint_policy.json")
_ARM = frozenset({"arm", "upperarm", "forearm", "elbow", "shoulder"})
_FINGERS = frozenset({"finger", "thumb", "index", "middle", "ring", "pinky", "little"})
_NOT_HAND = _FINGERS | frozenset({
    "foot", "feet", "toe", "hoof", "paw", "wing", "tail", "antenna", "leg", "calf", "shin",
})
_SIDE = {"left": "left", "l": "left", "right": "right", "r": "right"}
_PATH_EPS = 1e-8
# Verified locally from AlphaMotion1/dof_summary.json: select cache_xwbc_lafan,
# then apply extract_xwbc_pilot.py:45 exclusions. The two other entries in that
# summary (soma77 and g1_30) belong to cache_paired_fixed and are not allowed.
_XWBC_BODY_NAMES = frozenset({
    "human_smpl", "booster_k1", "booster_t1", "booster_t1_29dof",
    "fourier_gr3", "fourier_n1", "hightorque_hi", "kuavo_s45",
    "pnd_adam_lite", "tienkung", "unitree_g1_23dof", "unitree_g1_29dof",
    "unitree_h1", "unitree_h1_2", "unitree_h2", "unitree_r1",
})
_XWBC_ALLOWLIST_EVIDENCE = {
    "source": "source-snapshot/AlphaMotion1/dof_summary.json",
    "source_sha256": "770e15554770d90d16301de878e2cadbd3cea851a3316aef7e29c9e46a907bed",
    "source_filter": "cache == cache_xwbc_lafan",
    "extractor": "extract_xwbc_pilot.py:45",
    "excluded": ["pal_talos", "engineai_pm01"],
    "count": 16,
}


def _tokens(name: str) -> tuple[str, ...]:
    """Exact lexical tokens; never match 'hand' inside 'handle' or 'handrail'."""
    value = name.rsplit(":", 1)[-1]
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    return tuple(re.findall(r"[a-z]+", value.lower()))


def _policy() -> tuple[dict[str, Any], str | None, str]:
    try:
        raw = _POLICY_PATH.read_bytes()
    except FileNotFoundError:
        return {}, None, "missing_policy_hand_resolution_disabled"
    value = json.loads(raw)
    if value.get("schema_version") != "unimate_endpoint_policy_v1":
        raise ValueError("unsupported endpoint policy schema")
    return value, hashlib.sha256(raw).hexdigest(), "loaded"


def resolve_endpoints(body_name, domain, names, parents, offsets) -> dict[str, Any]:
    """Resolve evidenced hand/wrist anchors, otherwise real non-root leaves.

    Hand inference is enabled for the JSON-allowed UniMate/UniML3D bodies and
    the 16 verified XWBC bodies. XWBC may also use a uniquely distal named wrist
    when no hand/finger is represented, with an explicit wrist-only label.
    Every other body/domain gets generic leaves. Offsets use their native
    canonical units. A zero-length final bone is retained when its cumulative
    root path has nonzero length. Malformed topology yields an empty result
    with an explicit isolation reason, never a guessed fallback index.
    """
    policy, policy_sha, policy_status = _policy()
    body_name, domain = str(body_name), str(domain).strip().lower()
    body_policy = next((b for b in policy.get("bodies", []) if b["object_type"] == body_name), None)
    xwbc_enabled = domain == "xwbc" and body_name in _XWBC_BODY_NAMES
    hand_enabled = xwbc_enabled or (domain in {"unimate", "uniml3d"} and bool(
        body_policy and body_policy.get("allow_hand_name_resolution", False)
    ))
    result = {
        "indices": [], "kind": "unavailable_endpoint_condition", "names": [],
        "body_name": body_name, "domain": domain,
        "valid": False, "isolate_endpoint_condition": True,
        "policy_path": str(_POLICY_PATH), "policy_sha256": policy_sha,
        "policy_status": policy_status, "hand_resolution_enabled": hand_enabled,
        "xwbc_verified_body_enabled": xwbc_enabled,
        "xwbc_allowlist_evidence": _XWBC_ALLOWLIST_EVIDENCE if domain == "xwbc" else None,
        "wrist_chain_selection_audit": [],
        "strategy": "none", "selection_mode": "sample_up_to_2_without_replacement",
        "hand_candidates": [], "hand_candidates_rejected": [],
        "ambiguous_hand_sides": [], "generic_leaf_candidates": [],
        "rejected_leaves": [], "endpoint_labels": [],
        "semantics": "named skeleton joint position and own-joint rotation; no mesh contact, palm surface, hardware tool frame, or dexterity inferred",
        "sample_validation_required": ["per-sequence FK QC", "position_valid for sampled frames", "rotation_valid if SE3 is conditioned"],
    }

    def invalid(reason):
        result.update(resolution_status="invalid_or_unavailable", isolation_reason=reason)
        return result

    try:
        raw_parents = np.asarray(parents)
        offset_array = np.asarray(offsets, dtype=np.float64)
        raw_names = list(names)
    except (TypeError, ValueError) as exc:
        return invalid("malformed_metadata: " + str(exc))
    if raw_parents.ndim != 1 or raw_parents.dtype.kind not in "iu":
        return invalid("parents_must_be_1d_integer_indices")
    parent_array = raw_parents.astype(np.int64, copy=False)
    count = len(parent_array)
    if count < 1 or len(raw_names) != count or offset_array.shape != (count, 3):
        return invalid("joint_names_parents_offsets_shape_mismatch")
    if not all(isinstance(name, (str, np.str_)) and str(name).strip() for name in raw_names):
        return invalid("joint_names_must_be_nonempty_strings")
    names = [str(name) for name in raw_names]
    if not np.isfinite(offset_array).all():
        return invalid("nonfinite_offsets")
    if parent_array[0] != -1 or any(not 0 <= p < j for j, p in enumerate(parent_array[1:], 1)):
        return invalid("require_one_parent_first_tree_rooted_at_zero")
    token_sets = [set(_tokens(name)) for name in names]
    ancestors: list[list[int]] = [[] for _ in names]
    descendants: list[list[int]] = [[] for _ in names]
    path_lengths = np.zeros(count, dtype=np.float64)
    for j, parent in enumerate(parent_array[1:], 1):
        ancestors[j] = [int(parent)] + ancestors[parent]
        path_lengths[j] = path_lengths[parent] + np.linalg.norm(offset_array[j])
        for ancestor in ancestors[j]:
            descendants[ancestor].append(j)
    nonleaves = set(parent_array[1:].tolist())
    leaves = [j for j in range(1, count) if j not in nonleaves]
    valid_leaves = []
    for j in leaves:
        if path_lengths[j] > _PATH_EPS:
            valid_leaves.append(j)
        else:
            result["rejected_leaves"].append({"index": j, "name": names[j], "reason": "zero_cumulative_root_path"})
    result["generic_leaf_candidates"] = valid_leaves
    result["cumulative_rest_path_lengths"] = path_lengths.tolist()
    result["joint_name_tokens"] = [list(_tokens(name)) for name in names]
    result["topology_sha256"] = hashlib.sha256(
        parent_array.astype("<i8").tobytes() + offset_array.astype("<f8").tobytes() + json.dumps(names).encode()
    ).hexdigest()
    result["joint_count"] = count
    result["expected_joint_count_from_policy"] = body_policy.get("joint_count_expected_from_prior_remote_audit") if body_policy else None
    # The expected count is a prior inventory annotation, not a replacement for
    # the caller's verified conditioning hash and topology/array shape checks.
    result["expected_joint_count_matches"] = result["expected_joint_count_from_policy"] in {None, count}

    candidates = []
    if hand_enabled:
        for j in range(1, count):
            tokens = token_sets[j]
            if not tokens.intersection({"hand", "wrist"}):
                continue
            reasons = []
            if tokens & _NOT_HAND:
                reasons.append("non_hand_or_finger_anchor_tokens")
            if path_lengths[j] <= _PATH_EPS:
                reasons.append("zero_cumulative_root_path")
            arm_ancestors = [i for i in ancestors[j] if token_sets[i] & _ARM]
            if not arm_ancestors:
                reasons.append("missing_arm_chain_ancestor")
            child_evidence = [i for i in descendants[j] if token_sets[i] & ({"hand"} | _FINGERS)]
            if "hand" not in tokens and not child_evidence and not xwbc_enabled:
                reasons.append("wrist_without_hand_or_finger_descendant")
            # Do not let contradictory side markers quietly select an anchor.
            sides = {_SIDE[t] for i in [j] + ancestors[j] for t in token_sets[i] if t in _SIDE}
            if len(sides) > 1:
                reasons.append("contradictory_side_markers")
            side = next(iter(sides)) if len(sides) == 1 else "unknown"
            if xwbc_enabled and side == "unknown":
                reasons.append("xwbc_anchor_requires_unambiguous_side_marker")
            candidate = {
                "index": j, "name": names[j], "side": side,
                "anchor_type": "hand" if "hand" in tokens else "wrist",
                "arm_ancestor_indices": arm_ancestors,
                "hand_or_finger_descendant_indices": child_evidence,
                "evidence": "exact name tokens plus hierarchy; skeleton anchor only",
                "priority": 0 if "hand" in tokens else (1 if child_evidence else 2),
                "wrist_only_fallback": "hand" not in tokens and not child_evidence,
            }
            if reasons:
                result["hand_candidates_rejected"].append({**candidate, "reasons": reasons})
            else:
                candidates.append(candidate)
        result["hand_candidates"] = candidates

    best = {}
    for side in ("left", "right", "unknown"):
        options = [candidate for candidate in candidates if candidate["side"] == side]
        if not options:
            continue
        hand_options = [candidate for candidate in options if candidate["anchor_type"] == "hand"]
        if xwbc_enabled and not hand_options:
            # Every same-side wrist must lie on one ancestry chain. Distal depth
            # alone is insufficient: equal-side names on two branches are not
            # a license to guess which wrist is the kinematic endpoint.
            chain = all(
                left["index"] in ancestors[right["index"]] or right["index"] in ancestors[left["index"]]
                for k, left in enumerate(options) for right in options[k + 1:]
            )
            if not chain:
                result["ambiguous_hand_sides"].append(side)
                result["hand_candidates_rejected"].extend(
                    {**candidate, "reasons": ["same_side_wrist_candidates_branch_or_are_incomparable"]}
                    for candidate in options
                )
                continue
            # Prefer an existing hand/finger binding when present. Otherwise,
            # the verified old XWBC wrist-only chain may expose its distal joint.
            supported = [candidate for candidate in options if candidate["hand_or_finger_descendant_indices"]]
            pool = supported if supported else options
            distal = [candidate for candidate in pool if all(
                other["index"] == candidate["index"] or other["index"] in ancestors[candidate["index"]]
                for other in pool
            )]
            if len(distal) == 1:
                best[side] = distal[0]
                result["wrist_chain_selection_audit"].append({
                    "side": side, "candidate_indices": [candidate["index"] for candidate in options],
                    "selected_index": distal[0]["index"],
                    "selection": "unique_distal_wrist_with_hand_or_finger_descendant" if supported else "unique_distal_wrist_only_chain",
                    "all_wrist_candidates_form_ancestor_chain": True,
                    "label": "named_wrist_kinematic_anchor",
                })
                continue
            result["ambiguous_hand_sides"].append(side)
            continue
        priority = min(candidate["priority"] for candidate in options)
        options = [candidate for candidate in options if candidate["priority"] == priority]
        if len(options) == 1:
            best[side] = options[0]
        else:
            result["ambiguous_hand_sides"].append(side)
            result["hand_candidates_rejected"].extend(
                {**candidate, "reasons": ["multiple_equally_valid_anchors_on_side"]} for candidate in options
            )
    chosen = [best[side] for side in ("left", "right") if side in best]
    if not chosen and len(candidates) == 1 and "unknown" in best:
        chosen = [best["unknown"]]
    if chosen:
        indices = [candidate["index"] for candidate in chosen]
        types = {candidate["anchor_type"] for candidate in chosen}
        anchor_noun = next(iter(types)) if len(types) == 1 else "hand_or_wrist"
        strategy = ("double_" + anchor_noun + "_anchors") if len(chosen) == 2 else ("single_" + anchor_noun + "_anchor")
        result.update(
            indices=indices, names=[names[i] for i in indices],
            kind="named_hand_or_wrist_anchor", valid=True, isolate_endpoint_condition=False,
            strategy=strategy,
            resolution_status="resolved_from_names_and_topology",
            endpoint_labels=[{"index": candidate["index"], "name": candidate["name"], "label": "named_" + candidate["anchor_type"] + "_kinematic_anchor", "side": candidate["side"]} for candidate in chosen],
        )
    elif valid_leaves:
        result.update(
            indices=valid_leaves, names=[names[i] for i in valid_leaves],
            kind="generic_topological_leaf", valid=True, isolate_endpoint_condition=False,
            strategy="generic_leaf_pool", resolution_status="resolved_topology_only",
            endpoint_labels=[{"index": i, "name": names[i], "label": "generic_leaf", "side": "not_inferred"} for i in valid_leaves],
        )
    else:
        return invalid("no_valid_nonroot_leaf_or_evidenced_hand_anchor")
    return result
