"""Minimal GLB reader: just enough to inspect what an export contains."""
import json
import struct

_MAGIC = 0x46546C67     # "glTF"
_JSON_CHUNK = 0x4E4F534A


def read_json(path) -> dict:
    """Return the JSON chunk of a .glb file."""
    with open(path, "rb") as fh:
        magic, version, _length = struct.unpack("<III", fh.read(12))
        if magic != _MAGIC or version != 2:
            raise ValueError(f"{path}: not a glTF 2.0 binary")
        chunk_length, chunk_type = struct.unpack("<II", fh.read(8))
        if chunk_type != _JSON_CHUNK:
            raise ValueError(f"{path}: first chunk is not JSON")
        return json.loads(fh.read(chunk_length).decode("utf-8"))


def summary(path) -> dict:
    """Node hierarchy, skins and materials of a .glb, by name."""
    gltf = read_json(path)
    nodes = gltf.get("nodes", [])
    parent = {}
    for i, node in enumerate(nodes):
        for child in node.get("children", []):
            parent[child] = i
    joints = {j for skin in gltf.get("skins", []) for j in skin.get("joints", [])}
    out_nodes = {}
    for i, node in enumerate(nodes):
        out_nodes[node.get("name", f"node{i}")] = {
            "parent": nodes[parent[i]].get("name") if i in parent else None,
            "mesh": "mesh" in node,
            "skinned": "skin" in node,
            "joint": i in joints,
            "translation": node.get("translation", [0, 0, 0]),
            "rotation": node.get("rotation", [0, 0, 0, 1]),
            "scale": node.get("scale", [1, 1, 1]),
        }
    return {
        "generator": gltf.get("asset", {}).get("generator"),
        "nodes": out_nodes,
        "skins": [{"name": s.get("name"), "joints": [nodes[j].get("name") for j in s.get("joints", [])]}
                  for s in gltf.get("skins", [])],
        "materials": [m.get("name") for m in gltf.get("materials", [])],
        "meshes": len(gltf.get("meshes", [])),
        "animations": len(gltf.get("animations", [])),
        "extensions_used": gltf.get("extensionsUsed", []),
    }
