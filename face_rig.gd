class_name FaceRig
extends Node
## Turns robot expression controls into rigid plate motion.
##
## The GLB only carries geometry and rest transforms; each plate's rest transform is its
## mechanical frame (pivot + slide/hinge axes).  The control relationships live in
## face_rig.expanded.json, written by the Blender export from config/face_rig.json, and are
## evaluated here exactly as the Blender drivers evaluate them:
##
##   loc = clamp(sum(factor_i * loc_i)),  rot = clamp(sum(factor_i * rot_i))   (Euler XYZ)
##
## in the plate's own frame, Blender axes.  On top of that each link may carry a delay, so a
## driven plate follows its control late, like a linkage taking up slack.

const DEFAULT_RIG_PATH := "res://assets/face_rig.expanded.json"
const RIG_FORMAT := "rn.face_rig.expanded/1"
const _WHEN := {"both": 0, "pos": 1, "neg": 2}
const _RECORD_INTERVAL := 1.0 / 120.0

## Honour each link's delay (secondary mechanical motion).  Off: everything moves at once.
@export var delays_enabled := true

var control_names := PackedStringArray()

var _data: Dictionary = {}
var _values: Dictionary = {}       # control -> float, clamped to its range
var _ranges: Dictionary = {}       # control -> Vector2(min, max)
var _history: Dictionary = {}      # control -> History
var _plates: Array[Dictionary] = []
var _skeleton: Skeleton3D
var _head_bones: Array[Dictionary] = []
var _head := Vector3.ZERO          # yaw, pitch, roll in radians, clamped
var _head_min := Vector3.ZERO
var _head_max := Vector3.ZERO
var _head_history: Array[History] = []
var _clock := 0.0
var _last_record := -1.0


## Ring buffer of (time, value) samples: lets a link read its control as it was a moment ago.
class History:
	const SIZE := 96
	var _t := PackedFloat64Array()
	var _v := PackedFloat32Array()
	var _newest := -1
	var _count := 0

	func _init() -> void:
		_t.resize(SIZE)
		_v.resize(SIZE)

	func push(time: float, value: float) -> void:
		_newest = (_newest + 1) % SIZE
		_t[_newest] = time
		_v[_newest] = value
		_count = mini(_count + 1, SIZE)

	func sample(time: float) -> float:
		if _count == 0:
			return 0.0
		var newer := _newest
		if time >= _t[newer]:
			return _v[newer]
		for n in range(1, _count):
			var older := (_newest - n + SIZE) % SIZE
			if _t[older] <= time:
				var span := _t[newer] - _t[older]
				if span <= 0.0:
					return _v[newer]
				return lerpf(_v[older], _v[newer], (time - _t[older]) / span)
			newer = older
		return _v[newer]


func load_rig(path: String = DEFAULT_RIG_PATH) -> bool:
	var text := FileAccess.get_file_as_string(path)
	if text.is_empty():
		push_error("FaceRig: cannot read %s" % path)
		return false
	var parsed: Variant = JSON.parse_string(text)
	if typeof(parsed) != TYPE_DICTIONARY or parsed.get("format") != RIG_FORMAT:
		push_error("FaceRig: %s is not a %s file" % [path, RIG_FORMAT])
		return false
	_data = parsed
	_values.clear()
	_ranges.clear()
	_history.clear()
	control_names = PackedStringArray()
	var controls: Dictionary = _data["controls"]
	for control: String in controls:
		var spec: Dictionary = controls[control]
		control_names.append(control)
		_ranges[control] = Vector2(spec["min"], spec["max"])
		_values[control] = 0.0
		_history[control] = History.new()
	_head_history = [History.new(), History.new(), History.new()]
	return true


## Find the plates and head bones under [param model_root].  Returns the names that are missing.
func bind(model_root: Node) -> PackedStringArray:
	var missing := PackedStringArray()
	_plates.clear()
	var nodes: Dictionary = _data.get("nodes", {})
	for node_name: String in nodes:
		var node := model_root.find_child(node_name, true, false) as Node3D
		if node == null:
			missing.append(node_name)
			continue
		var entry: Dictionary = nodes[node_name]
		var links: Array[Dictionary] = []
		for link: Dictionary in entry["links"]:
			links.append({
				"control": link["control"],
				"when": _WHEN[link["when"]],
				"loc": _vec3(link["loc"]),
				"rot": _vec3(link["rot"]),
				"delay": float(link["delay"]),
			})
		var limits: Dictionary = entry["limits"]
		_plates.append({
			"node": node,
			"rest": node.transform,
			"loc_min": _bound(limits["loc"], 0), "loc_max": _bound(limits["loc"], 1),
			"rot_min": _bound(limits["rot"], 0), "rot_max": _bound(limits["rot"], 1),
			"links": links,
		})

	_head_bones.clear()
	_skeleton = null
	var skeletons := model_root.find_children("*", "Skeleton3D", true, false)
	if not skeletons.is_empty():
		_skeleton = skeletons[0]
	var head: Variant = _data.get("head")
	if head != null and _skeleton != null:
		var bones: Dictionary = head["bones"]
		for bone_name: String in bones:
			var index := _skeleton.find_bone(bone_name)
			if index < 0:
				missing.append("bone:" + bone_name)
				continue
			_head_bones.append({
				"index": index,
				"rest": _skeleton.get_bone_rest(index).basis.get_rotation_quaternion(),
				"share": float(bones[bone_name]["share"]),
				"lag": float(bones[bone_name]["lag"]),
			})
		var lim: Dictionary = head["limits"]
		_head_min = Vector3(lim["yaw"][0], lim["pitch"][0], lim["roll"][0])
		_head_max = Vector3(lim["yaw"][1], lim["pitch"][1], lim["roll"][1])
	return missing


func set_control(control: String, value: float) -> void:
	if not _ranges.has(control):
		return
	var limits: Vector2 = _ranges[control]
	_values[control] = clampf(value, limits.x, limits.y)


func get_control(control: String) -> float:
	return _values.get(control, 0.0)


## Head rotation in radians.  yaw > 0 turns to the character's left, pitch > 0 looks up,
## roll > 0 tips the head towards the character's left shoulder.
func set_head(yaw: float, pitch: float, roll: float) -> void:
	_head = Vector3(yaw, pitch, roll).clamp(_head_min, _head_max)


func reset() -> void:
	for control: String in _values:
		_values[control] = 0.0
	_head = Vector3.ZERO


## Pose immediately, ignoring link delays.
func apply_now() -> void:
	_apply(false)


func _process(delta: float) -> void:
	_clock += delta
	if _clock - _last_record >= _RECORD_INTERVAL:
		_last_record = _clock
		for control: String in _values:
			(_history[control] as History).push(_clock, _values[control])
		for axis in 3:
			_head_history[axis].push(_clock, _head[axis])
	_apply(delays_enabled)


func _apply(use_delays: bool) -> void:
	for plate in _plates:
		var loc := Vector3.ZERO
		var rot := Vector3.ZERO
		for link: Dictionary in plate["links"]:
			var value: float = _values[link["control"]]
			var delay: float = link["delay"]
			if use_delays and delay > 0.0:
				value = (_history[link["control"]] as History).sample(_clock - delay)
			var factor := value
			match link["when"]:
				1: factor = maxf(value, 0.0)
				2: factor = maxf(-value, 0.0)
			loc += (link["loc"] as Vector3) * factor
			rot += (link["rot"] as Vector3) * factor
		loc = loc.clamp(plate["loc_min"], plate["loc_max"])
		rot = rot.clamp(plate["rot_min"], plate["rot_max"])
		(plate["node"] as Node3D).transform = (plate["rest"] as Transform3D) * blender_delta(loc, rot)

	for bone in _head_bones:
		var angles := _head
		var lag: float = bone["lag"]
		if use_delays and lag > 0.0:
			angles = Vector3(
				_head_history[0].sample(_clock - lag),
				_head_history[1].sample(_clock - lag),
				_head_history[2].sample(_clock - lag))
		angles *= bone["share"]
		# The exported bones point up with identity rest rotation, so bone space is
		# X = character's left, Y = up, Z = forward.  Yaw is the outermost rotation.
		var delta := Quaternion(Vector3.UP, angles.x) \
				* Quaternion(Vector3.RIGHT, -angles.y) \
				* Quaternion(Vector3.BACK, -angles.z)
		_skeleton.set_bone_pose_rotation(bone["index"], (bone["rest"] as Quaternion) * delta)


## A plate-frame motion given in Blender axes (location in metres, Euler XYZ in radians)
## as a Godot transform.  Blender (x, y, z) is Godot (x, z, -y).
static func blender_delta(loc: Vector3, rot: Vector3) -> Transform3D:
	var basis := Basis(Vector3.UP, rot.z) * Basis(Vector3.BACK, -rot.y) * Basis(Vector3.RIGHT, rot.x)
	return Transform3D(basis, Vector3(loc.x, loc.z, -loc.y))


static func _vec3(values: Array) -> Vector3:
	return Vector3(values[0], values[1], values[2])


static func _bound(pairs: Array, which: int) -> Vector3:
	return Vector3(pairs[0][which], pairs[1][which], pairs[2][which])
