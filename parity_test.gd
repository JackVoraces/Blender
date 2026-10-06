extends SceneTree
## Blender <-> Godot parity test.
##
## Applies the test poses from parity_expected.json (written by rn_blender.parity) through
## the Godot rig and checks that every rigid part lands where Blender put it.
##
##   godot --headless --path godot -s res://tests/parity_test.gd [-- --tree]

const FaceRigScript := preload("res://scripts/face_rig.gd")
const EXPECTED_PATH := "res://tests/parity_expected.json"
const MODEL_PATH := "res://assets/robot_prototype.glb"
const MAX_POSITION_ERROR := 2.0e-5    # metres (0.02 mm)
const MAX_ROTATION_ERROR := 1.0e-4    # radians


func _initialize() -> void:
	_run()


func _run() -> void:
	var expected: Variant = JSON.parse_string(FileAccess.get_file_as_string(EXPECTED_PATH))
	if typeof(expected) != TYPE_DICTIONARY:
		_fail("cannot read %s" % EXPECTED_PATH)
		return
	var packed := load(MODEL_PATH) as PackedScene
	if packed == null:
		_fail("cannot load %s (has the project been imported?)" % MODEL_PATH)
		return
	var model := packed.instantiate() as Node3D
	root.add_child(model)
	if OS.get_cmdline_user_args().has("--tree"):
		_print_tree(model, 0)

	var rig: Node = FaceRigScript.new()
	rig.delays_enabled = false
	root.add_child(rig)
	if not rig.load_rig():
		_fail("rig definition did not load")
		return
	var missing: PackedStringArray = rig.bind(model)
	if not missing.is_empty():
		_fail("rig nodes missing from the model: %s" % ", ".join(missing))
		return
	await process_frame

	var node_names: Array = expected["nodes"]
	var nodes := {}
	for node_name: String in node_names:
		var node := model.find_child(node_name, true, false) as Node3D
		if node == null:
			_fail("node %s is not in the imported model" % node_name)
			return
		nodes[node_name] = node

	var worst_position := 0.0
	var worst_rotation := 0.0
	var worst_position_at := ""
	var worst_rotation_at := ""
	var poses: Array = expected["poses"]
	for pose: Dictionary in poses:
		rig.reset()
		var controls: Dictionary = pose["controls"]
		for control: String in controls:
			rig.set_control(control, float(controls[control]))
		var head: Dictionary = pose["head_deg"]
		rig.set_head(deg_to_rad(float(head.get("yaw", 0.0))), deg_to_rad(float(head.get("pitch", 0.0))),
				deg_to_rad(float(head.get("roll", 0.0))))
		rig.apply_now()
		await process_frame
		await process_frame
		for node_name: String in node_names:
			var want_raw: Array = pose["nodes"][node_name]
			var want := Transform3D(_vec3(want_raw[0]), _vec3(want_raw[1]), _vec3(want_raw[2]), _vec3(want_raw[3]))
			var got: Transform3D = (nodes[node_name] as Node3D).global_transform
			var position_error := got.origin.distance_to(want.origin)
			# Compare axes directly: for small angles the chord between two unit axes is the
			# angle in radians.  (Quaternion.angle_to bottoms out near 0.05 degrees in float32.)
			var rotation_error := maxf(got.basis.x.distance_to(want.basis.x),
					maxf(got.basis.y.distance_to(want.basis.y), got.basis.z.distance_to(want.basis.z)))
			if position_error > worst_position:
				worst_position = position_error
				worst_position_at = "%s / %s" % [pose["name"], node_name]
			if rotation_error > worst_rotation:
				worst_rotation = rotation_error
				worst_rotation_at = "%s / %s" % [pose["name"], node_name]

	print("PARITY poses=%d nodes=%d" % [poses.size(), node_names.size()])
	print("PARITY worst position error: %.6f mm  (%s)" % [worst_position * 1000.0, worst_position_at])
	print("PARITY worst rotation error: %.6f deg (%s)" % [rad_to_deg(worst_rotation), worst_rotation_at])
	var ok := worst_position <= MAX_POSITION_ERROR and worst_rotation <= MAX_ROTATION_ERROR
	print("PARITY RESULT: %s" % ("PASS" if ok else "FAIL"))
	quit(0 if ok else 1)


func _fail(message: String) -> void:
	printerr("PARITY ERROR: %s" % message)
	print("PARITY RESULT: FAIL")
	quit(1)


func _print_tree(node: Node, depth: int) -> void:
	print("TREE %s%s  [%s]" % ["  ".repeat(depth), node.name, node.get_class()])
	for child in node.get_children():
		_print_tree(child, depth + 1)


static func _vec3(values: Array) -> Vector3:
	return Vector3(values[0], values[1], values[2])
