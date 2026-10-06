extends Node3D
## Live avatar runtime (phase-1 prototype): tracker packets -> FaceRig.
##
## Options, after "--" on the command line:
##   --port=8765                 UDP port to listen on
##   --camera=front|face|three_quarter
##   --no-overlay                hide the stats text
##   --poses=FILE --out=DIR      render every pose of a JSON table to PNG, then quit
##                               ({name: {CONTROL: value, HEAD_YAW/PITCH/ROLL: degrees}})
##   --capture=PREFIX --every=SECONDS --count=N
##                               save N screenshots while running live, then quit

const TARGET := Vector3(0.0, 1.64, 0.0)
const CAMERAS := {
	"front": Vector3(0.0, 1.64, 1.15),
	"face": Vector3(0.0, 1.645, 0.70),
	"three_quarter": Vector3(0.66, 1.66, 0.94),
}
const LIGHTS := {
	"KeyLight": Vector3(-0.75, 2.15, 1.35),
	"FillLight": Vector3(1.05, 1.55, 1.0),
	"RimLight": Vector3(0.45, 2.05, -1.0),
}
const HEAD_KEYS := {"HEAD_YAW": 0, "HEAD_PITCH": 1, "HEAD_ROLL": 2}

@export var port := 8765
## Seconds.  Eases the ~30 Hz tracker values up to display rate.
@export_range(0.0, 0.2, 0.005) var follow_time := 0.035
## Seconds without packets before the face starts returning to neutral.
@export var packet_timeout := 0.5
## Seconds.  How gently the face returns to neutral when the tracker goes quiet.
@export var return_time := 0.45

@onready var _rig: FaceRig = $FaceRig
@onready var _udp: UdpReceiver = $UdpReceiver
@onready var _model: Node3D = $Model
@onready var _camera: Camera3D = $Camera3D
@onready var _stats: Label = $Overlay/Stats

var _face_target: Dictionary = {}
var _face_now: Dictionary = {}
var _head_target := Vector3.ZERO
var _head_now := Vector3.ZERO
var _tracking := false
var _note := ""
var _rate_window := 0.0
var _rate_packets := 0
var _packet_rate := 0.0


func _ready() -> void:
	var args := _user_args()
	for light_name: String in LIGHTS:
		(get_node(light_name) as Node3D).look_at_from_position(LIGHTS[light_name], TARGET)
	var camera_name: String = args.get("camera", "front")
	if CAMERAS.has(camera_name):
		_camera.look_at_from_position(CAMERAS[camera_name], Vector3(0.0, CAMERAS[camera_name].y, 0.0))
	_stats.visible = not args.has("no-overlay")

	if not _rig.load_rig():
		return
	var missing := _rig.bind(_model)
	if not missing.is_empty():
		push_warning("Rig nodes missing from the model: %s" % ", ".join(missing))
	for control in _rig.control_names:
		_face_target[control] = 0.0
		_face_now[control] = 0.0

	if args.has("poses"):
		_render_poses(args["poses"], args.get("out", "user://poses"))
		return
	port = int(args.get("port", str(port)))
	_udp.packet_received.connect(_on_packet)
	_udp.listen(port)
	if args.has("capture"):
		_capture_sequence(args["capture"], float(args.get("every", "0.5")), int(args.get("count", "6")))


func _process(delta: float) -> void:
	_udp.poll(delta)
	var silent := _udp.seconds_since_packet > packet_timeout
	var time_constant := return_time if silent else follow_time
	var blend := 1.0 if time_constant <= 0.0 else 1.0 - exp(-delta / time_constant)
	for control: String in _face_now:
		var target: float = 0.0 if silent else _face_target[control]
		_face_now[control] = lerpf(_face_now[control], target, blend)
		_rig.set_control(control, _face_now[control])
	_head_now = _head_now.lerp(Vector3.ZERO if silent else _head_target, blend)
	_rig.set_head(_head_now.x, _head_now.y, _head_now.z)
	_update_stats(delta, silent)


func _on_packet(packet: Dictionary) -> void:
	_rate_packets += 1
	_tracking = bool(packet.get("tracking", true))
	_note = str(packet.get("note", ""))
	var face: Dictionary = packet.get("face", {})
	for key: String in face:
		var control := key.to_upper()
		if _face_target.has(control):
			_face_target[control] = float(face[key])
	var head: Dictionary = packet.get("head", {})
	_head_target = Vector3(float(head.get("yaw", 0.0)), float(head.get("pitch", 0.0)), float(head.get("roll", 0.0)))


func _update_stats(delta: float, silent: bool) -> void:
	_rate_window += delta
	if _rate_window >= 0.5:
		_packet_rate = _rate_packets / _rate_window
		_rate_window = 0.0
		_rate_packets = 0
	if not _stats.visible:
		return
	var state := "no tracker" if silent else ("tracking" if _tracking else "face lost")
	_stats.text = "%d fps   udp :%d   %.0f packets/s   %s" % [
		Engine.get_frames_per_second(), port, _packet_rate, state]
	if not silent and not _note.is_empty():
		_stats.text += "\n" + _note


func _user_args() -> Dictionary:
	var out := {}
	for arg in OS.get_cmdline_user_args():
		var parts := arg.trim_prefix("--").split("=", true, 1)
		out[parts[0]] = parts[1] if parts.size() > 1 else ""
	return out


func _save_frame(path: String) -> void:
	await RenderingServer.frame_post_draw
	var error := get_viewport().get_texture().get_image().save_png(path)
	if error != OK:
		push_error("Could not save %s (%s)" % [path, error_string(error)])


func _render_poses(pose_file: String, out_dir: String) -> void:
	set_process(false)
	_stats.visible = false
	_rig.delays_enabled = false
	var table: Variant = JSON.parse_string(FileAccess.get_file_as_string(pose_file))
	if typeof(table) != TYPE_DICTIONARY:
		push_error("Cannot read pose table %s" % pose_file)
		get_tree().quit(1)
		return
	DirAccess.make_dir_recursive_absolute(out_dir)
	for pose_name: String in table:
		_rig.reset()
		var head := Vector3.ZERO
		var values: Dictionary = table[pose_name]
		for key: String in values:
			if HEAD_KEYS.has(key):
				head[HEAD_KEYS[key]] = deg_to_rad(float(values[key]))
			else:
				_rig.set_control(key, float(values[key]))
		_rig.set_head(head.x, head.y, head.z)
		for i in 3:
			await get_tree().process_frame
		await _save_frame(out_dir.path_join(pose_name + ".png"))
	get_tree().quit()


func _capture_sequence(prefix: String, every: float, count: int) -> void:
	DirAccess.make_dir_recursive_absolute(prefix.get_base_dir())
	for i in count:
		await get_tree().create_timer(every).timeout
		await _save_frame("%s_%02d.png" % [prefix, i])
	get_tree().quit()
