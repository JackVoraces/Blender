class_name UdpReceiver
extends Node
## Receives tracker packets (JSON over UDP, localhost) and keeps only the newest one.
## A dropped or late packet does not matter: the next one replaces it.

signal packet_received(packet: Dictionary)

var latest: Dictionary = {}
var packets_total := 0
var bad_packets := 0
var seconds_since_packet := INF

var _peer := PacketPeerUDP.new()
var _listening := false


func listen(port: int, address: String = "127.0.0.1") -> Error:
	if _listening:
		_peer.close()
	var error := _peer.bind(port, address)
	_listening = error == OK
	if not _listening:
		push_error("UdpReceiver: cannot listen on %s:%d (%s)" % [address, port, error_string(error)])
	return error


## Drain the socket.  Called by the owner at the top of its frame, so a packet that has
## already arrived is applied in this frame rather than the next.
func poll(delta: float) -> void:
	seconds_since_packet += delta
	if not _listening:
		return
	var newest := PackedByteArray()
	while _peer.get_available_packet_count() > 0:
		newest = _peer.get_packet()
		packets_total += 1
	if newest.is_empty():
		return
	var parsed: Variant = JSON.parse_string(newest.get_string_from_utf8())
	if typeof(parsed) != TYPE_DICTIONARY:
		bad_packets += 1
		return
	latest = parsed
	seconds_since_packet = 0.0
	packet_received.emit(latest)


func _exit_tree() -> void:
	_peer.close()
