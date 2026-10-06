"""Camera names.

OpenCV only knows cameras by number.  This asks DirectShow for the device
names in the same order OpenCV's DirectShow backend numbers them, so a camera
can be chosen by name ("Elgato Facecam") and the choice survives a device
being plugged in or a virtual camera being installed.

Listing names opens no camera.
"""
import ctypes
import sys
from ctypes import POINTER, byref, c_long, c_ubyte, c_ulong, c_ushort, c_void_p, c_wchar_p

_CLSID_SYSTEM_DEVICE_ENUM = "{62BE5D10-60EB-11D0-BD3B-00A0C911CE86}"
_IID_CREATE_DEV_ENUM = "{29840822-5B84-11D0-BD3B-00A0C911CE86}"
_CLSID_VIDEO_INPUT_CATEGORY = "{860BB310-5D01-11D0-BD3B-00A0C911CE86}"
_IID_PROPERTY_BAG = "{55272A00-42CB-11CE-8135-00AA004BB851}"
_CLSCTX_INPROC_SERVER = 1


class _GUID(ctypes.Structure):
    _fields_ = [("d1", c_ulong), ("d2", c_ushort), ("d3", c_ushort), ("d4", c_ubyte * 8)]


class _VARIANT(ctypes.Structure):
    _fields_ = [("vt", c_ushort), ("r1", c_ushort), ("r2", c_ushort), ("r3", c_ushort),
                ("bstr", c_wchar_p), ("pad", c_void_p)]


def _guid(text):
    guid = _GUID()
    ctypes.oledll.ole32.CLSIDFromString(c_wchar_p(text), byref(guid))
    return guid


def _method(pointer, index, *argtypes):
    """Entry ``index`` of a COM object's vtable, as a callable returning the HRESULT."""
    vtable = ctypes.cast(pointer, POINTER(POINTER(c_void_p))).contents
    return ctypes.WINFUNCTYPE(c_long, c_void_p, *argtypes)(vtable[index])


def _release(pointer):
    if pointer:
        _method(pointer, 2)(pointer)


def list_names():
    """Video capture device names, in DirectShow (= OpenCV CAP_DSHOW) index order."""
    if sys.platform != "win32":
        return []
    ole32 = ctypes.windll.ole32
    ole32.CoInitialize(None)
    names = []
    dev_enum, monikers = c_void_p(), c_void_p()
    try:
        hr = ole32.CoCreateInstance(byref(_guid(_CLSID_SYSTEM_DEVICE_ENUM)), None, _CLSCTX_INPROC_SERVER,
                                    byref(_guid(_IID_CREATE_DEV_ENUM)), byref(dev_enum))
        if hr != 0 or not dev_enum:
            return names
        # ICreateDevEnum::CreateClassEnumerator
        hr = _method(dev_enum, 3, POINTER(_GUID), POINTER(c_void_p), c_ulong)(
            dev_enum, byref(_guid(_CLSID_VIDEO_INPUT_CATEGORY)), byref(monikers), 0)
        if hr != 0 or not monikers:
            return names            # S_FALSE: no devices in the category
        bag_iid = _guid(_IID_PROPERTY_BAG)
        while True:
            moniker, fetched = c_void_p(), c_ulong()
            # IEnumMoniker::Next
            hr = _method(monikers, 3, c_ulong, POINTER(c_void_p), POINTER(c_ulong))(
                monikers, 1, byref(moniker), byref(fetched))
            if hr != 0 or not moniker:
                break
            name = None
            bag = c_void_p()
            # IMoniker::BindToStorage
            hr = _method(moniker, 9, c_void_p, c_void_p, POINTER(_GUID), POINTER(c_void_p))(
                moniker, None, None, byref(bag_iid), byref(bag))
            if hr == 0 and bag:
                value = _VARIANT()
                # IPropertyBag::Read
                hr = _method(bag, 3, c_wchar_p, POINTER(_VARIANT), c_void_p)(
                    bag, "FriendlyName", byref(value), None)
                if hr == 0:
                    name = value.bstr
                ctypes.windll.oleaut32.VariantClear(byref(value))
                _release(bag)
            _release(moniker)
            names.append(name or f"camera {len(names)}")
    finally:
        _release(monikers)
        _release(dev_enum)
    return names


def resolve(choice, names=None):
    """Camera index for a number, a numeric string or (part of) a device name.

    Returns (index, name or None).  Raises LookupError when a name matches nothing.
    """
    names = list_names() if names is None else names
    text = str(choice).strip()
    if text.lstrip("-").isdigit():
        index = int(text)
        return index, names[index] if 0 <= index < len(names) else None
    wanted = text.lower()
    exact = [i for i, name in enumerate(names) if name.lower() == wanted]
    partial = [i for i, name in enumerate(names) if wanted in name.lower()]
    matches = exact or partial
    if not matches:
        raise LookupError(f"no camera named like {choice!r}; found: {names or 'none'}")
    return matches[0], names[matches[0]]
