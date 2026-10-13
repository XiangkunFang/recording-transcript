"""System audio capture on macOS via ScreenCaptureKit — no virtual audio driver needed.

Requires Screen Recording permission for the app running Python (Terminal, iTerm, VS Code...).
"""
import threading
from typing import Callable

import CoreMedia as CM
import libdispatch
import numpy as np
import objc
import ScreenCaptureKit as SCK
from Foundation import NSObject

from vad import SAMPLE_RATE

PERMISSION_HELP = (
    "Screen Recording permission is needed to capture system audio.\n"
    "    System Settings → Privacy & Security → Screen & System Audio Recording →\n"
    "    enable your terminal app, then restart it."
)

SCStreamOutput = objc.protocolNamed("SCStreamOutput")
SCStreamDelegate = objc.protocolNamed("SCStreamDelegate")


class _Output(NSObject, protocols=[SCStreamOutput, SCStreamDelegate]):
    def initWithPush_(self, push):
        self = objc.super(_Output, self).init()
        self.push = push
        return self

    def stream_didOutputSampleBuffer_ofType_(self, stream, sbuf, kind):
        if kind != SCK.SCStreamOutputTypeAudio or not CM.CMSampleBufferIsValid(sbuf):
            return
        block = CM.CMSampleBufferGetDataBuffer(sbuf)
        if block is None:
            return
        length = CM.CMBlockBufferGetDataLength(block)
        status, data = CM.CMBlockBufferCopyDataBytes(block, 0, length, None)
        if status == 0 and data:
            self.push(np.frombuffer(bytes(data), dtype=np.float32).copy())

    def stream_didStopWithError_(self, stream, error):
        print(f"[!] System audio capture stopped: {error}")


def _call_async(fn, *args, timeout=10):
    """Run an ObjC method with a completion handler and wait for its result."""
    done, result = threading.Event(), {}

    def handler(*res):
        result["res"] = res
        done.set()

    fn(*args, handler)
    if not done.wait(timeout):
        raise RuntimeError("ScreenCaptureKit did not respond")
    return result["res"]


class SystemAudioSource:
    name = "System audio (ScreenCaptureKit)"

    def __init__(self):
        content, err = _call_async(SCK.SCShareableContent.getShareableContentWithCompletionHandler_)
        if err is not None or content is None or not content.displays():
            detail = err.localizedDescription() if err is not None else "no displays"
            raise PermissionError(f"{PERMISSION_HELP}\n    ({detail})")
        display = content.displays()[0]
        content_filter = SCK.SCContentFilter.alloc().initWithDisplay_excludingWindows_(display, [])

        cfg = SCK.SCStreamConfiguration.alloc().init()
        cfg.setCapturesAudio_(True)
        cfg.setExcludesCurrentProcessAudio_(True)
        cfg.setSampleRate_(SAMPLE_RATE)
        cfg.setChannelCount_(1)
        # We only want audio; keep the unavoidable video stream tiny and slow.
        cfg.setWidth_(2)
        cfg.setHeight_(2)
        cfg.setMinimumFrameInterval_(CM.CMTimeMake(1, 1))

        self._push: Callable[[np.ndarray], None] = lambda _: None
        self._output = _Output.alloc().initWithPush_(lambda chunk: self._push(chunk))
        self._stream = SCK.SCStream.alloc().initWithFilter_configuration_delegate_(
            content_filter, cfg, self._output
        )
        self._queue = libdispatch.dispatch_queue_create(b"system-audio", None)
        for kind in (SCK.SCStreamOutputTypeAudio, SCK.SCStreamOutputTypeScreen):
            ok, err = self._stream.addStreamOutput_type_sampleHandlerQueue_error_(
                self._output, kind, self._queue, None
            )
            if not ok:
                raise RuntimeError(f"Could not add stream output: {err}")

    def start(self, push: Callable[[np.ndarray], None]):
        self._push = push
        (err,) = _call_async(self._stream.startCaptureWithCompletionHandler_)
        if err is not None:
            raise PermissionError(f"{PERMISSION_HELP}\n    ({err.localizedDescription()})")

    def stop(self):
        _call_async(self._stream.stopCaptureWithCompletionHandler_)
