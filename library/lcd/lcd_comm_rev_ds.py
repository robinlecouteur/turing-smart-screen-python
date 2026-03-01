# SPDX-License-Identifier: GPL-3.0-or-later
#
# DS Turing Display driver for turing-smart-screen-python
# Nintendo DS Lite via DSpico flashcart USB display firmware
#
# Set in config.yaml:
#   display:
#     REVISION: DS
#     BRIGHTNESS: 100          # 0-100
#     RESET_ON_STARTUP: false  # DS reset is fast; set true if you want a fresh init
#
# Display specs:
#   Resolution : 256 x 384  (top 256x192  +  bottom 256x192)
#   Pixel fmt  : DS ABGR1555 little-endian (bit15=alpha, 14-10=B, 9-5=G, 4-0=R)
#   Interface  : USB CDC serial  VID=0x1209  PID=0xDD01
#   HELLO resp : "ds_256x384.lite\n"

import os
import sys
import time
from typing import Optional

import numpy as np
import serial
from PIL import Image
from serial.tools.list_ports import comports

from library.lcd.lcd_comm import LcdComm, Orientation
from library.log import logger


# ---------------------------------------------------------------------------
# Pixel-format helper (module level so it can be imported by other scripts)
# ---------------------------------------------------------------------------

def image_to_abgr1555(image: Image.Image) -> bytes:
    """
    Convert a PIL Image to DS-native ABGR1555 little-endian bytes.

    DS pixel layout (16-bit, little-endian):
      bit 15    : alpha  – must be 1 for the pixel to be opaque
      bits 14-10: blue   (5 bits)
      bits  9-5 : green  (5 bits)
      bits  4-0 : red    (5 bits)
    """
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")

    rgb = np.asarray(image).reshape((image.size[1] * image.size[0], -1))
    r = rgb[:, 0].astype(np.uint16) >> 3
    g = rgb[:, 1].astype(np.uint16) >> 3
    b = rgb[:, 2].astype(np.uint16) >> 3
    return (np.uint16(0x8000) | (b << 10) | (g << 5) | r).astype("<u2").tobytes()


# ---------------------------------------------------------------------------
# LcdComm implementation
# ---------------------------------------------------------------------------

class LcdCommRevDS(LcdComm):
    """
    Turing Smart Screen-compatible driver for Nintendo DS Lite via DSpico.

    Implements the Turing Rev A protocol (6-byte packed command headers) and
    sends pixels in DS-native ABGR1555 format instead of RGB565.

    Orientation is always fixed (portrait 256x384); SetOrientation is a no-op.
    Brightness uses the DS scale: 0 = off, 100 = maximum.
    """

    USB_VID = 0x1209
    USB_PID = 0xDD01

    DISPLAY_WIDTH  = 256
    DISPLAY_HEIGHT = 384

    CMD_RESET           = 101
    CMD_CLEAR           = 102
    CMD_SCREEN_OFF      = 108
    CMD_SCREEN_ON       = 109
    CMD_SET_BRIGHTNESS  = 110  # x0 = level 0-100
    CMD_SET_ORIENTATION = 121  # ignored by DS firmware
    CMD_DISPLAY_BITMAP  = 197
    CMD_HELLO           = 69

    def __init__(
        self,
        com_port: str = "AUTO",
        display_width: int = DISPLAY_WIDTH,
        display_height: int = DISPLAY_HEIGHT,
        update_queue=None,
    ):
        logger.debug("HW revision: DS (Nintendo DS Lite via DSpico)")
        LcdComm.__init__(self, com_port, display_width, display_height, update_queue)
        # Optional callback invoked after a successful reconnect, so callers
        # can redraw static content (background, labels) without the driver
        # needing to know about display.py.
        self.on_reconnect = None
        self.openSerial()

    def __del__(self):
        self.closeSerial()

    # ------------------------------------------------------------------
    # Serial open — override base class which uses rtscts=True.
    # The DS presents as a CDC ACM device; RTS/CTS hardware flow control
    # is not supported and prevents the port from opening correctly.
    # ------------------------------------------------------------------

    def openSerial(self):
        if self.com_port == 'AUTO':
            self.com_port = self.auto_detect_com_port()
            if not self.com_port:
                logger.error(
                    "Cannot find DS Turing Display COM port automatically. "
                    "Is the DS running the usb-display firmware and connected via USB?")
                try:
                    sys.exit(0)
                except Exception:
                    os._exit(0)
            else:
                logger.debug(f"Auto detected COM port: {self.com_port}")
        else:
            logger.debug(f"Static COM port: {self.com_port}")

        try:
            # rtscts=False, dsrdtr=False: the DS CDC device does not support
            # hardware flow control; enabling it blocks the port on most OSes.
            # write_timeout=30: prevents a stalled write from hanging the process.
            self.lcd_serial = serial.Serial(
                self.com_port,
                baudrate=115200,
                timeout=5,
                write_timeout=30,
                rtscts=False,
                dsrdtr=False,
            )
        except Exception as e:
            logger.error(f"Cannot open COM port {self.com_port}: {e}")
            try:
                sys.exit(0)
            except Exception:
                os._exit(0)

    def _reconnect(self):
        """Wait for the DS to re-enumerate and reopen the COM port.

        Called after a serial error. Keeps retrying every 2 seconds until the
        port can be opened. Also re-sends initialization so the DS screen is
        refreshed from the next update cycle.
        """
        logger.warning("DS display disconnected, waiting for reconnect...")
        while True:
            time.sleep(2)
            port = self.com_port
            if port == 'AUTO':
                port = self.auto_detect_com_port()
            if not port:
                logger.debug("DS display not found yet, retrying...")
                continue
            try:
                self.lcd_serial = serial.Serial(
                    port,
                    baudrate=115200,
                    timeout=5,
                    write_timeout=30,
                    rtscts=False,
                    dsrdtr=False,
                )
                self.com_port = port
                logger.info(f"DS display reconnected on {port}")
                self.InitializeComm()
                if self.on_reconnect:
                    self.on_reconnect()
                return
            except Exception as e:
                logger.debug(f"Reconnect attempt failed ({e}), retrying...")

    # ------------------------------------------------------------------
    # Auto-detect
    # ------------------------------------------------------------------

    @staticmethod
    def auto_detect_com_port() -> Optional[str]:
        ports = comports()

        # Prefer exact VID/PID match
        for p in ports:
            if p.vid == LcdCommRevDS.USB_VID and p.pid == LcdCommRevDS.USB_PID:
                logger.debug(f"Found DS Turing Display on {p.device} (VID/PID match)")
                return p.device

        # Fall back to first available port (OS may not expose VID/PID in listing)
        if ports:
            logger.debug(f"No VID/PID match; falling back to first available port {ports[0].device}")
            return ports[0].device

        return None

    # ------------------------------------------------------------------
    # Serial write — override base class to use reconnect loop
    # ------------------------------------------------------------------

    def WriteLine(self, line: bytes):
        while True:
            try:
                self.lcd_serial.write(line)
                return
            except serial.SerialTimeoutException:
                logger.warning("DS display write timeout, slowing down")
                time.sleep(0.1)
            except serial.SerialException:
                self.closeSerial()
                self._reconnect()

    # ------------------------------------------------------------------
    # Protocol helpers
    # ------------------------------------------------------------------

    def _make_header(self, cmd: int, x0: int, y0: int, x1: int, y1: int) -> bytearray:
        """Build a 6-byte Turing Rev A command header."""
        buf = bytearray(6)
        buf[0] = (x0 >> 2) & 0xFF
        buf[1] = (((x0 & 3) << 6) | (y0 >> 4)) & 0xFF
        buf[2] = (((y0 & 15) << 4) | (x1 >> 6)) & 0xFF
        buf[3] = (((x1 & 63) << 2) | (y1 >> 8)) & 0xFF
        buf[4] = y1 & 0xFF
        buf[5] = cmd & 0xFF
        return buf

    def SendCommand(self, cmd: int, x0: int = 0, y0: int = 0, x1: int = 0, y1: int = 0,
                    bypass_queue: bool = False):
        """Encode and dispatch a 6-byte command header."""
        buf = self._make_header(cmd, x0, y0, x1, y1)
        if not self.update_queue or bypass_queue:
            self.WriteData(buf)
        else:
            with self.update_queue_mutex:
                self.update_queue.put((self.WriteData, [buf]))

    # ------------------------------------------------------------------
    # LcdComm abstract method implementations
    # ------------------------------------------------------------------

    def InitializeComm(self):
        """Send HELLO and log the device identification string."""
        self.SendCommand(self.CMD_HELLO, bypass_queue=True)
        self.lcd_serial.flush()
        time.sleep(0.05)
        response = b""
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            waiting = self.lcd_serial.in_waiting
            if waiting:
                response += self.serial_read(waiting)
                if b"\n" in response:
                    break
            else:
                time.sleep(0.01)
        self.serial_flush_input()
        if response:
            logger.info(
                "DS Turing Display: "
                + response.decode("ascii", errors="ignore").strip()
            )
        else:
            logger.warning("DS Turing Display: no response to HELLO (is firmware running?)")

    def Reset(self):
        """Send RESET. The DS clears both screens; no USB re-enumeration occurs."""
        logger.info("DS display reset")
        self.SendCommand(self.CMD_RESET, bypass_queue=True)
        time.sleep(0.2)

    def Clear(self):
        """Clear both DS screens to black."""
        self.SendCommand(self.CMD_CLEAR)

    def ScreenOff(self):
        """Turn off the DS backlights."""
        self.SendCommand(self.CMD_SCREEN_OFF)

    def ScreenOn(self):
        """Turn on the DS backlights."""
        self.SendCommand(self.CMD_SCREEN_ON)

    def SetBrightness(self, level: int = 100):
        """
        Set display brightness.

        Args:
            level: 0 (backlight off) to 100 (maximum brightness).
        """
        level = max(0, min(100, int(level)))
        self.SendCommand(self.CMD_SET_BRIGHTNESS, x0=level)

    def SetOrientation(self, orientation: Orientation = Orientation.PORTRAIT):
        """DS has a fixed physical orientation; this is a no-op."""
        self.orientation = orientation

    def DisplayPILImage(
        self,
        image: Image.Image,
        x: int = 0,
        y: int = 0,
        image_width: int = 0,
        image_height: int = 0,
    ):
        """
        Send a PIL image to the DS display.

        Clips the image to the 256×384 display area, converts to ABGR1555, and
        streams the 6-byte header followed by raw pixel data.

        Args:
            image:        Source PIL Image (any mode).
            x:            Left edge on the DS display.
            y:            Top edge (0-191 = top screen, 192-383 = bottom screen).
            image_width:  Override image width (0 = use image.width).
            image_height: Override image height (0 = use image.height).
        """
        disp_w = self.DISPLAY_WIDTH
        disp_h = self.DISPLAY_HEIGHT

        if not image_height:
            image_height = image.size[1]
        if not image_width:
            image_width = image.size[0]

        # Clamp to display bounds
        if x + image_width > disp_w:
            image_width = disp_w - x
        if y + image_height > disp_h:
            image_height = disp_h - y

        if image_width <= 0 or image_height <= 0:
            return

        assert x < disp_w,  f"Image X {x} must be < display width {disp_w}"
        assert y < disp_h, f"Image Y {y} must be < display height {disp_h}"

        if image_width != image.size[0] or image_height != image.size[1]:
            image = image.crop((0, 0, image_width, image_height))

        x0, y0 = x, y
        x1, y1 = x + image_width - 1, y + image_height - 1

        pixel_data = image_to_abgr1555(image)
        header = self._make_header(self.CMD_DISPLAY_BITMAP, x0, y0, x1, y1)

        # Send header + all pixel chunks atomically so no other command can
        # interleave between the header and the pixel data.
        # Chunk size: 8 scan lines × 256 px × 2 bytes = 4 096 bytes per write.
        chunk_size = disp_w * 8 * 2

        with self.update_queue_mutex:
            if self.update_queue:
                self.update_queue.put((self.WriteData, [header]))
                for i in range(0, len(pixel_data), chunk_size):
                    self.update_queue.put((self.WriteLine, [pixel_data[i: i + chunk_size]]))
            else:
                self.WriteData(header)
                for i in range(0, len(pixel_data), chunk_size):
                    self.WriteLine(pixel_data[i: i + chunk_size])
