#!/usr/bin/env python3
"""
vre_video_player - a simple video player using tkinter
Usage: ./vre_video_player.py VIDEO [--hud_font_color 255 0 0] [--hud_font_size R G B] [--stats_path STATS.CSV]
                                    [--input_size H W]
If VIDEO is stdin ("-") or a socket (tcp://), '--input_size' must be set as it's passed to VREVideo.
"""
import tkinter as tk
from argparse import ArgumentParser, Namespace
from queue import Queue, Empty
from datetime import datetime
import time
import sys
import threading
from PIL import Image, ImageTk
import numpy as np

from vre_video.utils import logger, image_add_text
from vre_video import VREVideo

TIMEOUT_READ_S = 2
TIMEOUT_WRITE_S = 0.5

class VideoContainer:
    """Video Container class. Holds the state of the video player shared by the UI and the reading thread"""
    def __init__(self, video: VREVideo, font_size: float | None = None, font_color: tuple[int, int, int] | None = None):
        self.video = video
        self.fps_multiplier: float = 1.0
        self._current_frame: int = 0
        self.is_paused: bool = False
        self.font_color = tuple(font_color or (255, 0, 0))
        self.font_size = font_size or max(10, self.frame_shape[1] // 50)

    @property
    def fps(self) -> float:
        """The fps of the video"""
        return self.video.fps

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """The frame shape of the video"""
        return self.video.frame_shape

    @property
    def current_frame(self):
        """the current frame of the video"""
        return self._current_frame

    @current_frame.setter
    def current_frame(self, current_frame: int):
        if not 0 <= current_frame < len(self.video):
            logger.info(f"Frame {current_frame} wanted which is wrong ({len(self.video)=}). Resetting to 0.")
            current_frame = 0
        self._current_frame = int(current_frame)

class VideoPlayer(threading.Thread):
    """with the help of the chat-ge-pe-tees"""
    def __init__(self, queue: Queue, video_container: VideoContainer):
        super().__init__()
        self.queue = queue
        self.video_container = video_container
        self.show_stats = True
        self.frame_counter = 0
        self.fps_counter = np.full(100, dtype=np.float32, fill_value=float("nan"))
        self.root: tk.Tk | None = None
        self.canvas: tk.Canvas | None = None

    def on_key_press(self, e: tk.Event):
        """event callback for key presses"""
        if e.char == "-":
            self.video_container.fps_multiplier = max(self.video_container.fps_multiplier / 2, 0.25)
            logger.info(f"Speed: {self.video_container.fps_multiplier}x")
        if e.char == "=":
            self.video_container.fps_multiplier = min(self.video_container.fps_multiplier * 2, 4.0)
            logger.info(f"Speed: {self.video_container.fps_multiplier}x")
        if e.keycode == 9: # esc
            self.root.destroy()
        if e.keycode == 65: # space
            self.video_container.is_paused = not self.video_container.is_paused
        if e.keycode == 113: # left arrow -> jump half a second back
            logger.debug((self.video_container.current_frame,
                          round(self.video_container.current_frame - self.video_container.fps // 3)))
            self.video_container.current_frame = self.video_container.current_frame - self.video_container.fps // 3
        if e.keycode == 114: # right arrow -> jump half a second forward
            logger.debug((self.video_container.current_frame,
                          round(self.video_container.current_frame + self.video_container.fps // 3)))
            self.video_container.current_frame = self.video_container.current_frame + self.video_container.fps // 3
        if e.char == "i":
            self.show_stats = not self.show_stats

    def run(self):
        """main loop of the window thread"""
        self.root = tk.Tk()
        self.root.title(self.video_container.video.path)
        self.canvas = tk.Canvas(self.root, width=self.video_container.frame_shape[1],
                                height=self.video_container.frame_shape[0])
        self.canvas.pack()
        self.canvas.bind("<KeyPress>", self.on_key_press)
        self.canvas.focus_set()

        while True:
            self.root.update()
            now = datetime.now()
            try:
                img_arr = self.queue.get(block=True, timeout=TIMEOUT_READ_S * 10**(self.frame_counter == 0)) # * 10s
            except Empty:
                logger.warning("Empty queue. Exitting")
                self.root.destroy()
                break
            if self.show_stats:
                avg_fps = np.nanmean(self.fps_counter) if self.frame_counter > 0 else 0
                text = ""
                if self.video_container.current_frame != 0:
                    text = f"Frame: {self.video_container.current_frame}. "
                text += f"Avg fps: {avg_fps:.2f}"
                img_arr = image_add_text(img_arr, text=text, position=(0, 18),
                                         font_size_px=self.video_container.font_size,
                                         font_color=self.video_container.font_color)

            photo = ImageTk.PhotoImage(image=Image.fromarray(img_arr))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.frame_counter += 1
            if self.show_stats and not self.video_container.is_paused:
                took = (datetime.now() - now).total_seconds()
                self.fps_counter[self.frame_counter % len(self.fps_counter)] = 1 / took

def read_loop(video_container: VideoContainer, q: Queue, stats: list[str], stats_path: str | None):
    """simply reads the current frame as given by video.current_frame and writes to the queue."""
    video = video_container.video
    while True:
        try:
            q.put(video[video_container.current_frame], timeout=TIMEOUT_WRITE_S)
        except Exception as e:
            logger.info(f"Program closed: {e}")
            sys.exit(0)
        if stats_path is not None:
            stats.append(datetime.now().replace(tzinfo=None).isoformat())
            if len(stats) % 100 == 0 and len(stats) > 0:
                with open(stats_path, "w" if len(stats) == 100 else "a") as fp:
                    fp.write("frame,timestmap\n"+"\n".join(f"{i},{row}" for i, row in enumerate(stats)))
        if video_container.is_paused:
            next_frame = video_container.current_frame
        else:
            next_frame = (video_container.current_frame + 1) % len(video)
        video_container.current_frame = next_frame
        time.sleep(1 / (video.fps * video_container.fps_multiplier))

def get_args() -> Namespace:
    """cli args"""
    parser = ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--input_size", nargs=2, type=int, help="Must be set if video_path is a socket or stdin")
    parser.add_argument("--fps", type=float, help="Must be set if video_path is a socket or stdin. If webcam, use 1.")
    parser.add_argument("--stats_path", help="If set, it will output a per-frame stats file with timestamps for fps")
    parser.add_argument("--hud_font_size", type=int, help="Size in pixels. If not set, will use a default")
    parser.add_argument("--hud_font_color", help="R G B tuple. If not set, defaults to white.", nargs=3, type=int)
    args = parser.parse_args()
    if args.video_path == "-":
        assert args.input_size is not None, "--input_size must be set when reading frames from socket/stdin"
        assert args.fps is not None, "--fps must be set when reading frames from socket/stdin"
    return args

def main(args: Namespace):
    """main fn"""
    q = Queue(maxsize=1)
    reader_kwargs = {}
    if args.video_path == "-":
        reader_kwargs = {"resolution": args.input_size, "fps": args.fps, "async_worker": True}
    video_container = VideoContainer(VREVideo(args.video_path, **reader_kwargs), font_size=args.hud_font_size,
                                     font_color=args.hud_font_color)
    VideoPlayer(q, video_container).start()
    stats = None if args.stats_path is None else []
    read_loop(video_container, q, stats, args.stats_path)

if __name__ == "__main__":
    main(get_args())
