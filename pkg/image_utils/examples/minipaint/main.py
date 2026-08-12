#!/usr/bin/env python3
"""simple paint-like application to learn raylib and also develop image_utils"""
from __future__ import annotations
import sys
import os
from abc import ABC, abstractmethod
from typing import Callable
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from overrides import overrides
import raylib as rl
import numpy as np
from loggez import make_logger

rl.SetTraceLogLevel(rl.LOG_ERROR)
logger = make_logger("MINIPAINT")

sys.path.append(Path(__file__).parents[2].__str__())
from image_utils import Color, PointIJ
if os.getenv("MINIPAINT_PIL", "1") == "1":
    from image_utils_pil import image_draw_line_pil as image_draw_line
else:
    from image_utils import image_draw_line

# global canvas stuff. Acts as a big ass config.
HEIGHT, WIDTH = 1000, 1000
MENU_OFFSET_WIDTH = 100
INITIAL_THICKNESS = 1
MAX_THICKNESS = 2
STEP_THICKNESS = 0.1

# utility functions

def rl_Image_from_numpy(image_arr: np.ndarray) -> "rl.Image":
    """returns a raylib image from a numpy array"""
    img: "rl.Image" = rl.ffi.new("Image *")
    img.mipmaps = 1
    img.data = rl.ffi.cast("void *", image_arr.ctypes.data)
    img.height, img.width = image_arr.shape[0:2]
    img.format = rl.PIXELFORMAT_UNCOMPRESSED_R8G8B8
    return img[0]

# callbacks

def on_click_clear(pos: PointIJ, cfg: PaintConfig, objs: dict[str, SceneObject]):
    objs["canvas"].clear()
    cfg.state = State.LINE_INIT

def on_click_thick_increase(pos: PointIJ, cfg: PaintConfig, objs: dict[str, SceneObject]):
    if cfg.line_thickness < MAX_THICKNESS:
        cfg.line_thickness += STEP_THICKNESS
    logger.info(f"Current thickness: {cfg.line_thickness:.2f}")

def on_click_thick_decrease(pos: PointIJ, cfg: PaintConfig, objs: dict[str, SceneObject]):
    if cfg.line_thickness >= STEP_THICKNESS:
        cfg.line_thickness -= STEP_THICKNESS
    logger.info(f"Current thickness: {cfg.line_thickness:.2f}")

def on_click_canvas(pos: PointIJ, cfg: PaintConfig, objs: dict[str, SceneObject]):
    if cfg.state == State.LINE_INIT:
        logger.info(f"State 0: P1={pos}")
        cfg.line_p1 = (pos[0], pos[1] - MENU_OFFSET_WIDTH)
        cfg.state = State.LINE_TOP_LEFT_CLICKED
    else:
        logger.info(f"State 1: P1={cfg.line_p1} P2={pos}")
        p2 = (pos[0], pos[1] - MENU_OFFSET_WIDTH)
        objs["canvas"].append(
            partial(image_draw_line, p1=cfg.line_p1, p2=p2, color=Color.BLACK,
                    thickness=cfg.line_thickness, inplace=True)
        )
        cfg.state = State.LINE_INIT

# Object types of the scene. Do not use inheritance. Keep it flat and only inherit SceneObject and other traits (ABCs).

@dataclass(kw_only=True)
class SceneObject(ABC):
    """the base scene object of all objects. Defines the common stuff, like name, on_click callbacks etc."""
    name: str
    # callbacks
    on_click: Callable[[PointIJ, dict, dict[str, SceneObject]], bool] | None = None

    @abstractmethod
    def draw(self):
        """draws the object on the screen"""

    @abstractmethod
    def is_clicked(self, pos: PointIJ) -> bool:
        """checks if the object was clicked on"""

@dataclass
class Button(SceneObject):
    """A button in the scene"""
    top_left: PointIJ
    bottom_right: PointIJ
    background_color: "rl.Color"
    text: str
    text_position: PointIJ
    text_font_size: int
    text_color: "rl.Color"
    outline_color: "rl.Color" = None

    def __post_init__(self):
        self.h, self.w =  self.bottom_right[0] - self.top_left[0], self.bottom_right[1] - self.top_left[1]
        assert 0 <= self.text_position[0] < self.h, f"{self.text_position=}, {self.top_left=}, {self.bottom_right=}"
        assert 0 <= self.text_position[1] < self.w, f"{self.text_position=}, {self.top_left=}, {self.bottom_right=}"
        self._abs_text_position = (self.top_left[0] + self.text_position[0], self.top_left[1] + self.text_position[1])
        logger.debug(f"Created button: {repr(self)}")

    @overrides
    def draw(self):
        """draws the button on the main loop"""
        # logger.debug(f"Drawing {self.top_left[::-1], *self.bottom_right[::-1]}")
        rl.DrawRectangleRec((*self.top_left[::-1], self.w, self.h), self.background_color)
        if self.outline_color is not None:
            rl.DrawRectangleLines(*self.top_left[::-1], self.w, self.h, self.outline_color)
        rl.DrawText(self.text.encode("ascii"), *self._abs_text_position[::-1], self.text_font_size, self.text_color)

    @overrides
    def is_clicked(self, pos: PointIJ) -> bool:
        """returns true if it's clicked"""
        res = rl.CheckCollisionPointRec(pos[::-1], (*self.top_left[::-1], self.w, self.h))
        if res:
            logger.debug(f"{repr(self)} clicked")
        return res

    def __repr__(self) -> str:
        return f"[Button] tl={self.top_left} br={self.bottom_right} text='{self.text}' text_pos={self.text_position}"

@dataclass
class Canvas(SceneObject):
    """The canvas object. Contains an image+texture, a list of 'strokes' (which will be objects later) as well as
    undo/redo capabilities"""
    size: PointIJ
    position: PointIJ

    def __post_init__(self):
        self.img: np.ndarray = (np.ones(shape=(*self.size[::-1], 3)) * rl.WHITE[0:3]).astype(np.uint8)
        self._bottom_right = self.position[0] + self.size[0], self.position[1] + self.size[1]
        canvas_rl = rl_Image_from_numpy(self.img)
        self._canvas_tex: "rl.Texture" = rl.LoadTextureFromImage(canvas_rl)

        self.strokes: list[Callable] = []
        self.popped: list[Callable] = []

    @overrides
    def draw(self):
        rl.DrawTexture(self._canvas_tex, *self.position[::-1], rl.WHITE)

    @overrides
    def is_clicked(self, pos: PointIJ) -> bool:
        return rl.CheckCollisionPointRec(pos[::-1], (*self.position[::-1], *self._bottom_right[::-1]))

    def update(self):
        """reloads the new texture after updating the canvas"""
        self.img: np.ndarray = (np.ones(shape=(*self.size[::-1], 3)) * rl.WHITE[0:3]).astype(np.uint8)
        logger.info(f"Updating canvas. Doing {len(self.strokes)} strokes.")
        for stroke in self.strokes:
            stroke(image=self.img)
        rl.UpdateTexture(self._canvas_tex, rl.ffi.cast("void *", self.img.ctypes.data))

    def clear(self):
        """clears the canvas: all the strokes, all the popped and updates the screen"""
        self.strokes = []
        self.popped = []
        self.update()

    def append(self, item: Callable):
        """adds one stroke to the list of strokes and updates the screen"""
        self.strokes.append(item)
        self.popped = []
        self.update()

    def undo(self):
        """removes one stroke (if any) from the list, adds it to popped and updates the screen"""
        if len(self.strokes) > 0:
            self.popped.append(self.strokes.pop(-1))
            self.update()

    def redo(self):
        """removes one stroke (if any) from the list of popped, adds it back to strokes and updates the screen"""
        if len(self.popped) > 0:
            self.strokes.append(self.popped.pop(-1))
            self.update()

    def __del__(self):
        rl.UnloadTexture(self._canvas_tex)

class State:
    """the current state, i.e. if he clicked on line, it's LINE_INIT etc. Basically a flat state"""
    LINE_INIT = 0
    LINE_TOP_LEFT_CLICKED = 1

@dataclass
class PaintConfig:
    state: State = State.LINE_INIT # initially starts in LINE_INIT state
    line_p1: PointIJ | None = None # first point of the line
    line_thickness: float = INITIAL_THICKNESS

def main():
    """main fn - defines the main loop and calls the paint object when needed"""
    rl.SetConfigFlags(rl.FLAG_WINDOW_RESIZABLE)
    rl.InitWindow(WIDTH + MENU_OFFSET_WIDTH, HEIGHT, "minipaint".encode("ascii"))

    # The config which contains a state and various properties related to each state in a flat class.
    cfg = PaintConfig()
    scene_objects: dict[str, SceneObject] = {obj.name: obj for obj in [
        Canvas(name="canvas", size=(WIDTH, HEIGHT), position=(0, MENU_OFFSET_WIDTH), on_click=on_click_canvas),
        Button(name="btn_reset", top_left=(0, 0), bottom_right=(50, 100), background_color=rl.BLUE, text="Reset",
               text_position=(20, 20), text_font_size=20, text_color=rl.BLACK,
               outline_color=rl.BLACK, on_click=on_click_clear),
        Button(name="btn_thick++", top_left=(50, 0), bottom_right=(100, 100), background_color=rl.BLUE,
               text="Thick++", text_position=(20, 10), text_font_size=20, text_color=rl.BLACK,
               outline_color=rl.BLACK, on_click=on_click_thick_increase),
        Button(name="btn_thick--", top_left=(100, 0), bottom_right=(150, 100), background_color=rl.BLUE,
               text="Thick--", text_position=(20, 10), text_font_size=20, text_color=rl.BLACK,
               outline_color=rl.BLACK, on_click=on_click_thick_decrease),
    ]}
    canvas: Canvas = scene_objects["canvas"]
    logger.info(f"Created {len(scene_objects)} scene objects")

    while not rl.WindowShouldClose():

        if rl.IsMouseButtonPressed(rl.MOUSE_BUTTON_LEFT):
            pos = (rl.GetMouseY(), rl.GetMouseX())
            for scene_object in scene_objects.values():
                if scene_object.is_clicked(pos):
                    if scene_object.on_click is not None and scene_object.on_click(pos, cfg, scene_objects) != False:
                        break
            else:
                logger.info(f"Clicked on P={pos} which is not on any button or canvas.")
        if rl.IsKeyDown(rl.KEY_LEFT_CONTROL) and rl.IsKeyPressed(rl.KEY_Z):
            canvas.undo()
        if rl.IsKeyDown(rl.KEY_LEFT_CONTROL) and rl.IsKeyPressed(rl.KEY_Y):
            canvas.redo()

        rl.BeginDrawing()

        rl.ClearBackground(rl.BLACK)
        for scene_object in scene_objects.values():
            scene_object.draw()

        rl.EndDrawing()

    rl.CloseWindow()

if __name__ == "__main__":
    main()
