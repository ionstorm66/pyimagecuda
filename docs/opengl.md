# OpenGL Integration

PyImageCUDA provides GPU-to-GPU transfer with OpenGL through CUDA-OpenGL interop, enabling efficient real-time preview pipelines.

---

## Overview

**Traditional approach:**
```
GPU Processing → CPU Download → CPU Upload → OpenGL Display
```

**With OpenGL interop:**
```
GPU Processing → GPU Copy → OpenGL Display
```

**Performance:**

- Zero CPU involvement (all transfers GPU-side)
- ~10x faster than CPU download/upload
- Sub-millisecond updates for 2K images (~1ms measured)
- Single GPU-GPU DMA copy (PBO→Texture, asynchronous)

---

## How It Works

The pipeline has 5 stages, all happening on the GPU:

1. **Generate frame** in a PyImageCUDA `Image` (float32 RGBA).
2. **Convert to uint8** in an `ImageU8` (the format OpenGL expects).
3. **Copy to PBO** via `GLResource.copy_from()` (GPU→GPU, no CPU involved).
4. **Upload PBO to texture** with `glTexSubImage2D` (asynchronous on the GPU).
5. **Draw the texture** on a fullscreen quad.

The PBO (Pixel Buffer Object) is the bridge: it's an OpenGL buffer that CUDA can write to directly. PyImageCUDA registers it once via `GLResource(pbo_id)` and then writes to it every frame with `copy_from()`.

---

## Minimal Example

A complete, runnable script showing the full GPU→display pipeline. This is all you need to display PyImageCUDA output in a real-time window:

```python
import glfw
from OpenGL.GL import *
from pyimagecuda import Image, ImageU8, Fill, convert_float_to_u8, GLResource

W, H = 1280, 720

# 1. Create OpenGL window
glfw.init()
window = glfw.create_window(W, H, "PyImageCUDA Preview", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)  # vsync

# 2. Create PBO (Pixel Buffer Object). Size = W * H * 4 (RGBA uint8)
pbo = int(glGenBuffers(1))
glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo)
glBufferData(GL_PIXEL_UNPACK_BUFFER, W * H * 4, None, GL_STREAM_DRAW)
glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

# 3. Create OpenGL texture for display
tex = int(glGenTextures(1))
glBindTexture(GL_TEXTURE_2D, tex)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0,
             GL_RGBA, GL_UNSIGNED_BYTE, None)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

# 4. Register PBO with CUDA (one-time setup)
gl_res = GLResource(pbo)

# 5. PyImageCUDA buffers (reused every frame)
canvas = Image(W, H)
canvas_u8 = ImageU8(W, H)

# Render loop
while not glfw.window_should_close(window):
    glfw.poll_events()

    # --- Generate the frame on GPU ---
    Fill.gradient(canvas, (1, 0, 0, 1), (0, 0, 1, 1), 'radial')

    # --- Convert float32 → uint8 (still on GPU) ---
    convert_float_to_u8(canvas_u8, canvas)

    # --- Copy GPU buffer to PBO (zero-copy, GPU→GPU) ---
    gl_res.copy_from(canvas_u8)

    # --- Upload PBO to texture ---
    glBindTexture(GL_TEXTURE_2D, tex)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H,
                    GL_RGBA, GL_UNSIGNED_BYTE, None)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    # --- Draw the texture as a fullscreen quad ---
    fb_w, fb_h = glfw.get_framebuffer_size(window)
    glViewport(0, 0, fb_w, fb_h)
    glClear(GL_COLOR_BUFFER_BIT)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex)
    glBegin(GL_TRIANGLE_STRIP)
    # Y inverted to match image orientation
    glTexCoord2f(0, 1); glVertex2f(-1, -1)
    glTexCoord2f(1, 1); glVertex2f( 1, -1)
    glTexCoord2f(0, 0); glVertex2f(-1,  1)
    glTexCoord2f(1, 0); glVertex2f( 1,  1)
    glEnd()
    glDisable(GL_TEXTURE_2D)

    glfw.swap_buffers(window)

# Cleanup
gl_res.free()
canvas.free()
canvas_u8.free()
glDeleteTextures([tex])
glDeleteBuffers(1, [pbo])
glfw.terminate()
```

**Dependencies:**

```bash
pip install pyimagecuda glfw PyOpenGL PyOpenGL_accelerate
```

**Note:** `glGenBuffers(1)` and `glGenTextures(1)` return numpy types from PyOpenGL. Cast them with `int()` before passing to `GLResource`, otherwise you'll get a `ValueError`.

---

## GLResource API

### Constructor
```python
GLResource(pbo_id: int)
```
Registers an OpenGL PBO with CUDA for interop.

**Parameters:**
- `pbo_id`: OpenGL buffer ID from `glGenBuffers()`. Must be a Python int (cast with `int()` if it comes from PyOpenGL).

**Raises:**
- `ValueError`: If `pbo_id` is invalid
- `RuntimeError`: If CUDA registration fails

---

### copy_from()
```python
gl_resource.copy_from(image: ImageU8, sync: bool = True) -> None
```
Copies `ImageU8` data directly to the PBO (GPU→GPU, zero CPU overhead).

**Parameters:**
- `image`: Source `ImageU8` buffer. Must match PBO size.
- `sync`: If `True` (default), blocks until copy completes. If `False`, returns immediately (advanced usage).

**Raises:**
- `TypeError`: If image is not `ImageU8`
- `RuntimeError`: If resource has been freed

---

### free()
```python
gl_resource.free() -> None
```
Unregisters the resource. Must be called before deleting the PBO.

**Context Manager:**
```python
with GLResource(pbo) as gl_resource:
    gl_resource.copy_from(image)
```

---

## Buffer Sizing

When creating the PBO, its size must match what you'll write to it:

- **For `ImageU8` (RGBA uint8):** `width * height * 4` bytes
- The OpenGL texture must use `GL_RGBA8` internal format and `GL_RGBA, GL_UNSIGNED_BYTE` for `glTexImage2D` / `glTexSubImage2D`.

`GLResource` only works with `ImageU8`, not `Image` (float32). If your pipeline runs in float32, use `convert_float_to_u8()` before calling `copy_from()`.

---

## Full Application Example

For a complete production-quality implementation with PySide6 (Qt widget integration, mouse/keyboard handling, multiple panels):

**https://github.com/offerrall/pyimagecuda-studio/tree/main/pyimagecuda_studio/gui/preview**

---

## Best Practices

### Display Integration
- Invert texture Y-coordinates if the image appears upside-down (`glTexCoord2f(0, 1)` at bottom-left, not `(0, 0)`).
- Enable `GL_BLEND` for alpha channel support if compositing transparent layers.

### Resource Management
- Create PBO with `GL_STREAM_DRAW` for best performance (signals frequent updates).
- Pre-allocate buffers (`Image`, `ImageU8`) outside the render loop and reuse every frame.
- Call `gl_res.free()` before `glDeleteBuffers([pbo])`.

### Performance
- Reuse `Image` and `ImageU8` buffers across frames; never allocate inside the loop.
- Match texture size to PBO size exactly to avoid resampling.
- `glTexSubImage2D` with PBO is asynchronous: the GPU schedules the upload without stalling the CPU.
- Use `glfw.swap_interval(0)` to disable vsync and measure raw pipeline throughput.

### Common Pitfalls
- **`ValueError: Invalid PBO ID`**: PyOpenGL returns numpy types. Cast with `int(glGenBuffers(1))`.
- **Black or garbled texture**: PBO size doesn't match `width * height * 4`, or you're using `Image` (float32) instead of `ImageU8`.
- **Image upside-down**: invert Y in texture coordinates as shown in the minimal example.
- **`copy_from` fails after a while**: you freed the `GLResource` or the PBO before the last frame. Free in the right order: `gl_res.free()` first, then OpenGL objects.