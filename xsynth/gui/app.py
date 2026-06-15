from __future__ import annotations

import ast
import math
import operator
import threading
import time
import traceback
from datetime import datetime
from typing import Iterable

import numpy as np

import dearpygui.dearpygui as dpg

# Deliberately do not import xsynth.gui.adapters at module import time.
# adapters.py may import xsynth.gui.app for shared GUI helpers in some layouts,
# so importing AdaptSlot here can create a circular import.

SIGNAL_TYPES: list[str] = []
BEAM_REGIONS = ["ALL", "SA1", "SA2", "SA3", "SA4"]
DEVICE_NAMES = ["ADAPTX", "ADAPT_MLS"]
SLOT_1 = None
SLOT_2 = None
SCAN_THREAD = None
SCAN_CANCEL_REQUESTED = False
MAX_SCAN_LINES = 2
SCAN_CURRENT_INDEX = -1
SCAN_COMPLETED_COUNT = 0


# -----------------------------------------------------------------------------
# GUI options
# -----------------------------------------------------------------------------

DEFAULT_AUTOFIT_PLOT = False
DEFAULT_CLAMP_V0_V1_TO_BEAM_REGION = False
DEFAULT_BEAM_REGION = "SA3"
DEFAULT_OSCILLATOR = "line"


# -----------------------------------------------------------------------------
# Slot helpers
# -----------------------------------------------------------------------------


def initialise_slots() -> None:
    """Create both adaptation-server slots after this module has imported."""

    global SIGNAL_TYPES, SLOT_1, SLOT_2

    if SLOT_1 is not None and SLOT_2 is not None:
        return

    from xsynth.gui.adapters import AdaptSlot, get_oscillator_names

    SIGNAL_TYPES = get_oscillator_names()

    if SLOT_1 is None:
        SLOT_1 = AdaptSlot(
            device_name="ADAPTX",
            beam_region=DEFAULT_BEAM_REGION,
            oscillator_name=DEFAULT_OSCILLATOR,
        )

    if SLOT_2 is None:
        SLOT_2 = AdaptSlot(
            device_name="ADAPTY",
            beam_region=DEFAULT_BEAM_REGION,
            oscillator_name=DEFAULT_OSCILLATOR,
        )


def get_slot(slot_id: int):
    if slot_id == 1:
        return SLOT_1
    if slot_id == 2:
        return SLOT_2
    raise ValueError(f"Unknown slot_id {slot_id!r}")


def tag(slot_id: int, name: str) -> str:
    return f"slot{slot_id}_{name}"


# -----------------------------------------------------------------------------
# Logging / utility helpers
# -----------------------------------------------------------------------------


def log_message(message: str) -> None:
    """Append a timestamped message to the GUI log."""

    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}"

    if not dpg.does_item_exist("log_text"):
        print(line)
        return

    current = dpg.get_value("log_text") or ""
    dpg.set_value("log_text", f"{current}\n{line}".strip())


def log_exception(prefix: str) -> None:
    """Append an exception traceback to the GUI log."""

    log_message(prefix)
    log_message(traceback.format_exc())


def _exception_summary(exc: Exception) -> str:
    """Return a compact one-line exception description for status widgets."""

    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def set_slot_status(slot_id: int, message: str, is_error: bool = False) -> None:
    """Update the visible status line for one server slot."""

    status_tag = tag(slot_id, "status_text")
    prefix = "Error" if is_error else "Status"
    text = f"{prefix}: {message}"

    if dpg.does_item_exist(status_tag):
        dpg.set_value(status_tag, text)
    else:
        log_message(f"Slot {slot_id} {text}")


def _as_list(array_like) -> list:
    """Convert numpy/xarray/list-like data to a plain Python list."""

    if array_like is None:
        return []

    if hasattr(array_like, "values"):
        array_like = array_like.values

    if hasattr(array_like, "tolist"):
        return array_like.tolist()

    if isinstance(array_like, Iterable) and not isinstance(array_like, (str, bytes)):
        return list(array_like)

    return [array_like]


_ALLOWED_MATH_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def evaluate_math_input(value) -> float:
    """Evaluate a simple numeric expression from a GUI input field.

    Supported operations are +, -, *, /, //, %, ** and parentheses.
    Names, function calls, attributes, indexing and other Python syntax are
    intentionally rejected.
    """

    if isinstance(value, (int, float)):
        return float(value)

    expression = str(value).strip()
    if not expression:
        raise ValueError("empty expression")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("only numeric constants are allowed")

        if isinstance(node, ast.Num):  # pragma: no cover, older Python AST compatibility
            return float(node.n)

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_MATH_OPERATORS:
                raise ValueError(f"operator {op_type.__name__} is not allowed")
            return _ALLOWED_MATH_OPERATORS[op_type](_eval(node.left), _eval(node.right))

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_MATH_OPERATORS:
                raise ValueError(f"operator {op_type.__name__} is not allowed")
            return _ALLOWED_MATH_OPERATORS[op_type](_eval(node.operand))

        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    result = _eval(ast.parse(expression, mode="eval"))

    if not math.isfinite(result):
        raise ValueError("expression did not evaluate to a finite number")

    return float(result)


def _get_variable_step(name: str) -> float:
    """Return the increment used by the +/- buttons for a variable."""

    if name in {"V0", "V1"}:
        return 1.0

    return 1.0e-3


# -----------------------------------------------------------------------------
# Plot scaling / shading helpers
# -----------------------------------------------------------------------------


def _plot_autofit_enabled(slot_id: int) -> bool:
    """Return whether a single slot plot should autoscale on signal updates.

    This is deliberately per-slot. If it is disabled, manual pan/zoom/range
    adjustments made directly on the Dear PyGui plot are preserved.
    """

    checkbox = tag(slot_id, "autofit_plot_checkbox")
    if dpg.does_item_exist(checkbox):
        return bool(dpg.get_value(checkbox))
    return DEFAULT_AUTOFIT_PLOT


def _clamp_v0_v1_enabled(slot_id: int) -> bool:
    checkbox = tag(slot_id, "clamp_v0_v1_checkbox")
    if dpg.does_item_exist(checkbox):
        return bool(dpg.get_value(checkbox))
    return DEFAULT_CLAMP_V0_V1_TO_BEAM_REGION


def _set_variable_widget_value(slot_id: int, name: str, value: float) -> None:
    input_tag = tag(slot_id, f"input_{name}")
    if dpg.does_item_exist(input_tag):
        dpg.set_value(input_tag, f"{float(value):.12g}")


def get_beam_region_bounds(slot_id: int) -> tuple[float, float]:
    """Return the selected adaptation beam-region bounds for a slot."""

    slot = get_slot(slot_id)

    server = getattr(slot, "server", None)
    pulse_ids = getattr(server, "pulseId", None)

    if pulse_ids is not None:
        values = _as_list(pulse_ids)
        if values:
            return float(min(values)), float(max(values))

    t_values = _as_list(getattr(slot, "t", None))
    if t_values:
        return float(min(t_values)), float(max(t_values))

    return 0.0, 0.0


def _collect_current_plot_y_values(slot_id: int) -> list[float]:
    """Collect y-values from preview/readback series only."""

    y_values: list[float] = []

    for series_tag in (tag(slot_id, "preview_series"), tag(slot_id, "read_series")):
        if not dpg.does_item_exist(series_tag):
            continue

        try:
            value = dpg.get_value(series_tag)
        except Exception:
            continue

        if not value or len(value) < 2:
            continue

        for y in _as_list(value[1]):
            try:
                y_float = float(y)
            except Exception:
                continue
            if math.isfinite(y_float):
                y_values.append(y_float)

    return y_values


def _get_current_plot_y_span(slot_id: int) -> tuple[float, float]:
    y_values = _collect_current_plot_y_values(slot_id)

    if not y_values:
        return -1.0, 1.0

    y_min = min(y_values)
    y_max = max(y_values)

    if y_min == y_max:
        pad = 1.0 if y_min == 0.0 else abs(y_min) * 0.1
    else:
        pad = 0.05 * (y_max - y_min)

    return y_min - pad, y_max + pad


def _get_current_plot_x_span(slot_id: int) -> tuple[float, float] | None:
    """Return x-span from preview/readback traces only, not the shade series."""

    x_values: list[float] = []

    for series_tag in (tag(slot_id, "preview_series"), tag(slot_id, "read_series")):
        if not dpg.does_item_exist(series_tag):
            continue

        try:
            value = dpg.get_value(series_tag)
        except Exception:
            continue

        if not value or len(value) < 1:
            continue

        for x in _as_list(value[0]):
            try:
                x_float = float(x)
            except Exception:
                continue
            if math.isfinite(x_float):
                x_values.append(x_float)

    if not x_values:
        return None

    x_min = min(x_values)
    x_max = max(x_values)

    if x_min == x_max:
        pad = 1.0 if x_min == 0.0 else abs(x_min) * 0.1
    else:
        pad = 0.02 * (x_max - x_min)

    return x_min - pad, x_max + pad


def _fit_plot_axes_if_enabled(slot_id: int) -> None:
    """Autoscale slot plot using only line series, ignoring beam-region shade."""

    if not _plot_autofit_enabled(slot_id):
        return

    x_axis = tag(slot_id, "x_axis")
    y_axis = tag(slot_id, "y_axis")

    x_span = _get_current_plot_x_span(slot_id)
    if x_span is not None:
        dpg.set_axis_limits(x_axis, *x_span)
    elif dpg.does_item_exist(x_axis):
        dpg.fit_axis_data(x_axis)

    y_min, y_max = _get_current_plot_y_span(slot_id)
    dpg.set_axis_limits(y_axis, y_min, y_max)


def update_beam_region_shading(slot_id: int) -> None:
    """Shade the selected beam region without influencing autoscale."""

    shade_tag = tag(slot_id, "beam_region_shade")
    if not dpg.does_item_exist(shade_tag):
        return

    slot = get_slot(slot_id)

    if slot is None or slot.generator is None:
        dpg.set_value(shade_tag, [[], [], []])
        return

    try:
        ti, tf = get_beam_region_bounds(slot_id)
    except Exception:
        dpg.set_value(shade_tag, [[], [], []])
        return

    dpg.set_value(
        shade_tag,
        [
            [ti, tf],
            [-float(1e30), -float(1e30)],
            [float(1e30), float(1e30)],
        ],
    )


# -----------------------------------------------------------------------------
# Slot actions
# -----------------------------------------------------------------------------


def clamp_v0_v1_to_beam_region(slot_id: int) -> None:
    """Clip V0/V1 to selected beam-region bounds when clamp is enabled."""

    slot = get_slot(slot_id)

    if slot.generator is None or not _clamp_v0_v1_enabled(slot_id):
        return

    params = slot.signal_params or {}
    if "V0" not in params or "V1" not in params:
        return

    region_start, region_end = get_beam_region_bounds(slot_id)

    v0_old = float(params["V0"])
    v1_old = float(params["V1"])

    v0_new = min(max(v0_old, region_start), region_end)
    v1_new = min(max(v1_old, region_start), region_end)

    if v0_new > v1_new:
        v0_new, v1_new = v1_new, v0_new

    changed = (v0_new != v0_old) or (v1_new != v1_old)

    slot.update_variable("V0", v0_new)
    slot.update_variable("V1", v1_new)

    _set_variable_widget_value(slot_id, "V0", v0_new)
    _set_variable_widget_value(slot_id, "V1", v1_new)

    if changed:
        log_message(
            f"Slot {slot_id}: clamped V0/V1 to {slot.beam_region} "
            f"[{region_start:g}, {region_end:g}]."
        )


def toggle_clamp_v0_v1(sender, app_data, user_data) -> None:
    slot_id = int(user_data)
    if bool(app_data):
        clamp_v0_v1_to_beam_region(slot_id)
        update_preview_plot_only(slot_id)
        update_beam_region_shading(slot_id)
        log_message(f"Slot {slot_id}: enabled V0/V1 beam-region clamp.")
    else:
        log_message(f"Slot {slot_id}: disabled V0/V1 beam-region clamp.")


def toggle_autofit_plot(sender, app_data, user_data) -> None:
    slot_id = int(user_data)
    if bool(app_data):
        _fit_plot_axes_if_enabled(slot_id)
        log_message(f"Slot {slot_id}: enabled plot autofit.")
    else:
        log_message(f"Slot {slot_id}: disabled plot autofit; manual axis limits will be preserved.")


def fit_slot_plot_callback(sender, app_data, user_data) -> None:
    """Manually fit one plot once, without enabling automatic fitting."""

    slot_id = int(user_data)
    x_axis = tag(slot_id, "x_axis")
    y_axis = tag(slot_id, "y_axis")

    x_span = _get_current_plot_x_span(slot_id)
    if x_span is not None and dpg.does_item_exist(x_axis):
        dpg.set_axis_limits(x_axis, *x_span)

    if dpg.does_item_exist(y_axis):
        y_min, y_max = _get_current_plot_y_span(slot_id)
        dpg.set_axis_limits(y_axis, y_min, y_max)

    log_message(f"Slot {slot_id}: fitted plot to preview/readback traces.")


def fit_both_plots() -> None:
    fit_slot_plot_callback(None, None, 1)
    fit_slot_plot_callback(None, None, 2)


def connect_slot(slot_id: int) -> None:
    """Connect one GUI slot to the selected adaptation server."""

    slot = get_slot(slot_id)

    slot.device_name = dpg.get_value(tag(slot_id, "device"))
    slot.beam_region = dpg.get_value(tag(slot_id, "beam_region"))
    slot.oscillator_name = dpg.get_value(tag(slot_id, "oscillator"))

    try:
        slot.connect()
        clamp_v0_v1_to_beam_region(slot_id)
        rebuild_slot_variables(slot_id)
        refresh_scan_target_dropdowns()
        update_scan_preview()

        log_message(
            f"Connected slot {slot_id}: {slot.device_name}, "
            f"{slot.beam_region}, oscillator={slot.oscillator_name}"
        )
        set_slot_status(
            slot_id,
            f"Connected to {slot.device_name} {slot.beam_region}.",
        )

        update_preview_plot_only(slot_id)
        update_plot_from_readback(slot_id)

    except Exception as exc:
        set_slot_status(slot_id, f"Connect failed: {_exception_summary(exc)}", is_error=True)
        log_exception(f"Failed to connect slot {slot_id}.")


def connect_slot_callback(sender, app_data, user_data) -> None:
    connect_slot(int(user_data))


def set_slot_beam_region(sender, app_data, user_data) -> None:
    slot_id = int(user_data)
    slot = get_slot(slot_id)
    slot.beam_region = app_data

    server = getattr(slot, "server", None)
    if server is not None and hasattr(server, "beam_region"):
        try:
            server.beam_region = app_data
        except Exception:
            pass

    try:
        clamp_v0_v1_to_beam_region(slot_id)
        rebuild_slot_variables(slot_id)
        refresh_scan_target_dropdowns()
        update_scan_preview()
        update_preview_plot_only(slot_id)
        update_beam_region_shading(slot_id)
        set_slot_status(slot_id, f"Beam region set to {app_data}.")
        log_message(f"Slot {slot_id}: beam region set to {app_data}")
    except Exception as exc:
        set_slot_status(slot_id, f"Beam-region update failed: {_exception_summary(exc)}", is_error=True)
        log_exception(f"Failed to update slot {slot_id} beam region.")


def set_slot_oscillator(sender, app_data, user_data) -> None:
    slot_id = int(user_data)
    slot = get_slot(slot_id)
    slot.oscillator_name = app_data

    if slot.generator is None:
        return

    try:
        old_params = slot.signal_params or {}
        preserved_kwargs = {key: old_params[key] for key in ("V0", "V1") if key in old_params}

        slot.set_oscillator(app_data, **preserved_kwargs)

        if _clamp_v0_v1_enabled(slot_id):
            clamp_v0_v1_to_beam_region(slot_id)

        rebuild_slot_variables(slot_id)
        refresh_scan_target_dropdowns()
        update_scan_preview()
        update_preview_plot_only(slot_id)
        update_plot_from_readback(slot_id)

        log_message(
            f"Slot {slot_id}: oscillator set to {app_data}; "
            f"preserved V0={preserved_kwargs.get('V0')}, "
            f"V1={preserved_kwargs.get('V1')}"
        )

    except Exception:
        log_exception(f"Failed to set slot {slot_id} oscillator.")


def update_slot_variable(sender, app_data, user_data) -> None:
    """Update one editable oscillator variable."""

    slot_id, name = user_data
    slot = get_slot(int(slot_id))

    try:
        value = evaluate_math_input(app_data)
        slot.update_variable(name, value)

        if name in {"V0", "V1"}:
            clamp_v0_v1_to_beam_region(slot_id)
        else:
            _set_variable_widget_value(slot_id, name, value)

        update_preview_plot_only(slot_id)

    except Exception:
        log_exception(f"Failed to update slot {slot_id} {name} from input {app_data!r}.")


def step_slot_variable(sender, app_data, user_data) -> None:
    """Increment or decrement a variable using its +/- button."""

    slot_id, name, direction = user_data
    slot = get_slot(int(slot_id))

    try:
        input_tag = tag(slot_id, f"input_{name}")
        current = evaluate_math_input(dpg.get_value(input_tag))
        value = current + float(direction) * _get_variable_step(name)

        slot.update_variable(name, value)

        if name in {"V0", "V1"}:
            clamp_v0_v1_to_beam_region(slot_id)
        else:
            _set_variable_widget_value(slot_id, name, value)

        update_preview_plot_only(slot_id)

    except Exception:
        log_exception(f"Failed to step slot {slot_id} {name}.")


def rebuild_slot_variables(slot_id: int) -> None:
    """Rebuild variable controls for one selected oscillator."""

    group_tag = tag(slot_id, "variables_group")
    if not dpg.does_item_exist(group_tag):
        return

    dpg.delete_item(group_tag, children_only=True)

    slot = get_slot(slot_id)

    if slot.generator is None:
        dpg.add_text("Not connected.", parent=group_tag)
        return

    variables = slot.variables
    defaults = slot.signal_params

    if not variables:
        dpg.add_text("No variables found.", parent=group_tag)
        return

    with dpg.table(
        parent=group_tag,
        header_row=False,
        borders_innerH=True,
        borders_innerV=True,
        borders_outerH=True,
        borders_outerV=True,
        policy=dpg.mvTable_SizingStretchProp,
    ):
        dpg.add_table_column(init_width_or_weight=1.0)
        dpg.add_table_column(width_fixed=True, init_width_or_weight=34)
        dpg.add_table_column(width_fixed=True, init_width_or_weight=185)
        dpg.add_table_column(width_fixed=True, init_width_or_weight=34)

        for name, description in variables.items():
            default = float(defaults.get(name, 0.0))
            default_text = f"{default:.0f}" if name in {"V0", "V1"} else f"{default:.6e}"

            with dpg.table_row():
                dpg.add_text(f"{name}: {description}")
                dpg.add_button(
                    label="-",
                    tag=tag(slot_id, f"decrement_{name}"),
                    width=28,
                    callback=step_slot_variable,
                    user_data=(slot_id, name, -1.0),
                )
                dpg.add_input_text(
                    tag=tag(slot_id, f"input_{name}"),
                    default_value=default_text,
                    width=170,
                    callback=update_slot_variable,
                    on_enter=True,
                    user_data=(slot_id, name),
                )
                dpg.add_button(
                    label="+",
                    tag=tag(slot_id, f"increment_{name}"),
                    width=28,
                    callback=step_slot_variable,
                    user_data=(slot_id, name, 1.0),
                )


# -----------------------------------------------------------------------------
# Plot / write callbacks
# -----------------------------------------------------------------------------


def update_preview_plot_only(slot_id: int) -> None:
    """Plot the signal generated by one GUI oscillator."""

    slot = get_slot(slot_id)
    if slot.generator is None:
        return

    try:
        y_preview = _as_list(slot.preview_signal)
        x = _as_list(slot.t)

        if len(x) != len(y_preview):
            x = list(range(len(y_preview)))

        dpg.set_value(tag(slot_id, "preview_series"), [x, y_preview])
        update_beam_region_shading(slot_id)
        _fit_plot_axes_if_enabled(slot_id)

    except Exception:
        log_exception(f"Failed to update slot {slot_id} generated preview plot.")


def update_plot_from_readback(slot_id: int) -> None:
    """Read one current server value with .read() and plot it."""

    slot = get_slot(slot_id)
    if slot.server is None:
        log_message(f"Cannot update slot {slot_id} readback plot: slot is not connected.")
        return

    try:
        y_read = _as_list(slot.read_signal)
        x = _as_list(slot.t)

        if len(x) != len(y_read):
            x = list(range(len(y_read)))

        dpg.set_value(tag(slot_id, "read_series"), [x, y_read])
        update_beam_region_shading(slot_id)
        _fit_plot_axes_if_enabled(slot_id)

        set_slot_status(slot_id, f"Readback updated from {slot.device_name}.")
        log_message(f"Updated slot {slot_id} {slot.device_name} readback plot from server.read().")

    except Exception as exc:
        set_slot_status(slot_id, f"Readback failed: {_exception_summary(exc)}", is_error=True)
        log_exception(f"Failed to read or plot slot {slot_id} readback.")


def update_readback_callback(sender, app_data, user_data) -> None:
    update_plot_from_readback(int(user_data))


def write_slot(slot_id: int) -> None:
    """Write one generated signal to its adaptation server."""

    slot = get_slot(slot_id)

    try:
        clamp_v0_v1_to_beam_region(slot_id)
        slot.write()
        set_slot_status(slot_id, f"Wrote signal to {slot.device_name}.")
        log_message(f"Wrote slot {slot_id} signal to {slot.device_name}.")
        update_plot_from_readback(slot_id)

    except Exception as exc:
        set_slot_status(slot_id, f"Write failed: {_exception_summary(exc)}", is_error=True)
        log_exception(f"Failed to write slot {slot_id} signal.")


def write_slot_callback(sender, app_data, user_data) -> None:
    write_slot(int(user_data))


def update_all_readbacks() -> None:
    update_plot_from_readback(1)
    update_plot_from_readback(2)


def write_all_slots() -> None:
    write_slot(1)
    write_slot(2)


def clear_log() -> None:
    if dpg.does_item_exist("log_text"):
        dpg.set_value("log_text", "")



# -----------------------------------------------------------------------------
# MiniScan-style GUI scan helpers
# -----------------------------------------------------------------------------


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} min {sec:.0f} s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)} h {int(minutes)} min {sec:.0f} s"


def _get_slot_variable_names(slot_id: int) -> list[str]:
    """Return variable names for a connected slot, or a sensible fallback."""

    try:
        slot = get_slot(slot_id)
        variables = getattr(slot, "variables", None)
        if variables:
            return list(variables.keys())
    except Exception:
        pass

    return [f"V{i}" for i in range(8)]


def get_scan_target_items() -> list[str]:
    """Return dropdown entries such as 'Server 1: V2'."""

    items: list[str] = []
    for slot_id in (1, 2):
        for variable_name in _get_slot_variable_names(slot_id):
            items.append(f"Server {slot_id}: {variable_name}")
    return items


def parse_scan_target(item: str) -> tuple[int, str]:
    """Parse a scan target dropdown item."""

    prefix, variable_name = item.split(":", 1)
    slot_id = int(prefix.strip().split()[1])
    return slot_id, variable_name.strip()


def refresh_scan_target_dropdowns() -> None:
    """Refresh scan target combo items after a slot connects or oscillator changes."""

    items = get_scan_target_items()
    if not items:
        return

    for line_id in range(MAX_SCAN_LINES):
        combo_tag = f"scan_line_{line_id}_target"
        if not dpg.does_item_exist(combo_tag):
            continue

        current = dpg.get_value(combo_tag)
        dpg.configure_item(combo_tag, items=items)
        if current not in items:
            dpg.set_value(combo_tag, items[min(line_id, len(items) - 1)])


def _get_scan_line(line_id: int) -> dict | None:
    """Read one enabled scan line from the GUI."""

    enabled_tag = f"scan_line_{line_id}_enabled"
    target_tag = f"scan_line_{line_id}_target"
    start_tag = f"scan_line_{line_id}_start"
    stop_tag = f"scan_line_{line_id}_stop"
    points_tag = f"scan_line_{line_id}_points"

    if not dpg.does_item_exist(enabled_tag) or not dpg.get_value(enabled_tag):
        return None

    target = dpg.get_value(target_tag)
    if not target:
        return None

    slot_id, variable_name = parse_scan_target(target)
    start = evaluate_math_input(dpg.get_value(start_tag))
    stop = evaluate_math_input(dpg.get_value(stop_tag))
    points = int(max(1, round(evaluate_math_input(dpg.get_value(points_tag)))))

    return {
        "slot_id": slot_id,
        "variable_name": variable_name,
        "target": target,
        "start": start,
        "stop": stop,
        "points": points,
        "vector": np.linspace(start, stop, points, dtype=float),
    }


def get_scan_configuration() -> tuple[list[dict], float, np.ndarray]:
    """Build a MiniScan-style mesh from the enabled scan lines."""

    dwell = evaluate_math_input(dpg.get_value("scan_dwell_time")) if dpg.does_item_exist("scan_dwell_time") else 0.1
    dwell = max(0.0, float(dwell))

    lines = []
    for line_id in range(MAX_SCAN_LINES):
        line = _get_scan_line(line_id)
        if line is not None:
            lines.append(line)

    if not lines:
        return [], dwell, np.empty((0, 0), dtype=float)

    scan_vectors = [line["vector"] for line in lines]
    mesh = np.meshgrid(*scan_vectors, indexing="ij")
    scan_points = np.vstack([m.ravel() for m in mesh]).T
    return lines, dwell, scan_points



def make_scan_point_themes() -> None:
    """Create themes for completed/current/upcoming scan-point markers."""

    themes = {
        "scan_completed_theme": (70, 200, 120, 210),
        "scan_current_theme": (255, 210, 80, 255),
        "scan_upcoming_theme": (160, 160, 160, 120),
    }

    for theme_tag, colour in themes.items():
        if dpg.does_item_exist(theme_tag):
            continue
        with dpg.theme(tag=theme_tag):
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_MarkerFill,
                    colour,
                    category=dpg.mvThemeCat_Plots,
                )
                dpg.add_theme_color(
                    dpg.mvPlotCol_MarkerOutline,
                    colour,
                    category=dpg.mvThemeCat_Plots,
                )


def _get_scan_plot_xy(lines: list[dict], scan_points: np.ndarray) -> tuple[list[float], list[float]]:
    """Return x/y arrays for the scan-point preview plot."""

    n_points = len(scan_points)
    if n_points == 0:
        return [], []

    if len(lines) == 1:
        x = list(range(n_points))
        y = scan_points[:, 0].astype(float).tolist()
        if dpg.does_item_exist("scan_x_axis"):
            dpg.configure_item("scan_x_axis", label="scan index")
        if dpg.does_item_exist("scan_y_axis"):
            dpg.configure_item("scan_y_axis", label=lines[0]["target"])
        return x, y

    x = scan_points[:, 0].astype(float).tolist()
    y = scan_points[:, 1].astype(float).tolist()
    if dpg.does_item_exist("scan_x_axis"):
        dpg.configure_item("scan_x_axis", label=lines[0]["target"])
    if dpg.does_item_exist("scan_y_axis"):
        dpg.configure_item("scan_y_axis", label=lines[1]["target"])
    return x, y


def update_scan_point_plot(completed_count: int = 0, current_index: int = -1, fit_axes: bool = False) -> None:
    """Update scan-point plot with completed/current/upcoming marker groups."""

    try:
        lines, dwell, scan_points = get_scan_configuration()
        x, y = _get_scan_plot_xy(lines, scan_points)
        n_points = len(x)

        completed_count = min(max(int(completed_count), 0), n_points)
        current_index = int(current_index)

        completed_x = x[:completed_count]
        completed_y = y[:completed_count]

        if 0 <= current_index < n_points:
            current_x = [x[current_index]]
            current_y = [y[current_index]]
            upcoming_start = max(current_index + 1, completed_count)
        else:
            current_x = []
            current_y = []
            upcoming_start = completed_count

        upcoming_x = x[upcoming_start:]
        upcoming_y = y[upcoming_start:]

        if dpg.does_item_exist("scan_points_completed_series"):
            dpg.set_value("scan_points_completed_series", [completed_x, completed_y])
        if dpg.does_item_exist("scan_points_current_series"):
            dpg.set_value("scan_points_current_series", [current_x, current_y])
        if dpg.does_item_exist("scan_points_upcoming_series"):
            dpg.set_value("scan_points_upcoming_series", [upcoming_x, upcoming_y])

        if fit_axes:
            if dpg.does_item_exist("scan_x_axis"):
                dpg.fit_axis_data("scan_x_axis")
            if dpg.does_item_exist("scan_y_axis"):
                dpg.fit_axis_data("scan_y_axis")

    except Exception:
        log_exception("Failed to update scan-point plot.")

def update_scan_preview(sender=None, app_data=None, user_data=None) -> None:
    """Update scan duration estimate and scan-point preview plot."""

    try:
        lines, dwell, scan_points = get_scan_configuration()
        n_points = len(scan_points)
        total_duration = n_points * dwell

        if dpg.does_item_exist("scan_duration_text"):
            dpg.set_value(
                "scan_duration_text",
                f"Scan points: {n_points} | dwell: {dwell:g} s | estimated duration: {_format_seconds(total_duration)}",
            )

        update_scan_point_plot(completed_count=0, current_index=-1, fit_axes=True)

    except Exception:
        log_exception("Failed to update scan preview.")


def _set_scan_status(progress: float, text: str) -> None:
    progress = min(max(float(progress), 0.0), 1.0)
    if dpg.does_item_exist("scan_progress_bar"):
        dpg.set_value("scan_progress_bar", progress)
        dpg.configure_item("scan_progress_bar", overlay=f"{100.0 * progress:.1f}%")
    if dpg.does_item_exist("scan_status_text"):
        dpg.set_value("scan_status_text", text)


def _apply_scan_point(lines: list[dict], point: np.ndarray) -> list[int]:
    """Apply one scan point to the selected slot variables."""

    touched_slots: list[int] = []

    for line, value in zip(lines, point):
        slot_id = int(line["slot_id"])
        variable_name = line["variable_name"]
        slot = get_slot(slot_id)

        if slot.generator is None:
            raise RuntimeError(f"Server {slot_id} is not connected.")

        slot.update_variable(variable_name, float(value))
        _set_variable_widget_value(slot_id, variable_name, float(value))

        if variable_name in {"V0", "V1"}:
            clamp_v0_v1_to_beam_region(slot_id)

        if slot_id not in touched_slots:
            touched_slots.append(slot_id)

    return touched_slots



def write_both_slots_for_scan() -> None:
    """Write both configured server slots during a scan point.

    This intentionally mirrors the GUI-level 'Write both' behaviour rather than
    writing only the slot whose variable was changed at the current scan point.
    """

    for slot_id in (1, 2):
        slot = get_slot(slot_id)
        if slot.generator is None or slot.server is None:
            continue
        clamp_v0_v1_to_beam_region(slot_id)
        try:
            slot.write()
            set_slot_status(slot_id, f"Wrote scan point to {slot.device_name}.")
        except Exception as exc:
            set_slot_status(slot_id, f"Scan write failed: {_exception_summary(exc)}", is_error=True)
            raise


def _execute_scan_worker(lines: list[dict], dwell: float, scan_points: np.ndarray, write_each_point: bool) -> None:
    """Execute a simple MiniScan-like scan in a background thread."""

    global SCAN_CANCEL_REQUESTED, SCAN_THREAD, SCAN_CURRENT_INDEX, SCAN_COMPLETED_COUNT

    n_points = len(scan_points)
    start_time = time.time()

    try:
        for point_index, point in enumerate(scan_points):
            if SCAN_CANCEL_REQUESTED:
                _set_scan_status(point_index / max(n_points, 1), "Scan cancelled.")
                log_message("Scan cancelled.")
                return

            point_start = time.time()
            SCAN_CURRENT_INDEX = point_index
            SCAN_COMPLETED_COUNT = point_index
            update_scan_point_plot(completed_count=SCAN_COMPLETED_COUNT, current_index=SCAN_CURRENT_INDEX)

            touched_slots = _apply_scan_point(lines, point)

            for slot_id in touched_slots:
                update_preview_plot_only(slot_id)

            if write_each_point:
                write_both_slots_for_scan()

            elapsed = time.time() - start_time
            remaining = max(0.0, (n_points - point_index - 1) * dwell)
            _set_scan_status(
                (point_index + 1) / max(n_points, 1),
                f"Point {point_index + 1}/{n_points} | elapsed {_format_seconds(elapsed)} | remaining ~{_format_seconds(remaining)}",
            )

            sleep_time = dwell - (time.time() - point_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

            SCAN_COMPLETED_COUNT = point_index + 1
            update_scan_point_plot(completed_count=SCAN_COMPLETED_COUNT, current_index=-1)

        SCAN_CURRENT_INDEX = -1
        SCAN_COMPLETED_COUNT = n_points
        update_scan_point_plot(completed_count=SCAN_COMPLETED_COUNT, current_index=-1)
        _set_scan_status(1.0, f"Scan complete. Total elapsed {_format_seconds(time.time() - start_time)}.")
        log_message(f"Scan complete: {n_points} points in {_format_seconds(time.time() - start_time)}.")

    except Exception:
        log_exception("Scan failed.")
        _set_scan_status(0.0, "Scan failed. See log.")

    finally:
        SCAN_THREAD = None
        SCAN_CANCEL_REQUESTED = False


def start_scan(sender=None, app_data=None, user_data=None) -> None:
    """Start the configured scan."""

    global SCAN_THREAD, SCAN_CANCEL_REQUESTED, SCAN_CURRENT_INDEX, SCAN_COMPLETED_COUNT

    if SCAN_THREAD is not None and SCAN_THREAD.is_alive():
        log_message("A scan is already running.")
        return

    try:
        lines, dwell, scan_points = get_scan_configuration()
        if not lines or len(scan_points) == 0:
            log_message("No enabled scan lines. Enable at least one scan variable.")
            return

        write_each_point = bool(dpg.get_value("scan_write_each_point")) if dpg.does_item_exist("scan_write_each_point") else False
        SCAN_CANCEL_REQUESTED = False
        SCAN_CURRENT_INDEX = -1
        SCAN_COMPLETED_COUNT = 0
        update_scan_point_plot(completed_count=0, current_index=-1, fit_axes=True)
        _set_scan_status(0.0, "Starting scan...")

        SCAN_THREAD = threading.Thread(
            target=_execute_scan_worker,
            args=(lines, dwell, scan_points, write_each_point),
            daemon=True,
        )
        SCAN_THREAD.start()

        log_message(
            f"Started scan: {len(scan_points)} points, dwell={dwell:g} s, "
            f"estimated duration {_format_seconds(len(scan_points) * dwell)}, write={write_each_point}."
        )

    except Exception:
        log_exception("Failed to start scan.")


def cancel_scan(sender=None, app_data=None, user_data=None) -> None:
    """Request cancellation of the active scan."""

    global SCAN_CANCEL_REQUESTED
    SCAN_CANCEL_REQUESTED = True
    _set_scan_status(dpg.get_value("scan_progress_bar") if dpg.does_item_exist("scan_progress_bar") else 0.0, "Cancelling scan...")


def make_scan_panel() -> None:
    """Create the RHS MiniScan-style scan panel."""

    dpg.add_text("MiniScan")
    dpg.add_text("Enable one line for a 1D scan; enable two or more for a mesh scan.", wrap=320)

    with dpg.group(horizontal=True):
        dpg.add_input_text(
            tag="scan_dwell_time",
            label="Dwell / s",
            default_value="0.1",
            width=90,
            callback=update_scan_preview,
            on_enter=True,
        )
        dpg.add_checkbox(
            tag="scan_write_each_point",
            label="Write both at each point",
            default_value=False,
        )

    dpg.add_separator()

    items = get_scan_target_items()
    for line_id in range(MAX_SCAN_LINES):
        with dpg.group(horizontal=True):
            dpg.add_checkbox(
                tag=f"scan_line_{line_id}_enabled",
                default_value=(line_id == 0),
                callback=update_scan_preview,
            )
            dpg.add_combo(
                items,
                tag=f"scan_line_{line_id}_target",
                default_value=items[min(line_id, len(items) - 1)] if items else "",
                width=135,
                callback=update_scan_preview,
            )
            dpg.add_input_text(
                tag=f"scan_line_{line_id}_start",
                default_value="0",
                width=65,
                callback=update_scan_preview,
                on_enter=True,
            )
            dpg.add_input_text(
                tag=f"scan_line_{line_id}_stop",
                default_value="1",
                width=65,
                callback=update_scan_preview,
                on_enter=True,
            )
            dpg.add_input_text(
                tag=f"scan_line_{line_id}_points",
                default_value="11",
                width=48,
                callback=update_scan_preview,
                on_enter=True,
            )

    dpg.add_text("Columns: enable | target | start | stop | points", wrap=320)
    dpg.add_text("", tag="scan_duration_text", wrap=340)

    with dpg.group(horizontal=True):
        dpg.add_button(label="Update points", callback=update_scan_preview)
        dpg.add_button(label="Start scan", callback=start_scan)
        dpg.add_button(label="Cancel", callback=cancel_scan)

    dpg.add_progress_bar(tag="scan_progress_bar", default_value=0.0, width=-1, overlay="0.0%")
    dpg.add_text("Idle.", tag="scan_status_text", wrap=340)

    make_scan_point_themes()
    with dpg.plot(label="Scan points", height=330, width=-1):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label="scan index", tag="scan_x_axis")
        dpg.add_plot_axis(dpg.mvYAxis, label="scan value", tag="scan_y_axis")
        dpg.add_scatter_series([], [], label="Completed", parent="scan_y_axis", tag="scan_points_completed_series")
        dpg.add_scatter_series([], [], label="Current", parent="scan_y_axis", tag="scan_points_current_series")
        dpg.add_scatter_series([], [], label="Upcoming", parent="scan_y_axis", tag="scan_points_upcoming_series")
        dpg.bind_item_theme("scan_points_completed_series", "scan_completed_theme")
        dpg.bind_item_theme("scan_points_current_series", "scan_current_theme")
        dpg.bind_item_theme("scan_points_upcoming_series", "scan_upcoming_theme")

    update_scan_preview()


# -----------------------------------------------------------------------------
# GUI construction
# -----------------------------------------------------------------------------


def make_server_controls(slot_id: int) -> None:
    """Create compact server controls for one slot."""

    slot = get_slot(slot_id)

    with dpg.group():
        dpg.add_text(f"Server {slot_id}")

        dpg.add_combo(
            DEVICE_NAMES,
            default_value=slot.device_name,
            label="Device",
            tag=tag(slot_id, "device"),
            width=120,
        )

        dpg.add_combo(
            BEAM_REGIONS,
            default_value=slot.beam_region,
            label="Beam region",
            tag=tag(slot_id, "beam_region"),
            callback=set_slot_beam_region,
            user_data=slot_id,
            width=120,
        )

        dpg.add_combo(
            SIGNAL_TYPES,
            default_value=slot.oscillator_name,
            label="Oscillator",
            tag=tag(slot_id, "oscillator"),
            callback=set_slot_oscillator,
            user_data=slot_id,
            width=120,
        )

        with dpg.group(horizontal=True):
            dpg.add_button(label="Connect", callback=connect_slot_callback, user_data=slot_id)
            dpg.add_button(label="Read", callback=update_readback_callback, user_data=slot_id)
            dpg.add_button(label="Write", callback=write_slot_callback, user_data=slot_id)

        dpg.add_checkbox(
            label="Clamp V0/V1",
            tag=tag(slot_id, "clamp_v0_v1_checkbox"),
            default_value=DEFAULT_CLAMP_V0_V1_TO_BEAM_REGION,
            callback=toggle_clamp_v0_v1,
            user_data=slot_id,
        )

        dpg.add_text(
            "Status: Not connected.",
            tag=tag(slot_id, "status_text"),
            wrap=260,
        )


def make_variables_panel(slot_id: int) -> None:
    """Create variables section for one slot."""

    with dpg.child_window(label=f"Variables {slot_id}", height=230, width=-1, border=True):
        dpg.add_text(f"Variables {slot_id}")
        dpg.add_group(tag=tag(slot_id, "variables_group"))


def make_plot_panel(slot_id: int) -> None:
    """Create preview/readback plot for one slot."""

    slot = get_slot(slot_id)

    with dpg.group(horizontal=True):
        dpg.add_checkbox(
            label="Autofit this plot",
            tag=tag(slot_id, "autofit_plot_checkbox"),
            default_value=DEFAULT_AUTOFIT_PLOT,
            callback=toggle_autofit_plot,
            user_data=slot_id,
        )
        dpg.add_button(
            label="Fit once",
            callback=fit_slot_plot_callback,
            user_data=slot_id,
        )

    with dpg.plot(label=f"Server {slot_id}: {slot.device_name}", height=-1, width=-1):
        dpg.add_plot_legend()

        dpg.add_plot_axis(
            dpg.mvXAxis,
            label="pulseId / bunch index",
            tag=tag(slot_id, "x_axis"),
        )

        dpg.add_plot_axis(
            dpg.mvYAxis,
            label="signal",
            tag=tag(slot_id, "y_axis"),
        )

        # Draw the selected beam region first so preview/readback traces sit on top.
        if hasattr(dpg, "add_shade_series"):
            dpg.add_shade_series(
                [],
                [],
                y2=[],
                label="Selected beam region",
                parent=tag(slot_id, "y_axis"),
                tag=tag(slot_id, "beam_region_shade"),
            )
        elif hasattr(dpg, "add_inf_line_series"):
            dpg.add_inf_line_series(
                [],
                label="Beam-region start/end",
                parent=tag(slot_id, "x_axis"),
                tag=tag(slot_id, "beam_region_lines"),
            )

        dpg.add_line_series(
            [],
            [],
            label="Generated preview",
            parent=tag(slot_id, "y_axis"),
            tag=tag(slot_id, "preview_series"),
        )

        dpg.add_line_series(
            [],
            [],
            label=f"{slot.device_name} readback",
            parent=tag(slot_id, "y_axis"),
            tag=tag(slot_id, "read_series"),
        )


def make_log_panel() -> None:
    """Create left-side log and global action panel."""

    dpg.add_text("Log")

    dpg.add_input_text(
        tag="log_text",
        multiline=True,
        readonly=True,
        height=520,
        width=-1,
    )

    dpg.add_separator()

    with dpg.group(horizontal=True):
        dpg.add_button(label="Read both", callback=update_all_readbacks)
        dpg.add_button(label="Write both", callback=write_all_slots)

    dpg.add_button(label="Clear log", callback=clear_log)


def make_main_layout() -> None:
    """Create the main GUI layout.

    The outer layout is a three-column stretch table rather than a horizontal
    group. This guarantees that the scan panel remains visible instead of being
    pushed off-screen by a width=-1 child window.

    Columns:
        left:   log/options/actions
        middle: two independent server slots, plots, and variables
        right:  MiniScan-style scan panel
    """

    with dpg.table(
        header_row=False,
        policy=dpg.mvTable_SizingStretchProp,
        resizable=True,
        borders_innerV=True,
        width=-1,
        height=-1,
    ):
        # Use proportional widths so all three panels are visible on launch.
        dpg.add_table_column(init_width_or_weight=0.75)
        dpg.add_table_column(init_width_or_weight=2.45)
        dpg.add_table_column(init_width_or_weight=0.95)

        with dpg.table_row():
            # Left column: log and global actions.
            with dpg.child_window(label="Global / log", width=-1, height=-1, border=True):
                dpg.add_text("Global actions")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Read both", callback=update_all_readbacks)
                    dpg.add_button(label="Write both", callback=write_all_slots)
                dpg.add_button(label="Fit both plots once", callback=fit_both_plots)

                dpg.add_separator()
                make_log_panel()

            # Middle column: two independent slots, arranged in rows.
            with dpg.child_window(label="Server slots", width=-1, height=-1, border=True):
                # Row 1: independent server selectors / connection controls.
                with dpg.table(
                    header_row=False,
                    policy=dpg.mvTable_SizingStretchProp,
                    resizable=True,
                    borders_innerV=True,
                    width=-1,
                ):
                    dpg.add_table_column(init_width_or_weight=1.0)
                    dpg.add_table_column(init_width_or_weight=1.0)
                    with dpg.table_row():
                        with dpg.child_window(label="Server slot 1", height=190, border=True):
                            make_server_controls(1)
                        with dpg.child_window(label="Server slot 2", height=190, border=True):
                            make_server_controls(2)

                # Row 2: independent plots.
                with dpg.table(
                    header_row=False,
                    policy=dpg.mvTable_SizingStretchProp,
                    resizable=True,
                    borders_innerV=True,
                    width=-1,
                ):
                    dpg.add_table_column(init_width_or_weight=1.0)
                    dpg.add_table_column(init_width_or_weight=1.0)
                    with dpg.table_row():
                        with dpg.child_window(label="Plot 1", height=430, border=True):
                            make_plot_panel(1)
                        with dpg.child_window(label="Plot 2", height=430, border=True):
                            make_plot_panel(2)

                # Row 3: independent variable panels/signals.
                with dpg.table(
                    header_row=False,
                    policy=dpg.mvTable_SizingStretchProp,
                    resizable=True,
                    borders_innerV=True,
                    width=-1,
                ):
                    dpg.add_table_column(init_width_or_weight=1.0)
                    dpg.add_table_column(init_width_or_weight=1.0)
                    with dpg.table_row():
                        with dpg.child_window(label="Variables / signal 1", height=-1, border=True):
                            make_variables_panel(1)
                        with dpg.child_window(label="Variables / signal 2", height=-1, border=True):
                            make_variables_panel(2)

            # Right column: scan panel. This is now guaranteed visible.
            with dpg.child_window(label="Scan panel", width=-1, height=-1, border=True):
                make_scan_panel()

def main() -> None:
    """Launch the XSynth Dear PyGui application."""

    initialise_slots()

    dpg.create_context()

    with dpg.window(label="XSynth GUI", width=1700, height=950):
        make_main_layout()

    dpg.create_viewport(title="XSynth", width=1700, height=950)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    log_message("XSynth GUI started.")
    log_message("Connect each server to load its current readback.")
    log_message("Input boxes accept direct numeric math, e.g. 1e-3/2 or (10+5)*2.")
    log_message("Autofit is per plot and off by default, so manual pan/zoom is preserved.")
    log_message("MiniScan panel is available on the far right; enable one or two scan variables and press Update points.")

    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
