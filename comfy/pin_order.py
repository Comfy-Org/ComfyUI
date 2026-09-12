from __future__ import annotations

import weakref


class NullPinOrder:
    enabled = False
    budget_checked = False
    current = -1

    def advance(self):
        self.current += 1

    def preferred_indices(self):
        return []

    def close(self):
        pass


class PrefetchPinOrder:
    enabled = True

    def __init__(self, units, window=3):
        self.window = window
        self.current = -1
        self.budget_checked = False
        self.positions = weakref.WeakKeyDictionary()
        self.modules = []
        self.pin_states = []
        for index, unit in enumerate(units):
            roots = unit if isinstance(unit, (list, tuple)) else (unit,)
            modules = []
            for root in roots:
                for module in root.modules():
                    if hasattr(module, "_v"):
                        sources = [module]
                        for param_key in ("weight", "bias"):
                            lowvram_source = getattr(module, param_key + "_lowvram_function", None)
                            if lowvram_source is not None:
                                sources.append(lowvram_source)
                        for source in sources:
                            modules.append(source)
                            self.add_source(source, index)
            self.modules.append(modules)

    def add_source(self, source, index):
        if source in self.positions:
            return
        self.positions[source] = index
        pin_state = source._pin_state
        if not any(state is pin_state for state in self.pin_states):
            pin_state["prefetch_orders"].add(self)
            self.pin_states.append(pin_state)

    def copy_position(self, source, target):
        index = self.positions.get(source)
        if index is not None:
            self.add_source(target, index)

    def advance(self):
        self.current += 1
        self.budget_checked = False

    def state(self, module):
        index = self.positions.get(module)
        if index is None or self.current < 0:
            return None
        if index >= self.current:
            distance = index - self.current
            preferred = distance < self.window
        else:
            distance = len(self.modules) + index
            preferred = False
        return preferred, distance

    def preferred_indices(self):
        if self.current < 0:
            return []
        return list(range(self.current, min(len(self.modules), self.current + self.window)))

    def close(self):
        for pin_state in self.pin_states:
            pin_state["prefetch_orders"].discard(self)
        self.pin_states.clear()
        self.positions.clear()
        self.modules.clear()


def _prefetch_orders(module):
    return [order for order in module._pin_state["prefetch_orders"] if module in order.positions]


def prefetch_pin_state(module):
    states = [state for order in _prefetch_orders(module) if (state := order.state(module)) is not None]
    if not states:
        return None
    return min(states, key=lambda state: (not state[0], state[1]))


def prefetch_budget_checked(module):
    orders = _prefetch_orders(module)
    return bool(orders) and all(order.budget_checked for order in orders)
