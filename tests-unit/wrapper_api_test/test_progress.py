"""Tests for the wrapper job progress (api_wrapper/progress.py).

The registry is stubbed with plain objects: the real one lives in
comfy_execution.progress, whose import pulls in PIL, tqdm and the server
protocol — none of which this arithmetic needs.
"""

import enum
from types import SimpleNamespace

import pytest

from api_wrapper import progress as wp


class State(enum.Enum):
    """Stands in for comfy_execution.progress.NodeState (same values)."""

    Pending = "pending"
    Running = "running"
    Finished = "finished"
    Error = "error"


# A cut-down H3 graph: loaders, a scheduler asking for 30 steps, the sampler
# (no steps input of its own — it reads the scheduler's sigmas), decode, save.
PROMPT = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "h3.safetensors"}},
    "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "t5.safetensors"}},
    "3": {"class_type": "BasicScheduler", "inputs": {"model": ["1", 0], "steps": 30, "denoise": 1.0}},
    "4": {"class_type": "SamplerCustomAdvanced", "inputs": {"sigmas": ["3", 0]}},
    "5": {"class_type": "VAEDecode", "inputs": {"samples": ["4", 0]}},
    "6": {"class_type": "SaveVideo", "inputs": {"video": ["5", 0]}},
}


def node(state, value=0.0, maximum=1.0):
    return {"state": state, "value": value, "max": maximum}


def registry(job_id, nodes, prompt=PROMPT):
    dyn = SimpleNamespace(original_prompt=prompt, get_node=lambda node_id: prompt[node_id])
    return SimpleNamespace(prompt_id=job_id, nodes=nodes, dynprompt=dyn)


@pytest.fixture(autouse=True)
def fresh_memory():
    wp._remembered.clear()
    yield
    wp._remembered.clear()


class TestOverallFraction:
    def test_nothing_started_is_zero(self):
        assert wp.overall_fraction(PROMPT, {}) == 0

    def test_reserves_the_requested_steps_before_the_sampler_reports(self):
        # Loaders and scheduler done (3 of 6 unit nodes) but no step reported:
        # 3 / (6 + 30 reserved) — not the 50% a node count would claim.
        nodes = {"1": node(State.Finished), "2": node(State.Finished), "3": node(State.Finished)}
        assert wp.overall_fraction(PROMPT, nodes) == pytest.approx(3 / 36)

    def test_sampler_steps_dominate_once_reported(self):
        nodes = {
            "1": node(State.Finished),
            "2": node(State.Finished),
            "3": node(State.Finished),
            "4": node(State.Running, 15, 30),
        }
        # 5 unit nodes (sampler now weighs its 30 steps): (3 + 15) / (5 + 30)
        assert wp.overall_fraction(PROMPT, nodes) == pytest.approx(18 / 35)

    def test_reserved_steps_are_not_counted_twice(self):
        nodes = {"4": node(State.Running, 0, 30)}
        # 30 reported, 30 anticipated: no extra reserve.
        assert wp.overall_fraction(PROMPT, nodes) == pytest.approx(0 / 35)

    def test_finished_nodes_count_fully_even_if_value_lags(self):
        nodes = {key: node(State.Finished) for key in PROMPT}
        nodes["4"] = node(State.Finished, 29, 30)
        assert wp.overall_fraction(PROMPT, nodes) == pytest.approx(1.0)

    def test_accepts_plain_string_states_and_garbage(self):
        nodes = {"1": {"state": "finished", "value": "x", "max": None}, "2": {"state": None}}
        assert wp.overall_fraction(PROMPT, nodes) == pytest.approx(1 / 36)

    def test_ephemeral_nodes_join_the_total(self):
        nodes = {key: node(State.Finished) for key in PROMPT}
        nodes["99"] = node(State.Running, 0, 1)
        assert wp.overall_fraction(PROMPT, nodes) < 1.0

    def test_bool_steps_are_ignored(self):
        assert wp.anticipated_steps({"1": {"inputs": {"steps": True}}, "2": {"inputs": {"steps": ["3", 0]}}}) == 0


class TestJobProgress:
    def test_in_progress_reports_fraction_percent_node_and_step(self):
        reg = registry("job-a", {"1": node(State.Finished), "2": node(State.Finished), "3": node(State.Finished), "4": node(State.Running, 15, 30)})
        body = wp.job_progress("job-a", "in_progress", reg)
        assert body["percent"] == 51  # floor(18/35 * 100)
        assert body["fraction"] == pytest.approx(18 / 35, abs=1e-4)
        assert body["node"] == "4"
        assert body["node_class"] == "SamplerCustomAdvanced"
        assert body["step"] == {"value": 15, "max": 30}
        assert body["queue_position"] is None

    def test_never_decreases_for_a_job(self):
        reg = registry("job-a", {"1": node(State.Finished), "2": node(State.Finished), "3": node(State.Finished), "4": node(State.Running, 20, 30)})
        first = wp.job_progress("job-a", "in_progress", reg)["fraction"]
        # A second pass re-reports a larger step total: the raw number drops.
        reg.nodes["4"] = node(State.Running, 2, 60)
        assert wp.overall_fraction(PROMPT, reg.nodes) < first
        assert wp.job_progress("job-a", "in_progress", reg)["fraction"] == first

    def test_stays_below_100_until_completed(self):
        reg = registry("job-a", {key: node(State.Finished) for key in PROMPT})
        body = wp.job_progress("job-a", "in_progress", reg)
        assert body["percent"] == 99
        assert body["fraction"] == pytest.approx(0.99)
        done = wp.job_progress("job-a", "completed", reg)
        assert (done["percent"], done["fraction"], done["step"]) == (100, 1.0, None)
        assert "job-a" not in wp._remembered

    def test_registry_for_another_prompt_reads_as_not_started(self):
        reg = registry("someone-else", {key: node(State.Finished) for key in PROMPT})
        body = wp.job_progress("job-a", "in_progress", reg)
        assert (body["percent"], body["node"], body["step"]) == (0, None, None)

    def test_missing_registry_is_tolerated(self):
        assert wp.job_progress("job-a", "in_progress", None)["percent"] == 0

    def test_pending_reports_its_place_in_the_queue(self):
        queued = [(7, "job-c", {}, {}, []), (5, "job-b", {}, {}, []), (9, "job-d", {}, {}, [])]
        body = wp.job_progress("job-c", "pending", None, queued)
        assert (body["percent"], body["queue_position"]) == (0, 2)
        assert wp.job_progress("job-x", "pending", None, queued)["queue_position"] is None

    def test_failed_and_cancelled_have_no_progress_and_are_forgotten(self):
        reg = registry("job-a", {"4": node(State.Running, 10, 30)})
        wp.job_progress("job-a", "in_progress", reg)
        assert wp.job_progress("job-a", "failed", reg) is None
        assert "job-a" not in wp._remembered
        assert wp.job_progress("job-a", "cancelled", reg) is None

    def test_memory_is_bounded(self):
        for index in range(wp._MAX_REMEMBERED + 10):
            wp.job_progress(f"job-{index}", "pending", None, [])
        assert len(wp._remembered) == wp._MAX_REMEMBERED
        assert "job-0" not in wp._remembered

    def test_snapshot_retries_a_registry_mutated_mid_copy(self):
        class Flaky(dict):
            calls = 0

            def items(self):
                Flaky.calls += 1
                if Flaky.calls == 1:
                    raise RuntimeError("dictionary changed size during iteration")
                return super().items()

        reg = registry("job-a", Flaky({"4": node(State.Running, 15, 30)}))
        assert wp.job_progress("job-a", "in_progress", reg)["step"] == {"value": 15, "max": 30}


def test_a_finished_graph_holds_no_reserve_even_if_nothing_reported_steps():
    nodes = {key: node(State.Finished) for key in PROMPT}
    assert wp.overall_fraction(PROMPT, nodes) == pytest.approx(1.0)
