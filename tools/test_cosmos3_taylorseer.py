import inspect
from types import SimpleNamespace

import torch

from diffusers import Cosmos3OmniTaylorSeerPipeline, Cosmos3OmniTaylorSeerTransformer
from diffusers.models.transformers.transformer_cosmos3_taylorseer import (
    Cosmos3TaylorSeerLayerState,
    Cosmos3TaylorSeerVLTextMoTDecoderLayer,
)
from tools import diffusers_taylorseer_t2v_benchmark as taylorseer_benchmark


def tiny_transformer(num_layers: int = 4) -> Cosmos3OmniTaylorSeerTransformer:
    return Cosmos3OmniTaylorSeerTransformer(
        hidden_size=8,
        intermediate_size=16,
        head_dim=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=num_layers,
        patch_latent_dim=4,
        vocab_size=32,
        rope_scaling={"mrope_section": [2, 1, 1]},
        latent_channel=1,
    )


def full_steps(
    model: Cosmos3OmniTaylorSeerTransformer,
    *,
    num_steps: int,
    layer_index: int | None = None,
) -> list[int]:
    entry = Cosmos3TaylorSeerLayerState()
    result = []
    for step in range(num_steps):
        if model._taylorseer_should_full(entry, step, num_steps, layer_index=layer_index):
            result.append(step)
            sample = torch.tensor(float(step))
            model._taylorseer_update_factors(entry, sample, step)
    return result


def cal_types(
    model: Cosmos3OmniTaylorSeerTransformer,
    *,
    num_steps: int,
    layer_index: int | None = None,
) -> list[str]:
    entry = Cosmos3TaylorSeerLayerState()
    result = []
    for step in range(num_steps):
        step_type = model._taylorseer_cal_type(entry, step, num_steps, layer_index=layer_index)
        result.append(step_type)
        if step_type == "full":
            sample = torch.tensor(float(step))
            model._taylorseer_update_factors(entry, sample, step, num_steps)
    return result


def benchmark_args(**overrides):
    args = SimpleNamespace(
        taylorseer_interval=5,
        taylorseer_fresh_threshold=None,
        taylorseer_force_scheduler=False,
        taylorseer_layer_indices=None,
        taylorseer_first_enhance=1,
        taylorseer_last_enhance=1,
        taylorseer_force_final_full=True,
        taylorseer_max_order=1,
        taylorseer_branches="both",
        taylorseer_delta_change_threshold=None,
        taylorseer_prediction_target="layer_delta",
        taylorseer_cache_und=True,
        taylorseer_stagger_layers=False,
        taylorseer_slope_scale=1.0,
        taylorseer_cache_max_gib=64.0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class DummyTaylorSeerTransformer:
    def __init__(self):
        self._taylorseer_config = {"sentinel": True}
        self.enable_kwargs = None

    def enable_taylorseer(self, **kwargs):
        self.enable_kwargs = kwargs

    def is_taylorseer_enabled(self):
        return True

    def clear_taylorseer_context(self):
        pass


def test_taylorseer_schedule_defaults_and_final_full():
    model = tiny_transformer()
    model.enable_taylorseer()
    assert full_steps(model, num_steps=35) == [0, 5, 10, 15, 20, 25, 30, 34]
    assert cal_types(model, num_steps=7) == ["full", "Taylor", "Taylor", "Taylor", "Taylor", "full", "full"]
    assert model._taylorseer_config.stagger_layers is False

    model.enable_taylorseer(interval=5, max_order=1, first_enhance=1, force_final_full=False)
    assert full_steps(model, num_steps=10) == [0, 5]

    model.enable_taylorseer(interval=5, max_order=1, first_enhance=1, last_enhance=0, force_final_full=True)
    assert full_steps(model, num_steps=10) == [0, 5, 9]

    model.enable_taylorseer(interval=5, max_order=1, first_enhance=1, last_enhance=3, force_final_full=True)
    assert full_steps(model, num_steps=10) == [0, 5, 7, 8, 9]

    model.enable_taylorseer(interval=5, max_order=1, first_enhance=1, last_enhance=0, force_final_full=True)
    assert full_steps(model, num_steps=10) == [0, 5, 9]

    model.enable_taylorseer(interval=1, max_order=1, first_enhance=1, force_final_full=True)
    assert full_steps(model, num_steps=10) == list(range(10))


def test_taylorseer_fresh_threshold_controls_cal_type_cadence():
    model = tiny_transformer()
    model.enable_taylorseer(
        fresh_threshold=3,
        first_enhance=1,
        last_enhance=0,
        force_final_full=False,
    )

    assert cal_types(model, num_steps=8) == [
        "full",
        "Taylor",
        "Taylor",
        "full",
        "Taylor",
        "Taylor",
        "full",
        "Taylor",
    ]
    assert full_steps(model, num_steps=8) == [0, 3, 6]

    model.begin_taylorseer_run(num_steps=8, do_classifier_free_guidance=False)
    model.finish_taylorseer_run()
    stats = model.get_taylorseer_stats()
    assert stats["fresh_threshold"] == 3
    assert stats["force_scheduler"] is False
    assert stats["stagger_layers"] is False


def test_taylorseer_force_scheduler_sets_cal_threshold_for_cal_type():
    model = tiny_transformer()
    model.enable_taylorseer(
        fresh_threshold=5,
        force_scheduler=True,
        first_enhance=1,
        last_enhance=0,
        force_final_full=False,
    )
    entry = Cosmos3TaylorSeerLayerState()

    assert model._taylorseer_cal_type(entry, 0, 7) == "full"
    model._taylorseer_update_factors(entry, torch.tensor(0.0), 0, 7)
    assert entry.cal_threshold == 5
    assert [model._taylorseer_cal_type(entry, step, 7) for step in range(1, 5)] == ["Taylor"] * 4
    assert model._taylorseer_cal_type(entry, 5, 7) == "full"

    model.begin_taylorseer_run(num_steps=7, do_classifier_free_guidance=False)
    model.finish_taylorseer_run()
    stats = model.get_taylorseer_stats()
    assert stats["fresh_threshold"] == 5
    assert stats["force_scheduler"] is True


def test_pipeline_call_exposes_taylorseer_scheduler_cadence_options():
    parameters = inspect.signature(Cosmos3OmniTaylorSeerPipeline.__call__).parameters
    assert parameters["taylorseer_fresh_threshold"].default is None
    assert parameters["taylorseer_force_scheduler"].default is False

    pipe = object.__new__(Cosmos3OmniTaylorSeerPipeline)
    pipe.transformer = DummyTaylorSeerTransformer()
    try:
        pipe(
            prompt="test",
            enable_taylorseer=True,
            enable_sound=True,
            taylorseer_fresh_threshold=3,
            taylorseer_force_scheduler=True,
            taylorseer_prediction_target="gen_component_delta",
        )
    except ValueError as exc:
        assert "text-to-video" in str(exc)
    else:
        raise AssertionError("TaylorSeer sound guard should stop the lightweight pipeline call")

    assert pipe.transformer.enable_kwargs["fresh_threshold"] == 3
    assert pipe.transformer.enable_kwargs["force_scheduler"] is True
    assert pipe.transformer.enable_kwargs["prediction_target"] == "gen_component_delta"


def test_benchmark_config_propagates_scheduler_cadence_options():
    args = benchmark_args(taylorseer_fresh_threshold=3, taylorseer_force_scheduler=True)
    config = taylorseer_benchmark.build_taylorseer_config(args)
    assert config["fresh_threshold"] == 3
    assert config["force_scheduler"] is True
    assert "_ft3" in config["label"]
    assert "_fs" in config["label"]

    call_kwargs = taylorseer_benchmark.taylorseer_call_kwargs(config)
    assert call_kwargs["fresh_threshold"] == 3
    assert call_kwargs["force_scheduler"] is True


def test_benchmark_stats_validation_checks_scheduler_cadence_options():
    args = benchmark_args(taylorseer_fresh_threshold=3, taylorseer_force_scheduler=True)
    config = taylorseer_benchmark.build_taylorseer_config(args)
    stats = {
        "enabled": True,
        "interval": config["interval"],
        "fresh_threshold": config["fresh_threshold"],
        "force_scheduler": config["force_scheduler"],
        "max_order": config["max_order"],
        "first_enhance": config["first_enhance"],
        "last_enhance": config["last_enhance"],
        "force_final_full": config["force_final_full"],
        "branches": config["branches"],
        "delta_change_threshold": config["delta_change_threshold"],
        "prediction_target": config["prediction_target"],
        "cache_und": config["cache_und"],
        "stagger_layers": config["stagger_layers"],
        "slope_scale": config["slope_scale"],
        "selected_layers": [0],
        "predicted_layer_calls_by_branch": {"cond": 1},
    }
    taylorseer_benchmark.validate_taylorseer_stats(config, stats, num_steps=8)

    stats["force_scheduler"] = False
    try:
        taylorseer_benchmark.validate_taylorseer_stats(config, stats, num_steps=8)
    except RuntimeError as exc:
        assert "force_scheduler" in str(exc)
    else:
        raise AssertionError("benchmark stats validation must reject force_scheduler mismatches")


def test_benchmark_rejects_invalid_fresh_threshold_flag():
    old_argv = taylorseer_benchmark.sys.argv
    taylorseer_benchmark.sys.argv = ["benchmark", "--taylorseer-fresh-threshold", "0"]
    try:
        try:
            taylorseer_benchmark.parse_args()
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("invalid fresh threshold flag must exit with parser error")
    finally:
        taylorseer_benchmark.sys.argv = old_argv


def test_benchmark_accepts_gen_component_delta_prediction_target_flag():
    old_argv = taylorseer_benchmark.sys.argv
    taylorseer_benchmark.sys.argv = [
        "benchmark",
        "--taylorseer-prediction-target",
        "gen_component_delta",
    ]
    try:
        args = taylorseer_benchmark.parse_args()
    finally:
        taylorseer_benchmark.sys.argv = old_argv

    assert args.taylorseer_prediction_target == "gen_component_delta"


def test_taylorseer_default_selects_all_layers():
    model = tiny_transformer(num_layers=6)
    model.enable_taylorseer()
    assert model._taylorseer_selected_layers() == (0, 1, 2, 3, 4, 5)


def test_taylorseer_stagger_layers_offsets_refresh_phases():
    model = tiny_transformer(num_layers=4)
    model.enable_taylorseer(
        interval=3,
        max_order=0,
        first_enhance=1,
        last_enhance=0,
        force_final_full=False,
        stagger_layers=True,
    )

    assert full_steps(model, num_steps=7, layer_index=0) == [0, 3, 6]
    assert full_steps(model, num_steps=7, layer_index=1) == [0, 1, 4]
    assert full_steps(model, num_steps=7, layer_index=2) == [0, 2, 5]

    model.begin_taylorseer_run(num_steps=7, do_classifier_free_guidance=False)
    model.finish_taylorseer_run()
    assert model.get_taylorseer_stats()["stagger_layers"] is True


def test_taylorseer_first_order_prediction_uses_per_step_slope():
    model = tiny_transformer()
    model.enable_taylorseer(interval=5, max_order=1, first_enhance=1, force_final_full=False)
    entry = Cosmos3TaylorSeerLayerState()
    model._taylorseer_update_factors(entry, torch.tensor(10.0), 0)
    model._taylorseer_update_factors(entry, torch.tensor(20.0), 5)
    prediction = model._taylorseer_predict(entry, 6)
    assert torch.equal(prediction, torch.tensor(22.0))

    model.enable_taylorseer(interval=5, max_order=1, slope_scale=0.5)
    entry = Cosmos3TaylorSeerLayerState()
    model._taylorseer_update_factors(entry, torch.tensor(10.0), 0)
    model._taylorseer_update_factors(entry, torch.tensor(20.0), 5)
    prediction = model._taylorseer_predict(entry, 6)
    assert torch.equal(prediction, torch.tensor(21.0))
    model.begin_taylorseer_run(num_steps=10, do_classifier_free_guidance=False)
    model.finish_taylorseer_run()
    assert model.get_taylorseer_stats()["slope_scale"] == 0.5


def test_taylorseer_factors_preserve_model_dtype():
    model = tiny_transformer()
    model.enable_taylorseer(interval=2, max_order=1)
    entry = Cosmos3TaylorSeerLayerState()
    model._taylorseer_update_factors(entry, torch.tensor(1.0, dtype=torch.bfloat16), 0)
    model._taylorseer_update_factors(entry, torch.tensor(3.0, dtype=torch.bfloat16), 2)
    assert entry.factors[0].dtype == torch.bfloat16
    assert entry.factors[1].dtype == torch.bfloat16
    assert torch.equal(entry.factors[1], torch.tensor(1.0, dtype=torch.bfloat16))


def test_taylorseer_delta_change_guard_requires_stable_full_steps():
    model = tiny_transformer()
    model.enable_taylorseer(
        interval=2,
        first_enhance=0,
        last_enhance=0,
        force_final_full=False,
        delta_change_threshold=0.1,
    )
    entry = Cosmos3TaylorSeerLayerState()
    model._taylorseer_update_factors(entry, torch.ones(2), 0)
    assert not entry.prediction_allowed
    assert model._taylorseer_should_full(entry, 1, 10)

    model._taylorseer_update_factors(entry, torch.full((2,), 2.0), 1)
    assert entry.last_delta_change_ratio > 0.1
    assert not entry.prediction_allowed
    assert model._taylorseer_should_full(entry, 2, 10)

    model._taylorseer_update_factors(entry, torch.full((2,), 2.01), 2)
    assert entry.last_delta_change_ratio < 0.1
    assert entry.prediction_allowed
    assert not model._taylorseer_should_full(entry, 3, 10)


def test_taylorseer_branch_states_are_independent():
    model = tiny_transformer(num_layers=2)
    model.enable_taylorseer(interval=5, max_order=1, layer_indices=(0,))
    model.begin_taylorseer_run(num_steps=10, do_classifier_free_guidance=True)
    model._taylorseer_branch_states["cond"].layers[0] = Cosmos3TaylorSeerLayerState()
    model._taylorseer_update_factors(model._taylorseer_branch_states["cond"].layers[0], torch.tensor(1.0), 0)
    assert 0 not in model._taylorseer_branch_states["uncond"].layers



def test_taylorseer_can_target_single_cfg_branch():
    model = tiny_transformer(num_layers=2)
    model.enable_taylorseer(interval=2, branches="uncond", layer_indices=(0,))
    model.begin_taylorseer_run(num_steps=10, do_classifier_free_guidance=True)
    assert model._taylorseer_branch_count == 1
    assert not model._taylorseer_branch_enabled("cond")
    assert model._taylorseer_branch_enabled("uncond")
    model.finish_taylorseer_run()
    assert model.get_taylorseer_stats()["branches"] == "uncond"


def test_taylorseer_can_target_attention_delta_prediction():
    model = tiny_transformer(num_layers=2)
    model.enable_taylorseer(interval=2, prediction_target="attention_delta", layer_indices=(0,))
    model.begin_taylorseer_run(num_steps=10, do_classifier_free_guidance=False)
    model.finish_taylorseer_run()
    assert model.get_taylorseer_stats()["prediction_target"] == "attention_delta"


def test_taylorseer_can_target_gen_component_delta_prediction():
    model = tiny_transformer(num_layers=2)
    model.enable_taylorseer(interval=2, prediction_target="gen_component_delta", layer_indices=(0,))
    model.begin_taylorseer_run(num_steps=10, do_classifier_free_guidance=False)
    model.finish_taylorseer_run()
    assert model.get_taylorseer_stats()["prediction_target"] == "gen_component_delta"


def test_attention_delta_prediction_replays_exact_gen_mlp_on_current_input():
    model = tiny_transformer(num_layers=1)
    model.enable_taylorseer(
        interval=2,
        first_enhance=0,
        last_enhance=0,
        force_final_full=False,
        layer_indices=(0,),
        prediction_target="attention_delta",
    )
    model.begin_taylorseer_run(num_steps=4, do_classifier_free_guidance=False)
    rotary_emb = (
        torch.ones(3, 4),
        torch.zeros(3, 4),
        torch.ones(5, 4),
        torch.zeros(5, 4),
    )
    position_ids = torch.zeros(3, 8, dtype=torch.long)
    und_seq = torch.randn(3, 8)
    gen_seq = torch.randn(5, 8)

    model.set_taylorseer_context(branch="cond", step_index=0, timestep=0)
    model._taylorseer_forward_layers(
        und_seq=und_seq,
        gen_seq=gen_seq,
        rotary_emb=rotary_emb,
        sequence_length=8,
        und_len=3,
        position_ids=position_ids,
    )
    entry = model._taylorseer_branch_states["cond"].layers[0]
    stored_attn_delta = entry.factors[0].to(dtype=gen_seq.dtype)

    next_gen_seq = torch.randn(5, 8)
    expected_residual = next_gen_seq + stored_attn_delta
    expected_gen = expected_residual + model.layers[0].mlp_moe_gen(
        model.layers[0].post_attention_layernorm_moe_gen(expected_residual)
    )
    expected_und = entry.und_after

    model.set_taylorseer_context(branch="cond", step_index=1, timestep=1)
    und_out, gen_out = model._taylorseer_forward_layers(
        und_seq=torch.randn(3, 8),
        gen_seq=next_gen_seq,
        rotary_emb=rotary_emb,
        sequence_length=8,
        und_len=3,
        position_ids=position_ids,
    )
    assert torch.equal(und_out, expected_und)
    assert torch.equal(gen_out, expected_gen)
    assert entry.predicted_count == 1


def test_mlp_delta_prediction_replays_exact_attention_and_und_mlp():
    model = tiny_transformer(num_layers=1)
    model.enable_taylorseer(
        interval=2,
        first_enhance=0,
        last_enhance=0,
        force_final_full=False,
        layer_indices=(0,),
        prediction_target="mlp_delta",
    )
    model.begin_taylorseer_run(num_steps=4, do_classifier_free_guidance=False)
    rotary_emb = (
        torch.ones(3, 4),
        torch.zeros(3, 4),
        torch.ones(5, 4),
        torch.zeros(5, 4),
    )
    position_ids = torch.zeros(3, 8, dtype=torch.long)
    model.set_taylorseer_context(branch="cond", step_index=0, timestep=0)
    model._taylorseer_forward_layers(
        und_seq=torch.randn(3, 8),
        gen_seq=torch.randn(5, 8),
        rotary_emb=rotary_emb,
        sequence_length=8,
        und_len=3,
        position_ids=position_ids,
    )
    entry = model._taylorseer_branch_states["cond"].layers[0]
    stored_mlp_delta = entry.factors[0].to(dtype=torch.float32)
    assert entry.und_after is None

    next_und_seq = torch.randn(3, 8)
    next_gen_seq = torch.randn(5, 8)
    expected_und, expected_gen = model.layers[0].forward_with_predicted_gen_mlp_delta(
        next_und_seq,
        next_gen_seq,
        rotary_emb,
        stored_mlp_delta,
    )
    model.set_taylorseer_context(branch="cond", step_index=1, timestep=1)
    und_out, gen_out = model._taylorseer_forward_layers(
        und_seq=next_und_seq,
        gen_seq=next_gen_seq,
        rotary_emb=rotary_emb,
        sequence_length=8,
        und_len=3,
        position_ids=position_ids,
    )
    assert torch.equal(und_out, expected_und)
    assert torch.equal(gen_out, expected_gen)
    assert entry.predicted_count == 1
    model.finish_taylorseer_run()
    assert model.get_taylorseer_stats()["prediction_target"] == "mlp_delta"


def test_gen_component_delta_full_step_stores_separate_gen_component_factors():
    model = tiny_transformer(num_layers=1)
    model.enable_taylorseer(
        interval=2,
        first_enhance=0,
        last_enhance=0,
        force_final_full=False,
        layer_indices=(0,),
        prediction_target="gen_component_delta",
    )
    model.begin_taylorseer_run(num_steps=4, do_classifier_free_guidance=False)
    rotary_emb = (
        torch.ones(3, 4),
        torch.zeros(3, 4),
        torch.ones(5, 4),
        torch.zeros(5, 4),
    )
    position_ids = torch.zeros(3, 8, dtype=torch.long)

    model.set_taylorseer_context(branch="cond", step_index=0, timestep=0)
    model._taylorseer_forward_layers(
        und_seq=torch.randn(3, 8),
        gen_seq=torch.randn(5, 8),
        rotary_emb=rotary_emb,
        sequence_length=8,
        und_len=3,
        position_ids=position_ids,
    )
    entry = model._taylorseer_branch_states["cond"].layers[0]

    assert len(entry.gen_attn_factors) == 1
    assert len(entry.gen_mlp_factors) == 1
    assert entry.gen_attn_factors[0].shape == (5, 8)
    assert entry.gen_mlp_factors[0].shape == (5, 8)
    assert entry.factors == []


def test_gen_component_delta_component_factors_use_separate_slopes_and_refresh_on_threshold():
    model = tiny_transformer(num_layers=1)
    model.enable_taylorseer(
        interval=2,
        max_order=1,
        slope_scale=0.5,
        delta_change_threshold=0.1,
        first_enhance=0,
        last_enhance=0,
        force_final_full=False,
        layer_indices=(0,),
        prediction_target="gen_component_delta",
    )
    entry = Cosmos3TaylorSeerLayerState()

    first_attn_delta = torch.tensor([2.0, 4.0])
    first_mlp_delta = torch.tensor([6.0, 8.0])
    model._taylorseer_update_gen_component_factors(entry, first_attn_delta, first_mlp_delta, 0, 10)
    assert torch.equal(entry.gen_attn_factors[0], first_attn_delta)
    assert torch.equal(entry.gen_mlp_factors[0], first_mlp_delta)
    assert entry.prediction_allowed is False

    second_attn_delta = torch.tensor([10.0, 14.0])
    second_mlp_delta = torch.tensor([20.0, 28.0])
    model._taylorseer_update_gen_component_factors(entry, second_attn_delta, second_mlp_delta, 2, 10)

    expected_attn_slope = (second_attn_delta - first_attn_delta) / 2
    expected_mlp_slope = (second_mlp_delta - first_mlp_delta) / 2
    assert torch.equal(entry.gen_attn_factors[0], second_attn_delta)
    assert torch.equal(entry.gen_attn_factors[1], expected_attn_slope)
    assert torch.equal(entry.gen_mlp_factors[0], second_mlp_delta)
    assert torch.equal(entry.gen_mlp_factors[1], expected_mlp_slope)

    predicted_attn_delta = model._taylorseer_predict_from_factors(entry.gen_attn_factors, entry.last_full_step, 3)
    predicted_mlp_delta = model._taylorseer_predict_from_factors(entry.gen_mlp_factors, entry.last_full_step, 3)
    assert torch.equal(predicted_attn_delta, second_attn_delta + expected_attn_slope * 0.5)
    assert torch.equal(predicted_mlp_delta, second_mlp_delta + expected_mlp_slope * 0.5)
    assert entry.last_delta_change_ratio is not None and entry.last_delta_change_ratio > 0.1
    assert entry.prediction_allowed is False
    assert model._taylorseer_should_full(entry, 3, 10)


def test_gen_component_delta_prediction_reconstructs_cached_und_and_gen_components():
    model = tiny_transformer(num_layers=1)
    model.enable_taylorseer(
        interval=2,
        max_order=0,
        first_enhance=0,
        last_enhance=0,
        force_final_full=False,
        layer_indices=(0,),
        prediction_target="gen_component_delta",
    )
    model.begin_taylorseer_run(num_steps=4, do_classifier_free_guidance=False)
    rotary_emb = (
        torch.ones(3, 4),
        torch.zeros(3, 4),
        torch.ones(5, 4),
        torch.zeros(5, 4),
    )
    position_ids = torch.zeros(3, 8, dtype=torch.long)

    model.set_taylorseer_context(branch="cond", step_index=0, timestep=0)
    model._taylorseer_forward_layers(
        und_seq=torch.randn(3, 8),
        gen_seq=torch.randn(5, 8),
        rotary_emb=rotary_emb,
        sequence_length=8,
        und_len=3,
        position_ids=position_ids,
    )
    entry = model._taylorseer_branch_states["cond"].layers[0]
    stored_attn_delta = entry.gen_attn_factors[0].to(dtype=torch.float32)
    stored_mlp_delta = entry.gen_mlp_factors[0].to(dtype=torch.float32)
    assert entry.und_after is not None

    next_gen_seq = torch.randn(5, 8)
    expected_gen = next_gen_seq + stored_attn_delta + stored_mlp_delta
    expected_und = entry.und_after

    model.set_taylorseer_context(branch="cond", step_index=1, timestep=1)
    und_out, gen_out = model._taylorseer_forward_layers(
        und_seq=torch.randn(3, 8),
        gen_seq=next_gen_seq,
        rotary_emb=rotary_emb,
        sequence_length=8,
        und_len=3,
        position_ids=position_ids,
    )
    assert torch.equal(und_out, expected_und)
    assert torch.equal(gen_out, expected_gen)
    assert entry.predicted_count == 1


def test_und_cache_prediction_replays_exact_gen_path_with_static_und():
    model = tiny_transformer(num_layers=1)
    model.enable_taylorseer(
        interval=1,
        first_enhance=0,
        last_enhance=0,
        force_final_full=False,
        layer_indices=(0,),
        prediction_target="und_cache",
    )
    model.begin_taylorseer_run(num_steps=4, do_classifier_free_guidance=False)
    rotary_emb = (
        torch.ones(3, 4),
        torch.zeros(3, 4),
        torch.ones(5, 4),
        torch.zeros(5, 4),
    )
    position_ids = torch.zeros(3, 8, dtype=torch.long)
    und_seq = torch.randn(3, 8)
    gen_seq = torch.randn(5, 8)

    model.set_taylorseer_context(branch="cond", step_index=0, timestep=0)
    model._taylorseer_forward_layers(
        und_seq=und_seq,
        gen_seq=gen_seq,
        rotary_emb=rotary_emb,
        sequence_length=8,
        und_len=3,
        position_ids=position_ids,
    )
    entry = model._taylorseer_branch_states["cond"].layers[0]
    assert entry.und_after is not None
    assert entry.und_k is not None
    assert entry.und_v is not None
    assert entry.factors == []

    next_gen_seq = torch.randn(5, 8)
    expected_gen = model.layers[0].forward_gen_with_und_kv_cache(
        next_gen_seq,
        rotary_emb,
        entry.und_k,
        entry.und_v,
    )
    model.set_taylorseer_context(branch="cond", step_index=1, timestep=1)
    und_out, gen_out = model._taylorseer_forward_layers(
        und_seq=und_seq,
        gen_seq=next_gen_seq,
        rotary_emb=rotary_emb,
        sequence_length=8,
        und_len=3,
        position_ids=position_ids,
    )
    assert torch.equal(und_out, entry.und_after)
    assert torch.equal(gen_out, expected_gen)
    assert entry.predicted_count == 1
    model.finish_taylorseer_run()
    assert model.get_taylorseer_stats()["prediction_target"] == "und_cache"


def test_und_cache_still_respects_final_full_window():
    model = tiny_transformer()
    model.enable_taylorseer(
        interval=5,
        first_enhance=1,
        last_enhance=2,
        force_final_full=True,
        prediction_target="und_cache",
    )
    entry = Cosmos3TaylorSeerLayerState()
    assert model._taylorseer_should_full(entry, 0, 5)
    entry.und_k = torch.empty(1)
    entry.und_v = torch.empty(1)
    entry.last_full_step = 0
    assert not model._taylorseer_should_full(entry, 1, 5)
    assert model._taylorseer_should_full(entry, 3, 5)
    assert model._taylorseer_should_full(entry, 4, 5)


def test_taylorseer_rejects_invalid_prediction_target():
    model = tiny_transformer(num_layers=2)
    try:
        model.enable_taylorseer(prediction_target="bad")
    except ValueError as exc:
        assert "prediction_target" in str(exc)
    else:
        raise AssertionError("invalid TaylorSeer prediction target must be rejected")


def test_taylorseer_rejects_invalid_fresh_threshold():
    model = tiny_transformer(num_layers=2)
    try:
        model.enable_taylorseer(fresh_threshold=0)
    except ValueError as exc:
        assert "fresh_threshold" in str(exc)
    else:
        raise AssertionError("invalid TaylorSeer fresh_threshold must be rejected")


def test_taylorseer_rejects_invalid_branch_selector():
    model = tiny_transformer(num_layers=2)
    try:
        model.enable_taylorseer(branches="bad")
    except ValueError as exc:
        assert "branches" in str(exc)
    else:
        raise AssertionError("invalid TaylorSeer branch selector must be rejected")


def test_taylorseer_rejects_invalid_slope_scale():
    model = tiny_transformer(num_layers=2)
    for value in (-0.1, float("nan"), float("inf")):
        try:
            model.enable_taylorseer(slope_scale=value)
        except ValueError as exc:
            assert "slope_scale" in str(exc)
        else:
            raise AssertionError(f"invalid TaylorSeer slope_scale {value!r} must be rejected")


def test_taylorseer_accepts_exact_und_layer_delta_path():
    model = tiny_transformer(num_layers=2)
    model.enable_taylorseer(interval=2, cache_und=False, layer_indices=(0,))
    model.begin_taylorseer_run(num_steps=4, do_classifier_free_guidance=False)
    model.finish_taylorseer_run()
    stats = model.get_taylorseer_stats()
    assert stats["cache_und"] is False


def test_taylorseer_rejects_uncached_und_cache_prediction():
    model = tiny_transformer(num_layers=2)
    try:
        model.enable_taylorseer(interval=2, cache_und=False, prediction_target="und_cache")
    except ValueError as exc:
        assert "cache_und=True" in str(exc)
    else:
        raise AssertionError("und_cache prediction requires cache_und=True")


def test_layer_delta_exact_und_path_recomputes_current_und():
    model = tiny_transformer(num_layers=1)
    model.enable_taylorseer(
        interval=2,
        max_order=0,
        first_enhance=0,
        last_enhance=0,
        force_final_full=False,
        layer_indices=(0,),
        cache_und=False,
    )
    model.begin_taylorseer_run(num_steps=4, do_classifier_free_guidance=False)
    rotary_emb = (
        torch.ones(3, 4),
        torch.zeros(3, 4),
        torch.ones(5, 4),
        torch.zeros(5, 4),
    )
    position_ids = torch.zeros(3, 8, dtype=torch.long)

    model.set_taylorseer_context(branch="cond", step_index=0, timestep=0)
    model._taylorseer_forward_layers(
        und_seq=torch.randn(3, 8),
        gen_seq=torch.randn(5, 8),
        rotary_emb=rotary_emb,
        sequence_length=8,
        und_len=3,
        position_ids=position_ids,
    )
    entry = model._taylorseer_branch_states["cond"].layers[0]
    stored_gen_delta = entry.factors[0].to(dtype=torch.float32)
    assert entry.und_after is None

    next_und_seq = torch.randn(3, 8)
    next_gen_seq = torch.randn(5, 8)
    expected_und = model.layers[0].forward_und_only(next_und_seq, rotary_emb)
    expected_gen = next_gen_seq + stored_gen_delta

    model.set_taylorseer_context(branch="cond", step_index=1, timestep=1)
    und_out, gen_out = model._taylorseer_forward_layers(
        und_seq=next_und_seq,
        gen_seq=next_gen_seq,
        rotary_emb=rotary_emb,
        sequence_length=8,
        und_len=3,
        position_ids=position_ids,
    )
    assert torch.equal(und_out, expected_und)
    assert torch.equal(gen_out, expected_gen)
    assert entry.predicted_count == 1
    model.finish_taylorseer_run()
    assert model.get_taylorseer_stats()["cache_und"] is False


def test_attention_delta_exact_und_path_recomputes_current_und_and_gen_mlp():
    model = tiny_transformer(num_layers=1)
    model.enable_taylorseer(
        interval=2,
        max_order=0,
        first_enhance=0,
        last_enhance=0,
        force_final_full=False,
        layer_indices=(0,),
        prediction_target="attention_delta",
        cache_und=False,
    )
    model.begin_taylorseer_run(num_steps=4, do_classifier_free_guidance=False)
    rotary_emb = (
        torch.ones(3, 4),
        torch.zeros(3, 4),
        torch.ones(5, 4),
        torch.zeros(5, 4),
    )
    position_ids = torch.zeros(3, 8, dtype=torch.long)

    model.set_taylorseer_context(branch="cond", step_index=0, timestep=0)
    model._taylorseer_forward_layers(
        und_seq=torch.randn(3, 8),
        gen_seq=torch.randn(5, 8),
        rotary_emb=rotary_emb,
        sequence_length=8,
        und_len=3,
        position_ids=position_ids,
    )
    entry = model._taylorseer_branch_states["cond"].layers[0]
    stored_attn_delta = entry.factors[0].to(dtype=torch.float32)
    assert entry.und_after is None

    next_und_seq = torch.randn(3, 8)
    next_gen_seq = torch.randn(5, 8)
    expected_und = model.layers[0].forward_und_only(next_und_seq, rotary_emb)
    expected_residual = next_gen_seq + stored_attn_delta
    expected_gen = expected_residual + model.layers[0].mlp_moe_gen(
        model.layers[0].post_attention_layernorm_moe_gen(expected_residual)
    )

    model.set_taylorseer_context(branch="cond", step_index=1, timestep=1)
    und_out, gen_out = model._taylorseer_forward_layers(
        und_seq=next_und_seq,
        gen_seq=next_gen_seq,
        rotary_emb=rotary_emb,
        sequence_length=8,
        und_len=3,
        position_ids=position_ids,
    )
    assert torch.equal(und_out, expected_und)
    assert torch.equal(gen_out, expected_gen)
    assert entry.predicted_count == 1


def test_gen_component_delta_exact_und_path_recomputes_current_und_and_gen_components():
    model = tiny_transformer(num_layers=1)
    model.enable_taylorseer(
        interval=2,
        max_order=0,
        first_enhance=0,
        last_enhance=0,
        force_final_full=False,
        layer_indices=(0,),
        prediction_target="gen_component_delta",
        cache_und=False,
    )
    model.begin_taylorseer_run(num_steps=4, do_classifier_free_guidance=False)
    rotary_emb = (
        torch.ones(3, 4),
        torch.zeros(3, 4),
        torch.ones(5, 4),
        torch.zeros(5, 4),
    )
    position_ids = torch.zeros(3, 8, dtype=torch.long)

    model.set_taylorseer_context(branch="cond", step_index=0, timestep=0)
    model._taylorseer_forward_layers(
        und_seq=torch.randn(3, 8),
        gen_seq=torch.randn(5, 8),
        rotary_emb=rotary_emb,
        sequence_length=8,
        und_len=3,
        position_ids=position_ids,
    )
    entry = model._taylorseer_branch_states["cond"].layers[0]
    stored_attn_delta = entry.gen_attn_factors[0].to(dtype=torch.float32)
    stored_mlp_delta = entry.gen_mlp_factors[0].to(dtype=torch.float32)
    assert entry.und_after is None

    next_und_seq = torch.randn(3, 8)
    next_gen_seq = torch.randn(5, 8)
    expected_und = model.layers[0].forward_und_only(next_und_seq, rotary_emb)
    expected_gen = next_gen_seq + stored_attn_delta + stored_mlp_delta

    model.set_taylorseer_context(branch="cond", step_index=1, timestep=1)
    und_out, gen_out = model._taylorseer_forward_layers(
        und_seq=next_und_seq,
        gen_seq=next_gen_seq,
        rotary_emb=rotary_emb,
        sequence_length=8,
        und_len=3,
        position_ids=position_ids,
    )
    assert torch.equal(und_out, expected_und)
    assert torch.equal(gen_out, expected_gen)
    assert entry.predicted_count == 1


def test_decoder_forward_with_gen_delta_matches_exact_shape_and_delta_identity():
    layer = Cosmos3TaylorSeerVLTextMoTDecoderLayer(
        hidden_size=8,
        head_dim=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=16,
        attention_bias=False,
        rms_norm_eps=1e-6,
    )
    und_seq = torch.randn(3, 8)
    gen_seq = torch.randn(5, 8)
    rotary_emb = (
        torch.ones(3, 4),
        torch.zeros(3, 4),
        torch.ones(5, 4),
        torch.zeros(5, 4),
    )

    exact_und, exact_gen = layer(und_seq, gen_seq, rotary_emb)
    und_next, gen_next, gen_delta, gen_attn_delta, gen_mlp_delta = layer.forward_with_gen_delta(
        und_seq, gen_seq, rotary_emb
    )

    assert und_next.shape == exact_und.shape == und_seq.shape
    assert gen_next.shape == exact_gen.shape == gen_seq.shape
    assert gen_delta.shape == gen_attn_delta.shape == gen_mlp_delta.shape == gen_seq.shape
    assert torch.allclose(gen_next, gen_seq + gen_delta, atol=1e-5, rtol=1e-5)
    assert torch.equal(und_next, exact_und)
    assert torch.equal(gen_next, exact_gen)


def test_decoder_forward_und_only_matches_full_und_output():
    layer = Cosmos3TaylorSeerVLTextMoTDecoderLayer(
        hidden_size=8,
        head_dim=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=16,
        attention_bias=False,
        rms_norm_eps=1e-6,
    )
    und_seq = torch.randn(3, 8)
    gen_seq = torch.randn(5, 8)
    rotary_emb = (
        torch.ones(3, 4),
        torch.zeros(3, 4),
        torch.ones(5, 4),
        torch.zeros(5, 4),
    )

    full_und, _ = layer(und_seq, gen_seq, rotary_emb)
    und_only = layer.forward_und_only(und_seq, rotary_emb)

    assert torch.equal(und_only, full_und)


def test_public_import_surface():
    assert Cosmos3OmniTaylorSeerPipeline.__name__ == "Cosmos3OmniTaylorSeerPipeline"
    assert Cosmos3OmniTaylorSeerTransformer.__name__ == "Cosmos3OmniTaylorSeerTransformer"
