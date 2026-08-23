import pytest
import torch

from model import Cache, RotaryEmbedding


def test_cache_snapshot_restore_round_trip():
    cache = Cache(batch_size=1, n_ctx=4, n_kv_heads=2, d_head=2, device=torch.device("cpu"))
    k = torch.arange(4, dtype=torch.bfloat16).reshape(1, 1, 2, 2)
    v = k + 10
    cache.extend(k, v)
    snapshot = cache.snapshot()

    cache.extend(k + 20, v + 20)
    assert cache.offset.item() == 2
    cache.restore(snapshot)

    assert cache.offset.item() == 1
    torch.testing.assert_close(cache.k[:, :1], k)
    torch.testing.assert_close(cache.v[:, :1], v)


def test_rotary_rejects_positions_past_configured_ceiling():
    rotary = RotaryEmbedding(
        head_dim=4,
        base=10_000,
        dtype=torch.float32,
        initial_context_length=8,
        max_content_length=8,
        scaling_factor=1.0,
        device=torch.device("cpu"),
    )
    query = torch.zeros((1, 2, 1, 4))
    key = torch.zeros((1, 2, 1, 4))

    with pytest.raises(ValueError, match="never wrapped"):
        rotary(query, key, offset=torch.tensor([7]))


def test_grouped_moe_dispatch_matches_gather_reference():
    from model import MLPBlock, ModelConfigs

    torch.manual_seed(3)
    config = ModelConfigs(
        num_experts=4,
        experts_per_token=2,
        hidden_size=8,
        intermediate_size=4,
        moe_dispatch="gather",
    )
    block = MLPBlock(config, device=torch.device("cpu"))
    for parameter in block.parameters():
        torch.nn.init.normal_(parameter, mean=0.0, std=0.05)
    x = torch.randn((2, 3, 8), dtype=torch.bfloat16)

    gather = block(x, dispatch="gather")
    grouped = block(x, dispatch="grouped")
    torch.testing.assert_close(grouped, gather, rtol=0.02, atol=0.02)


def test_router_telemetry_is_opt_in_and_normalized():
    from model import MLPBlock, ModelConfigs

    torch.manual_seed(9)
    config = ModelConfigs(num_experts=4, experts_per_token=2, hidden_size=8, intermediate_size=4)
    block = MLPBlock(config, device=torch.device("cpu"))
    for parameter in block.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    block.set_router_stats(True)
    block(torch.randn((2, 3, 8), dtype=torch.bfloat16))

    stats = block.router_stats
    assert stats["routes"] == 2 * 3 * 2
    assert sum(stats["counts"]) == stats["routes"]
    assert sum(stats["utilization"]) == pytest.approx(1.0)
    assert stats["max_to_mean_load"] >= 1.0
