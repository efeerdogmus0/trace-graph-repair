"""Interrupted-and-resumed training must end bit-identical to an uninterrupted run."""
import pytest

torch = pytest.importorskip("torch")
from trace_vr import resume  # noqa: E402
from trace_vr.rng import Rng  # noqa: E402


def _setup(seed=7):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    model[0].bias.requires_grad_(False)  # frozen parameter, like the base weights
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-2, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: max(0.0, 1 - u / 20))
    return model, params, opt, sched


def _train(run_dir, stop_after=None, ckpt_every=0, epochs=3, accum=2):
    """Same loop shape as scripts/train_arm.py. Returns final trainable state or None if stopped."""
    gen = torch.Generator().manual_seed(0)
    rows = [(torch.randn(4, generator=gen), torch.randn(1, generator=gen)) for _ in range(9)]
    model, params, opt, sched = _setup()
    key = {"epochs": epochs, "accum": accum, "samples": len(rows)}
    restored = resume.load(run_dir, model, opt, sched, key)
    seen, log = (restored["seen"], restored["log"]) if restored else (0, [])
    skip, order = seen, Rng("train-order", 1, "T")
    for _ in range(epochs):
        for x, y in order.shuffled(rows):
            if skip:
                skip -= 1
                continue
            loss = ((model(x) - y) ** 2).mean()
            (loss / accum).backward()
            seen += 1
            if seen % accum == 0 or seen == len(rows) * epochs:
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            log.append({"sample": seen, "loss": float(loss.detach())})
            if ckpt_every and seen % ckpt_every == 0:
                resume.save(run_dir, model, opt, sched, seen, log, key)
            if stop_after is not None and seen == stop_after:
                return None
    return resume.trainable_state(model), log


def test_resume_matches_uninterrupted_run(tmp_path):
    ref_state, ref_log = _train(tmp_path / "ref")
    run = tmp_path / "run"
    assert _train(run, stop_after=13, ckpt_every=4) is None  # dies after sample 13; last ckpt at 12
    assert (run / resume.CKPT / "meta.json").exists() and not (run / (resume.CKPT + ".tmp")).exists()
    state, log = _train(run, ckpt_every=4)
    assert set(state) == set(ref_state) and "0.bias" not in state
    for n in ref_state:
        assert torch.equal(state[n], ref_state[n]), n
    assert [r["sample"] for r in log] == list(range(1, 28))
    assert [r["loss"] for r in log] == [r["loss"] for r in ref_log]


def test_resume_refuses_other_configuration(tmp_path):
    _train(tmp_path, stop_after=4, ckpt_every=4)
    model, _, opt, sched = _setup()
    with pytest.raises(ValueError):
        resume.load(tmp_path, model, opt, sched, {"epochs": 5, "accum": 2, "samples": 9})
