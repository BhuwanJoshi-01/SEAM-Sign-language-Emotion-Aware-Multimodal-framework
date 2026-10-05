"""M4's instruments must be falsifiable before any M4 number means anything.

Three separate ways a separation claim can be hollow, and one test each:

* **The loss terms can be degenerate.** An orthogonality penalty that can be minimised
  by shrinking a branch, a gradient reversal with the wrong sign, an MI bound that
  rewards entanglement - all produce a model that trains, converges, and reports
  excellent separation.
* **The splits can leak.** A random split puts the same signer on both sides, and a
  model that identifies the signer predicts their affect from it. The overlap
  assertion has to be unskippable, not a comment.
* **The metric can be blind.** A probe that cannot detect entanglement when
  entanglement is definitely present would pass a completely entangled model. Hence
  the signer positive control, which must score ~0.5 on clean factors and >0.8 on
  factors that have memorised their signer.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from seam.affect.encoder import (
    MAX_PARAMS_PER_TRUNK,
    EntangledBaseline,
    FactorizedEncoder,
    FactorizerConfig,
    default_lambda_grid,
    factorizer_loss,
)
from seam.affect.grl import GradientReversal, grad_reverse
from seam.affect.orthogonality import (
    canonical_angles,
    orthogonality_penalty,
    shared_subspace_fraction,
)
from seam.affect.vclub import (
    held_out_joint_accuracy,
    held_out_jsd_nats,
    interpretability,
)
from seam.eval.loso import SplitOverlapError, assert_no_signer_overlap, make_loso
from seam.eval.probes import (
    SIGNER_CONTROL_MIN_AUC,
    SeparationReport,
    auc,
    cross_prediction,
    cross_prediction_multilabel,
    signer_probe_auc,
)


def _cfg(**kw) -> FactorizerConfig:
    return FactorizerConfig(
        dim_nm=20,
        dim_p=10,
        n_linguistic=4,
        n_affect=8,
        **kw,  # type: ignore[arg-type]
    )


# --- gradient reversal -----------------------------------------------------


def test_grad_reverse_negates_backward_and_passes_forward() -> None:
    x = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    y = grad_reverse(x, 3.0)
    assert torch.allclose(y, x.detach()), "forward must be the identity"
    y.sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -3.0)), "backward must negate by lambda"


def test_grad_reverse_module_lambda_is_readable() -> None:
    g = GradientReversal(0.25)
    assert g.lambda_value() == pytest.approx(0.25)
    assert "0.250" in g.extra_repr()


# --- orthogonality ---------------------------------------------------------


def test_orthogonality_penalty_orders_correctly() -> None:
    """Identical factors must score strictly worse than orthogonal ones."""
    same = torch.ones(32, 8)
    ortho = torch.eye(8).repeat(4, 1)
    p_same = float(orthogonality_penalty(same, same))
    p_ortho = float(orthogonality_penalty(ortho, ortho))
    assert p_same > p_ortho
    # Identical rows in a (B, d) batch: the cross-correlation is all ones, so the sum
    # is B^2 and the penalty is exactly B. Pinned because it makes the batch
    # dependence explicit - 32.0 here, 64.0 at batch 64 for the same factors - which
    # is precisely why `shared_subspace_fraction` is the figure that gets reported.
    assert p_same == pytest.approx(32.0, rel=1e-5)
    # eye(8).repeat(4, 1) is 32 rows over 8 distinct directions, each appearing 4
    # times. Exactly 8 * 4 * 4 = 128 of the 1024 cross-correlations are non-zero, so
    # the penalty is 128/32 = 4.0 - well below the identical case but not zero, because
    # the directions are *repeated* rather than unique.
    assert p_ortho == pytest.approx(4.0, rel=1e-5)


def test_orthogonality_penalty_is_scale_invariant() -> None:
    """Otherwise the loss is minimised by shrinking a branch.

    A branch that shrinks to nothing scores a perfect penalty while contributing
    nothing to the prediction heads - a degenerate solution that looks like ideal
    separation and destroys the model.
    """
    a, b = torch.randn(16, 6), torch.randn(16, 6)
    assert float(orthogonality_penalty(a, b)) == pytest.approx(
        float(orthogonality_penalty(a * 1000, b * 0.001)), rel=1e-5
    )


def test_shared_subspace_fraction_does_not_scale_with_batch() -> None:
    """The penalty grows with batch size; the reported figure must not.

    ``||Z_L Z_A^T||_F^2`` summed over a batch is not a readable "how much overlap"
    number: for identical factors it equals the batch size exactly, so the same
    factors score 16.0 at batch 16 and 32.0 at batch 32. The normalised mean is
    bounded and near-batch-invariant, which is why it is the one reported.
    """
    torch.manual_seed(0)
    same = torch.ones(32, 8)
    a_lo = shared_subspace_fraction(same[:16], same[16:32])
    a_hi = shared_subspace_fraction(same[:32], same)
    assert 0.0 <= a_lo <= 1.0 + 1e-6
    assert a_lo == pytest.approx(a_hi, rel=1e-5), "the fraction must not track batch size"
    assert a_hi == pytest.approx(1.0), "identical factors share all variance"

    # The penalty, by contrast, doubles exactly with the batch.
    p16 = float(orthogonality_penalty(same[:16], same[16:32]))
    p32 = float(orthogonality_penalty(same, same))
    assert p32 == pytest.approx(2 * p16, rel=1e-5)


def test_orthogonality_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="batch"):
        orthogonality_penalty(torch.randn(8, 4), torch.randn(7, 4))
    with pytest.raises(ValueError, match="2D"):
        orthogonality_penalty(torch.randn(8), torch.randn(8))


def test_canonical_angles_separate_overlap_regimes() -> None:
    ident = torch.ones(24, 6)
    assert canonical_angles(ident, ident).min() < 1.0
    a = canonical_angles(torch.randn(24, 6), torch.randn(24, 6))
    assert a.min() > 30.0, "random factors should be mostly far from shared"


# --- the MI bound ----------------------------------------------------------


def test_discriminator_is_at_chance_on_independent_factors() -> None:
    """The calibration that matters most, and the one that failed for vCLUB.

    Evaluated on *held-out* pairs. The same check on training pairs scores ~0.94 and
    that number is a memorisation artefact, which is why `held_out_joint_accuracy`
    takes fresh draws and has no access to the training set.
    """
    torch.manual_seed(0)
    d = 8
    x = torch.randn(400, d)
    y_indep = torch.randn(400, d)
    y_depend = x.clone()

    # The helpers refit and score on a held-out split internally, so the passed-in
    # discriminator is a formality; average over seeds because a single 300-sample
    # split has real variance. The bound is 0.65 rather than 0.56 because a
    # *memorising* discriminator scores 0.94 here - anything under 0.65 is already a
    # decisive separation between "at chance" and "memorised the samples".
    accs_indep = [held_out_joint_accuracy(None, x, y_indep, seed=s) for s in range(3)]
    accs_depend = [held_out_joint_accuracy(None, x, y_depend, seed=s) for s in range(3)]
    acc_indep = sum(accs_indep) / 3
    acc_depend = sum(accs_depend) / 3
    assert acc_indep < 0.65, f"independent factors scored {acc_indep:.3f}, not chance"
    assert acc_depend > 0.9, f"dependent factors scored only {acc_depend:.3f}"
    assert acc_depend > acc_indep + 0.3


def test_jsd_convention_is_stated_and_monotone() -> None:
    """Lower JSD means *more* entanglement, which is the opposite of the loss."""
    torch.manual_seed(0)
    x = torch.randn(400, 8)
    y = x.clone()
    jsd_sep = held_out_jsd_nats(None, x, torch.randn(400, 8))
    jsd_ent = held_out_jsd_nats(None, x, y)
    assert jsd_ent < jsd_sep, (
        "entangled pairs must give a *lower* JSD than independent ones; the opposite "
        "means the convention in the code contradicts the convention in the docs"
    )
    assert "chance" in interpretability(0.5)
    assert "separable" in interpretability(0.95)


# --- the encoder -----------------------------------------------------------


def test_trunks_respect_the_parameter_budget() -> None:
    """A budget nobody checks is a budget nobody meets."""
    m = FactorizedEncoder(_cfg(hidden=256))
    for name, n in m.trunk_param_counts().items():
        assert n <= MAX_PARAMS_PER_TRUNK, f"{name} trunk has {n} params"


def test_factorized_and_baseline_both_train() -> None:
    cfg = _cfg()
    nm, p = torch.randn(24, cfg.dim_nm), torch.randn(24, cfg.dim_p)
    yl = torch.randint(0, cfg.n_linguistic, (24,))
    # Affect is multi-label: a multi-hot target, not a class index. Passing a class
    # index to a sigmoid head with BCE is the same category error the single-expression
    # framing made, and it fails loudly here rather than training quietly wrong.
    ya = (torch.rand(24, cfg.n_affect) < 0.3).float()
    for model in (FactorizedEncoder(cfg), EntangledBaseline(cfg)):
        out = model(nm, p)
        total, parts = factorizer_loss(out, yl, ya, cfg)
        total.backward()
        assert torch.isfinite(total)
        # Both adversarial terms must be present: a GRL branch that is in the graph
        # but not in the loss is an untrained head, which is what the first M4 run
        # measured and mistook for adversarial defeat.
        assert {"loss_l", "loss_a", "orthogonality"} <= set(parts)
        assert {"loss_a_from_l", "loss_l_from_a"} <= set(parts), (
            "an adversarial term is missing from the objective, so that GRL head is "
            "untrained and its accuracy would be meaningless"
        )


def test_the_supervised_terms_have_one_definition() -> None:
    """The factorized objective and the baseline's must be the same two numbers.

    Until 2026-10-05 the entangled baseline was trained with unweighted losses while the
    factorized model was class- and pos-weighted, so "the only difference is the absence
    of separation pressure" was false and the baseline's affect micro-F1 of 0.06 said
    nothing about separation.
    """
    from seam.affect.encoder import class_weights, direct_task_losses

    cfg = _cfg()
    torch.manual_seed(0)
    nm, p = torch.randn(32, cfg.dim_nm), torch.randn(32, cfg.dim_p)
    yl = (torch.rand(32) < 0.15).long()
    ya = (torch.rand(32, cfg.n_affect) < 0.2).float()
    w_l = class_weights(yl, cfg.n_linguistic)
    out = FactorizedEncoder(cfg)(nm, p)

    loss_l, loss_a = direct_task_losses(out, yl, ya, cfg, weight_l=w_l)
    _, parts = factorizer_loss(out, yl, ya, cfg, weight_l=w_l)
    assert parts["loss_l"] == pytest.approx(float(loss_l.detach()))
    assert parts["loss_a"] == pytest.approx(float(loss_a.detach()))

    # ...and that definition is the weighted one. On an imbalanced batch the weighted
    # and unweighted losses differ, so a baseline that drops the weights is detectable.
    plain_a = torch.nn.functional.binary_cross_entropy_with_logits(out["logits_a"], ya)
    plain_l = torch.nn.functional.cross_entropy(out["logits_l"], yl)
    assert float(loss_a.detach()) != pytest.approx(float(plain_a.detach()))
    assert float(loss_l.detach()) != pytest.approx(float(plain_l.detach()))


def test_checkpoints_are_selected_on_the_training_criterion_not_a_softmax() -> None:
    """Selection once used softmax cross-entropy over the eight affect logits.

    That was the single-expression framing surviving in one line after the training loss
    had moved to independent binary targets. A softmax is the wrong instrument for
    choosing a multi-label checkpoint for two reasons, both pinned here:

    * it is invariant to shifting every logit, so a model that calls **all eight**
      emotions present scores exactly as well as one that calls the right two;
    * a clip with no emotion above threshold contributes zero whatever is predicted -
      and 130 of the 1,765 windows are such clips - so false alarms on them are free.
    """
    from seam.affect.encoder import selection_loss

    cfg = _cfg()
    yl = torch.zeros(4, dtype=torch.long)
    logits_l = torch.zeros(4, cfg.n_linguistic)

    def crit(logits_a: torch.Tensor, ya: torch.Tensor) -> float:
        return selection_loss({"logits_l": logits_l, "logits_a": logits_a}, yl, ya, cfg)

    def softmax_ce(logits_a: torch.Tensor, ya: torch.Tensor) -> float:
        return float(torch.nn.functional.cross_entropy(logits_a, ya))

    two = torch.zeros(4, cfg.n_affect)
    two[:, :2] = 1.0  # two emotions present on every clip
    right = torch.full((4, cfg.n_affect), -6.0)
    right[:, :2] = 6.0
    everything = right + 20.0  # same ranking, every label called present

    assert softmax_ce(right, two) == pytest.approx(softmax_ce(everything, two), abs=1e-4)
    assert crit(right, two) < crit(everything, two)

    none = torch.zeros(4, cfg.n_affect)  # no emotion on these clips
    quiet = torch.full((4, cfg.n_affect), -6.0)
    alarm = torch.full((4, cfg.n_affect), 6.0)
    assert softmax_ce(quiet, none) == softmax_ce(alarm, none) == 0.0
    assert crit(quiet, none) < crit(alarm, none)


def test_affect_loss_rejects_a_class_index_target() -> None:
    """Affect is multi-label; a softmax-style class index must not silently train."""
    cfg = _cfg()
    out = FactorizedEncoder(cfg)(torch.randn(16, cfg.dim_nm), torch.randn(16, cfg.dim_p))
    yl = torch.randint(0, cfg.n_linguistic, (16,))
    y_class = torch.randint(0, cfg.n_affect, (16,))  # WRONG shape: 1D class index
    with pytest.raises((RuntimeError, ValueError)):
        factorizer_loss(out, yl, y_class, cfg)


def test_factors_are_separate_tensors_not_views() -> None:
    """Sharing a tensor between the branches would make separation impossible by
    construction, and would still train cleanly."""
    cfg = _cfg()
    out = FactorizedEncoder(cfg)(torch.randn(16, cfg.dim_nm), torch.randn(16, cfg.dim_p))
    assert out["z_l"] is not out["z_a"]
    assert out["z_l"].shape[1] == cfg.dim_z_l
    assert out["z_a"].shape[1] == cfg.dim_z_a


def test_forward_rejects_mismatched_batches() -> None:
    cfg = _cfg()
    with pytest.raises(ValueError, match="batch"):
        FactorizedEncoder(cfg)(torch.randn(8, cfg.dim_nm), torch.randn(9, cfg.dim_p))


def test_lambda_grid_ablates_exactly_one_mechanism() -> None:
    grid = default_lambda_grid(_cfg())
    base = grid.config_for("full")
    assert grid.config_for("no_orthogonality").w_orthogonality == 0.0
    assert grid.config_for("no_orthogonality").lambda_grl == base.lambda_grl
    no_sep = grid.config_for("no_separation")
    assert no_sep.w_orthogonality == 0.0 and no_sep.w_mi == 0.0 and no_sep.lambda_grl == 0.0
    # The grid must not mutate the config it was built from.
    assert base.w_orthogonality == pytest.approx(grid.base.w_orthogonality)


def test_lambda_grid_rejects_unknown_fields() -> None:
    grid = default_lambda_grid(_cfg())
    grid.variants["typo"] = {"w_orthogonalityy": 0.0}  # type: ignore[dict-item]
    with pytest.raises(KeyError, match="unknown fields"):
        grid.config_for("typo")


# --- LOSO ------------------------------------------------------------------


def test_loso_folds_are_signer_disjoint() -> None:
    groups = ["Ben"] * 7 + ["Cory"] * 87 + ["Jonathan"] * 54 + ["Rachel"] * 52
    splits = make_loso(groups)
    assert len(splits) == 4
    assert_no_signer_overlap(splits)
    for f in splits:
        assert f.signers_test == (f.held_out,)
        assert f.held_out not in f.signers_train


def test_a_leaky_split_cannot_be_constructed() -> None:
    """The assertion is a constructor invariant, not a convention."""
    import numpy as np

    from seam.eval.loso import _assert_disjoint

    groups = np.array(["A", "A", "B", "B"])
    _assert_disjoint(np.array([0, 1]), np.array([2, 3]), groups)  # fine
    # Either leak is caught; the sample-overlap check fires first, so match loosely.
    with pytest.raises(SplitOverlapError, match=r"appear in both|signer"):
        _assert_disjoint(np.array([0, 1]), np.array([1, 2]), groups)
    with pytest.raises(SplitOverlapError, match="empty"):
        _assert_disjoint(np.array([], dtype=int), np.array([2, 3]), groups)


def test_loso_covers_every_sample_exactly_once_for_test() -> None:
    groups = ["A"] * 10 + ["B"] * 20 + ["C"] * 30
    splits = make_loso(groups)
    seen = np.concatenate([f.test_idx for f in splits])
    assert sorted(seen.tolist()) == list(range(60))


# --- the gate metric -------------------------------------------------------


def test_auc_is_correct_on_known_cases() -> None:
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    assert np.isnan(auc(np.array([1, 1]), np.array([0.1, 0.2]))), "one class -> undefined"


def test_signer_control_is_at_chance_on_clean_factors() -> None:
    """The negative half of the positive control.

    A control that scores high on everything licenses nothing; it has to be high
    only when signer identity is actually recoverable.
    """
    rng = np.random.default_rng(0)
    signers = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 100)
    z_clean = rng.normal(size=(300, 16))
    assert signer_probe_auc(z_clean, signers) < 0.65


def test_signer_control_passes_on_memorised_factors() -> None:
    """The positive half: the metric must detect entanglement when it exists."""
    rng = np.random.default_rng(0)
    signers = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 100)
    codes = np.zeros((300, 16))
    for i, s in enumerate(signers):
        codes[i, "ABC".index(s)] = 1.0
    z_memo = rng.normal(size=(300, 16)) + codes * 6.0
    assert signer_probe_auc(z_memo, signers) >= SIGNER_CONTROL_MIN_AUC


def test_gate_refuses_when_the_control_fails() -> None:
    rng = np.random.default_rng(0)
    signers = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 100)
    z = rng.normal(size=(300, 16))
    ya = rng.integers(0, 2, 300)
    rep = SeparationReport(z, z, ya, ya, signers)
    rep.results = [cross_prediction(z, ya, label="z->affect")]
    v = rep.verdict()
    assert not v["control_passed"]
    assert not v["gate_passed"], "the gate must not pass on an insensitive metric"
    assert "no separation number" in v["reason"]


def test_gate_fails_on_a_genuinely_entangled_model() -> None:
    """A factor carrying the other factor's label must be caught, not passed."""
    rng = np.random.default_rng(0)
    signers = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 100)
    codes = np.zeros((300, 16))
    for i, s in enumerate(signers):
        codes[i, "ABC".index(s)] = 1.0
    y = rng.integers(0, 2, 300).astype(float)
    z_entangled = rng.normal(size=(300, 16)) + codes * 6.0
    z_entangled[:, 8] = y * 4.0  # the other factor's label, plainly present
    cp = cross_prediction(z_entangled, y, label="z->affect")
    assert cp.auc > 0.7, f"entanglement went undetected: AUC {cp.auc:.3f}"

    rep = SeparationReport(z_entangled, z_entangled, y, y, signers)
    rep.results = [cp]
    v = rep.verdict()
    assert v["control_passed"], "control should pass here - signer info is present"
    assert not v["gate_passed"], "an entangled model must not pass the gate"


def test_cross_prediction_is_cross_fitted() -> None:
    """A probe scored on its own training pairs would report spurious signal.

    Measured in `seam.affect.vclub`: the same detector reads 0.94 accuracy on
    training pairs and 0.4996 on held-out ones. `cross_prediction` must be the
    held-out number, and its permuted control must sit near chance.
    """
    rng = np.random.default_rng(0)
    z = rng.normal(size=(300, 16))
    y = rng.integers(0, 2, 300)
    cp = cross_prediction(z, y, label="z->noise")
    assert cp.auc < 0.62, f"noise gave AUC {cp.auc:.3f}"
    assert 0.4 < cp.auc_permuted < 0.6


def test_cross_prediction_reports_too_few_samples_instead_of_guessing() -> None:
    z = np.random.default_rng(0).normal(size=(10, 4))
    cp = cross_prediction(z, np.array([0, 1] * 5), label="tiny")
    assert not cp.interpretable
    assert "only 10 samples" in cp.note


def test_a_fold_that_cannot_measure_its_labels_is_not_interpretable() -> None:
    """A fold that skips most of its affect labels must not yield a gate number.

    The M4 Ben fold has 7 clips, 112 test windows, and 3 of 8 affect labels reaching
    the support threshold. It alone produced cross L->A = 0.884 and failed the gate on
    a number computed from the five labels it could not estimate. A metric that fails
    a model on data it never measured is not a metric.
    """
    rng = np.random.default_rng(0)
    z = rng.normal(size=(112, 16))
    y = np.zeros((112, 8))
    for k in range(3):
        y[:, k] = rng.random(112) < 0.25  # enough support
    for k in range(3, 8):
        y[:, k] = rng.random(112) < 0.02  # far too rare to estimate
    r = cross_prediction_multilabel(z, y, label="thin")
    assert r["n_usable"] == 3
    assert r["n_skipped"] == 5
    assert not r["coverage_ok"]
    assert r["n_labels"] == 8

    full = np.zeros((400, 8))
    for k in range(8):
        full[:, k] = rng.random(400) < 0.3
    r2 = cross_prediction_multilabel(rng.normal(size=(400, 16)), full, label="thick")
    assert r2["coverage_ok"], "a well-supported fold must stay interpretable"
