"""Factorized non-manual encoding: linguistic and affective factors, learned apart.

The claim this package exists to support is that sign language carries two kinds of
information on the same face at the same time - a *linguistic* factor (`z_L`: the
brow raise of a question, the head shake of negation) and an *affective* factor
(`z_A`: how the utterance is coloured). A single-branch encoder has no way to keep
them apart, so a marker that happens to correlate with the emotional channel is
invisible to it and gets absorbed into the affect prediction - which is exactly the
confound M1 measured.

Separation is therefore not asserted, it is measured: frozen cross-probes are fitted
on one factor and asked to predict the other, and cross-prediction AUC near 0.5 is
the only acceptable result. That metric is only trustworthy if the probes can detect
entanglement when it exists, so :mod:`seam.eval.probes` carries a signer-identity
positive control that has to score high before any null is believed.
"""
