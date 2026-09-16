# Third-party baseline policy

The environment preparation script links existing `VGGT-Omega` and `VGGT4D`
repositories without modifying them, and clones small code-only baselines when
they are absent. Datasets and checkpoints are deliberately excluded from this
directory. Every cloned repository gets a `<name>.commit` file recording the
exact revision used for an experiment.
