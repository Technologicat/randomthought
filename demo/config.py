"""Configuration for the examples."""

# --------------------------------------------------------------------------------
# Common

fig_format = "png"

# --------------------------------------------------------------------------------
# CVAE

latent_dim = 7

vae_output_dir = "demo/output/vae/"

elbo_fig_filename = "elbo"

# E.g. "test_sample_0000.png" and so on; each of these must be a unique prefix,
# so that `anim.py` can shellglob over it.
test_sample_fig_basename = "test_sample"
latent_space_fig_basename = "latent_space"
overlay_fig_basename = "dataset"

test_sample_anim_filename = f"evolution_{test_sample_fig_basename}.gif"
latent_space_anim_filename = f"evolution_{latent_space_fig_basename}.gif"
overlay_anim_filename = f"evolution_{overlay_fig_basename}.gif"

# --------------------------------------------------------------------------------
# WLSQM

difftest_output_dir = "demo/output/difftest/"

surfaces_fig_basename = "surfaces"
errors_fig_basename = "errors"
