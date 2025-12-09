import numpy as np
import os
import cProfile
import shutil
import subprocess

from bart_playground.generators import generate_defaultbart_prior_with_cov, generate_data_heatmap

def _default_params_call(n, p):
    return generate_defaultbart_prior_with_cov(
        n=n,
        p=p,
        n_trees=200,
        random_state=42,
        max_depth=10000,
        quick_decay=False,
        return_latent=True
    )
    
def test_all_close():
    n = 20000
    p = 5
    res = _default_params_call(n, p)
    X, y, f, sigma2, trees = res # type: ignore
    assert X.shape == (n, p)
    assert f.shape == (n,)
    assert np.allclose(np.sum([t.evaluate(X) for t in trees], axis=0), f)

def _cleanup(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

def test_generate_data_heatmap():
    out_path = generate_data_heatmap(output_dir='tests/output')
    assert isinstance(out_path, str)
    assert os.path.exists(out_path)
    _cleanup(out_path)

if __name__ == "__main__":
    # Parameters
    n = 20000
    p = 2
    
    file_name = "profile_bart_generator"
    # Output files (use variables instead of hard-coded inline names)
    profile_path = f"tests/output/{file_name}.prof"
    dot_path = f"tests/output/{file_name}.dot"
    png_path = f"tests/output/{file_name}.png"

    # Profile the call and dump stats to file
    profiler = cProfile.Profile()
    profiler.enable()
    _default_params_call(n, p)
    profiler.disable()
    profiler.dump_stats(profile_path)

    # Convert profile to dot if available
    if shutil.which("gprof2dot"):
        subprocess.run(["gprof2dot", "-f", "pstats", profile_path, "-o", dot_path], check=False)
    else:
        print("gprof2dot not found; skipping DOT generation")

    # Convert dot to png if available and dot file exists
    if shutil.which("dot") and os.path.exists(dot_path):
        subprocess.run(["dot", "-Tpng", dot_path, "-o", png_path], check=False)
    else:
        print("Graphviz 'dot' not found; skipping PNG generation")
    