# main.py
import argparse
import os
from pathlib import Path
import importlib.util


def load_module_from_path(py_path: str):
    py_path = os.path.abspath(py_path)
    if not os.path.isfile(py_path):
        raise FileNotFoundError(f"--exec_file not found: {py_path}")

    module_name = Path(py_path).stem  # e.g. id_aj
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {py_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def build_argparser():
    p = argparse.ArgumentParser()

    p.add_argument("--model_name", type=str, default="gpt-5",required=True,help="Model name (default: gpt-5)")
    p.add_argument("--input_file", type=str, default="PS-AJ.json", required=True)
    p.add_argument("--result_fp_base", type=str,default="PS-AJ", required=True)
    p.add_argument("--api_key", type=str, required=True)
    p.add_argument("--api_base", type=str, required=True)


    p.add_argument("--exec_file", type=str, default="id_aj.py", required=True,
                   help="Path to runner script, e.g. id_aj.py")

    return p


def main():
    args = build_argparser().parse_args()

    out_dir = Path(args.result_fp_base)
    out_dir.mkdir(parents=True, exist_ok=True)


    input_stem = Path(args.input_file).stem
    exec_stem = Path(args.exec_file).stem
    output_file = str(out_dir / f"{input_stem}.{exec_stem}.json")

    mod = load_module_from_path(args.exec_file)

    if not hasattr(mod, "run"):
        raise AttributeError(
            f"{args.exec_file} must define a function: run(model_name, input_file, output_file, api_key, api_base)"
        )


    mod.run(
        model_name=args.model_name,
        input_file=args.input_file,
        output_file=output_file,
        api_key=args.api_key,
        api_base=args.api_base,
    )


if __name__ == "__main__":
    main()
