import os
import yaml
import subprocess

from src.utils.io import load_json, save_json, ensure_dir

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main(cfg_exp="configs/experiment.yaml", cfg_model="configs/model.yaml"):
    exp = load_yaml(cfg_exp)
    model_cfg = load_yaml(cfg_model)

    n_list = exp["ablation"]["n_blocks_list"]
    out_dir = exp["run"]["out_dir"]
    exp_name = exp["run"]["name"]

    results = []
    for n_blocks in n_list:
        # Patch model.yaml in-memory approach is messy; easiest:
        # run train/eval with env var override or write temp yaml.
        # Here: write a temp model yaml per n_blocks.
        tmp_model_path = f"/tmp/model_blocks_{n_blocks}.yaml"
        patched = model_cfg.copy()
        patched["model"]["resnet"]["n_blocks"] = int(n_blocks)
        with open(tmp_model_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(patched, f, sort_keys=False, allow_unicode=True)

        print(f"\n==== Ablation n_blocks={n_blocks} ====")
        subprocess.check_call(["python", "scripts/train.py", "--cfg_model", tmp_model_path])
        subprocess.check_call(["python", "scripts/eval.py", "--cfg_model", tmp_model_path])

        run_dir = os.path.join(out_dir, exp_name, f"ablation_{n_blocks}_blocks", f"blocks_{n_blocks}_dilations_{'_'.join(map(str, patched['model']['resnet']['dilations']))}")
        os.makedirs(run_dir, exist_ok=True)
        rep_path = os.path.join(run_dir, "eval_report.json")
        rep = load_json(rep_path)

        results.append({
            "n_blocks": n_blocks,
            "seg_acc_mean": rep["segment"]["accuracy_mean"],
            "seg_acc_std": rep["segment"]["accuracy_std"],
            "seg_f1_mean": rep["segment"]["f1_mean"],
            "seg_f1_std": rep["segment"]["f1_std"],
            "subj_acc_mean": rep["subject"]["accuracy_mean"],
            "subj_acc_std": rep["subject"]["accuracy_std"],
            "subj_f1_mean": rep["subject"]["f1_mean"],
            "subj_f1_std": rep["subject"]["f1_std"],
            "seg_thr": rep["segment"]["threshold"],
            "subj_thr": rep["subject"]["threshold"],
        })

    out_json = os.path.join(out_dir, exp_name, f"ablation_results_{'_'.join(map(str, patched['model']['resnet']['dilations']))}.json")
    ensure_dir(os.path.dirname(out_json))
    save_json(out_json, {"results": results})

    # also save a simple markdown table
    md_lines = []
    md_lines.append("| n_blocks | subj_f1 (mean±std) | subj_acc (mean±std) | seg_f1 (mean±std) | seg_acc (mean±std) |")
    md_lines.append("|---:|---:|---:|---:|---:|")
    for r in results:
        md_lines.append(
            f"| {r['n_blocks']} | {r['subj_f1_mean']:.4f}±{r['subj_f1_std']:.4f} | "
            f"{r['subj_acc_mean']:.4f}±{r['subj_acc_std']:.4f} | "
            f"{r['seg_f1_mean']:.4f}±{r['seg_f1_std']:.4f} | "
            f"{r['seg_acc_mean']:.4f}±{r['seg_acc_std']:.4f} |"
        )
    out_md = os.path.join(out_dir, exp_name, f"ablation_results_{'_'.join(map(str, patched['model']['resnet']['dilations']))}.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("Saved:", out_json)
    print("Saved:", out_md)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_model", default="configs/model.yaml")
    ap.add_argument("--cfg_exp", default="configs/experiment.yaml")
    args = ap.parse_args()
    main(args.cfg_exp, args.cfg_model)
