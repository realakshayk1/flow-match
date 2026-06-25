import pandas as pd
from posebusters import PoseBusters
import json
import os
import numpy as np

def process_dir(run_dir):
    csv_path = os.path.join(run_dir, "results_raw.csv")
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    
    pb = PoseBusters(config="mol")
    
    new_rows = []
    
    for i, row in df.iterrows():
        cid = row["complex_id"]
        sdf_path = os.path.join(run_dir, "poses", f"{cid}_pred.sdf")
        row_dict = row.to_dict()
        row_dict["pb_error"] = None
        
        if os.path.exists(sdf_path):
            try:
                pb_res = pb.bust(sdf_path)
                if len(pb_res) > 0:
                    for col in pb_res.columns:
                        row_dict[col] = pb_res.iloc[0][col]
            except Exception as e:
                row_dict["pb_error"] = str(e)
        else:
            row_dict["pb_error"] = "SDF not found"
            
        new_rows.append(row_dict)
        
    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(csv_path, index=False)
    
    check_cols = [c for c in new_df.columns
                  if c not in ("complex_id", "rmsd", "uff_postprocess", "uff_failed", "pb_error")
                  and new_df[c].dtype == bool]
                  
    if check_cols:
        new_df["pb_valid"] = new_df[check_cols].all(axis=1)
        pb_valid_rate = new_df["pb_valid"].mean() * 100
        
        rmsd_all = new_df["rmsd"].dropna().to_numpy()

        summary = {
            "n_total": len(new_df),
            "n_pb_valid": int(new_df["pb_valid"].sum()),
            "pb_valid_pct": round(pb_valid_rate, 1),
            "rmsd_median": round(float(np.median(rmsd_all)), 3) if len(rmsd_all) else None,
            "rmsd_pct_under_2A": round(float((rmsd_all < 2.0).mean() * 100), 1) if len(rmsd_all) else None,
            "per_check_pass_rate": {
                col: round(new_df[col].mean() * 100, 1)
                for col in sorted(check_cols)
            }
        }
        
        summary_path = os.path.join(run_dir, "results_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Processed {run_dir} -> {summary['pb_valid_pct']}% PB-valid")

process_dir("eval/posebusters_raw")
process_dir("eval/posebusters_uff")
