# ---------------  plot_metrics.py  ----------------
import pandas as pd
import matplotlib.pyplot as plt
import glob, os, sys

plt.style.use("default")

def plot_single_csv(filepath):

    df = pd.read_csv(filepath)
    if {"Step", "Value"}.issubset(df.columns):
        x = df["Step"].astype(float)
        y = df["Value"].astype(float)

        plt.figure(figsize=(10, 4))
        plt.plot(x, y, linewidth=2, label=os.path.basename(filepath))
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.title(f"{os.path.basename(filepath)} Curve")
        plt.grid(True)
        plt.tight_layout()
        out_png = f"{os.path.splitext(filepath)[0]}_curve.png"
        plt.savefig(out_png)
        plt.close()
        print(f"[✓] Saved plot → {out_png}")

        # 也可回傳最後 3 筆均值、最終值供 Summary 表使用
        return {
            "file": os.path.basename(filepath),
            "avg_last3": y.tail(3).mean(),
            "final_value": y.iloc[-1],
        }
    else:
        print(f"[!] Skip {filepath} (missing Step / Value columns)")
        return None

def main():
    # 允許使用者以參數指定檔案，否則抓取目前資料夾所有 *.csv
    csv_files = sys.argv[1:] if len(sys.argv) > 1 else glob.glob("*.csv")
    summary = []

    for csv in csv_files:
        info = plot_single_csv(csv)
        if info:
            summary.append(info)

    # 將 summary 輸出成 table.txt 方便貼到投影片
    if summary:
        with open("summary_table.txt", "w") as f:
            f.write("| File | Avg(last3) | Final |\n|------|-------------|-------|\n")
            for row in summary:
                f.write(f"| {row['file']} | {row['avg_last3']:.4f} | {row['final_value']:.4f} |\n")
        print("[✓] Summary table saved → summary_table.txt")

if __name__ == "__main__":
    main()
# --------------------------------------------------
