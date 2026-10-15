import pandas as pd

PRICE = {
    "easy": (1.00,  5.00),   # Claude Haiku 4.5
    "hard": (5.00, 25.00),   # Claude Opus 5
}

df = pd.read_csv("results_run.csv")      
df["in_tok"] = df["prompt"].str.split().str.len() * 1.3   # rough estimate

def total(tier, out_tok, fixed=False):
    s = 0.0
    for r in df.itertuples():
        t = tier if fixed else getattr(r, tier)
        pin, pout = PRICE[t]
        s += r.in_tok / 1e6 * pin + out_tok / 1e6 * pout
    return s

print("=== COST FOR 200 PROMPTS ===")
print(f"{'out tok':>8} {'always-Opus':>13} {'router':>11} {'saving':>9} {'perfect':>9}")
for out_tok in [100, 300, 1000]:
    a  = total("hard", out_tok, fixed=True)    # baseline: everything to Opus
    rt = total("pred_complexity", out_tok)     # what the router actually does
    pf = total("complexity", out_tok)          # upper bound: perfect classifier
    print(f"{out_tok:>8} ${a:>12.4f} ${rt:>10.4f} {1-rt/a:>8.1%} {1-pf/a:>8.1%}")

print("\n=== TIER MIX ===")
print(pd.DataFrame({
    "routed to":  df.pred_complexity.value_counts(normalize=True),
    "should be":  df.complexity.value_counts(normalize=True),
}).reindex(["easy", "hard"]).to_string(float_format=lambda x: f"{x:.1%}"))

print("\n=== MISROUTES ===")
under = ((df.complexity == "hard") & (df.pred_complexity == "easy")).sum()
over  = ((df.complexity == "easy") & (df.pred_complexity == "hard")).sum()
print(f"under-routed (hard sent to Haiku) {under:3}/{len(df)} = {under/len(df):.1%}  <- quality risk")
print(f"over-routed  (easy sent to Opus)  {over:3}/{len(df)} = {over/len(df):.1%}  <- cost only")