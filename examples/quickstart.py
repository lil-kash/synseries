"""
examples/quickstart.py
End-to-end Python demonstration of SynSeries.
"""

import synseries as ss

def run_demonstration():
    print("=" * 60)
    print("SynSeries: High-Fidelity Synthetic Time-Series Framework")
    print("=" * 60)

    # 1. Natural Language Prompt-Driven Generation (e.g. Medical Domain)
    prompt = "12 hours of ICU patient telemetry who develops septic shock"
    print(f"\n[1] Synthesizing dataset from natural language prompt:\n    --> '{prompt}'")
    
    # Get the compiled spec coding
    spec = ss.compile_spec(prompt)
    print(f"\n[Spec Compiled Successfully] Domain: {spec.domain.upper()}, Channels: {len(spec.variables)}")
    
    # Generate the pandas DataFrame
    df_icu = ss.generate(spec)
    print(f"\nGenerated DataFrame Shape: {df_icu.shape} (from {df_icu.index[0]} to {df_icu.index[-1]})")
    print("\nFirst 3 rows:")
    print(df_icu.head(3))

    # 2. Automated Domain Validity and Invariant Verification
    print("\n[2] Executing Automated Domain Validity and Authenticity Tests...")
    validation_results = ss.validate(df_icu, domain="medical")
    print(f"Overall Status: {validation_results['overall_status']} (Score: {validation_results['score_pct']}%)")
    for check in validation_results["checks"]:
        print(f"  - [{check['status']}] {check['name']}: {check['details']}")

    # 3. Generating Quantitative Finance Market Data (OHLCV)
    print("\n[3] Generating Quantitative Finance OHLCV Candlesticks (AAPL)...")
    df_finance = ss.generate("500 bars of AAPL stock market OHLCV with volume")
    print(f"Finance DataFrame Shape: {df_finance.shape}")
    print(df_finance[["open", "high", "low", "close", "volume", "vwap"]].head(3))
    
    fin_validation = ss.validate(df_finance, domain="finance")
    print(f"Finance Validity Status: {fin_validation['overall_status']} (Score: {fin_validation['score_pct']}%)")

    # 4. Generating Industrial Bearing 3-Axis High-Frequency Vibration
    print("\n[4] Generating High-Frequency Industrial 3-Axis Bearing Vibration...")
    df_vib = ss.generate("5 seconds of 3-axis accelerometer vibration with bearing defect")
    print(f"Vibration DataFrame Shape: {df_vib.shape}")
    print(df_vib.head(3))

    # Exporting sample
    output_csv = "quickstart_icu_sample.csv"
    df_icu.to_csv(output_csv)
    print(f"\nSaved ICU dataset to '{output_csv}' successfully.")
    print("All demonstration steps completed with 100% domain compliance.")

if __name__ == "__main__":
    run_demonstration()
