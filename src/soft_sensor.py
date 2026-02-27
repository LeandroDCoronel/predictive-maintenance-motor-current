import argparse
import pandas as pd
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from glob import glob

def preprocess(df, sensor_type='current', max_value=100, window_size=100):
    if sensor_type not in df.columns:
        print(f"Warning: column {sensor_type} not found")
        return df
    df[sensor_type] = (df[sensor_type] - df[sensor_type].min()) / (df[sensor_type].max() - df[sensor_type].min()) * max_value
    df[f'{sensor_type}_smooth'] = df[sensor_type].rolling(window=window_size, min_periods=1).mean()
    return df

def process_csv(csv_file, output_folder, voltage=380, cost_hour=5000, window_size=100, sensor_type='current', show_plot=True):
    df = pd.read_csv(csv_file)
    df = preprocess(df, sensor_type=sensor_type, max_value=100, window_size=window_size)

    # Anomaly detection
    model = IsolationForest(contamination=0.01, random_state=42)
    df['anomaly'] = model.fit_predict(df[[f'{sensor_type}_smooth']])
    df['anomaly'] = df['anomaly'].map({1:0, -1:1})

    total_anomalies = df['anomaly'].sum()
    estimated_loss = (total_anomalies / len(df)) * cost_hour

    os.makedirs(output_folder, exist_ok=True)
    result_csv = os.path.join(output_folder, os.path.basename(csv_file))
    df.to_csv(result_csv, index=False)

    # --- Plotly interactive ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df[sensor_type], mode='lines', name=sensor_type))
    fig.add_trace(go.Scatter(y=df[f'{sensor_type}_smooth'], mode='lines', name=f'{sensor_type}_smooth'))

    anomalies_idx = df.index[df['anomaly']==1].tolist()
    fig.add_trace(go.Scatter(
        x=anomalies_idx,
        y=df.loc[anomalies_idx, f'{sensor_type}_smooth'],
        mode='markers',
        marker=dict(color='red', size=6),
        name='Anomalies'
    ))

    fig.update_layout(
        title=f"{os.path.basename(csv_file)} - Detected Anomalies",
        xaxis_title='Samples',
        yaxis_title='Normalized Value',
        template='plotly_white'
    )

    plot_file_html = os.path.join(output_folder, os.path.basename(csv_file).replace('.csv','.html'))
    fig.write_html(plot_file_html)

    if show_plot:
        fig.show()

    # TXT Report
    report_file = os.path.join(output_folder, os.path.basename(csv_file).replace('.csv','_report.txt'))
    with open(report_file,'w') as f:
        f.write(f"File: {os.path.basename(csv_file)}\n")
        f.write(f"Sensor: {sensor_type}\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Anomalies detected: {total_anomalies}\n")
        f.write(f"Estimated economic loss: ${estimated_loss:.2f}\n")
        f.write(f"Interactive plot: {plot_file_html}\n")
        f.write(f"CSV with anomalies: {result_csv}\n")

    print(f"{os.path.basename(csv_file)}: {total_anomalies} anomalies, estimated loss: ${estimated_loss:.2f}")
    return {
        'file': os.path.basename(csv_file),
        'anomalies_detected': total_anomalies,
        'estimated_loss': estimated_loss,
        'plot_file_html': plot_file_html,
        'result_csv': result_csv,
        'report_file': report_file
    }

def main():
    parser = argparse.ArgumentParser(description="Soft Sensor v2 - Motor anomaly detection")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", type=str, help="Path to a single CSV file")
    group.add_argument("--folder", type=str, help="Process all CSV files in this folder")
    parser.add_argument("--output", type=str, default="results", help="Folder to save results")
    parser.add_argument("--voltage", type=float, default=380, help="Operating voltage")
    parser.add_argument("--cost-hour", type=float, default=5000, help="Cost per hour of downtime")
    parser.add_argument("--window-size", type=int, default=100, help="Rolling window size for smoothing")
    parser.add_argument("--method", type=str, default="isolation_forest", help="Detection method (only isolation_forest)")
    parser.add_argument("--sensor", type=str, default=None, help="Sensor type: current, vibration, etc. (optional)")
    parser.add_argument("--no-show", action="store_true", help="Do not display plots automatically")
    args = parser.parse_args()

    if args.csv:
        files = [args.csv]
    else:
        files = glob(os.path.join(args.folder, "*.csv"))

    if not files:
        print("No CSV files found to process.")
        return

    summary = []

    for file in files:
        df_sample = pd.read_csv(file, nrows=1)
        if args.sensor:
            sensor = args.sensor
        elif 'current' in df_sample.columns:
            sensor = 'current'
        elif 'vibration' in df_sample.columns:
            sensor = 'vibration'
        else:
            sensor = df_sample.columns[0]

        result = process_csv(
            file,
            output_folder=args.output,
            voltage=args.voltage,
            cost_hour=args.cost_hour,
            window_size=args.window_size,
            sensor_type=sensor,
            show_plot=not args.no_show
        )
        summary.append(result)

    # Global summary CSV
    summary_df = pd.DataFrame(summary)
    summary_file = os.path.join(args.output, "summary_report.csv")
    summary_df.to_csv(summary_file, index=False)

    # --- Global plot ---
    global_plot_file = os.path.join(args.output, "global_anomalies_plot.html")
    fig_global = go.Figure()
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']

    for idx, result in enumerate(summary):
        df = pd.read_csv(result['result_csv'])
        sensor_col = 'current_smooth' if 'current_smooth' in df.columns else df.columns[0]
        fig_global.add_trace(go.Scatter(
            y=df[sensor_col],
            mode='lines',
            name=result['file'],
            line=dict(color=colors[idx % len(colors)], width=1),
            opacity=0.6
        ))
        anomalies_idx = df.index[df['anomaly']==1].tolist()
        fig_global.add_trace(go.Scatter(
            x=anomalies_idx,
            y=df.loc[anomalies_idx, sensor_col],
            mode='markers',
            marker=dict(color='red', size=4),
            showlegend=False
        ))

    fig_global.update_layout(
        title="Aggregated Anomalies of All Motors",
        xaxis_title="Samples",
        yaxis_title="Normalized Amperage",
        template='plotly_white'
    )

    fig_global.write_html(global_plot_file)
    if not args.no_show:
        fig_global.show()

    print(f"Global interactive plot saved to: {global_plot_file}")
    print(f"\nGlobal summary generated: {summary_file}")
    print(f"Total anomalies: {summary_df['anomalies_detected'].sum()}")
    print(f"Estimated total economic loss: ${summary_df['estimated_loss'].sum():.2f}")

if __name__ == "__main__":
    main()