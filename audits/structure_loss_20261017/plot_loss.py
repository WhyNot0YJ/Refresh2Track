"""Plot completed-epoch, frame-weighted training losses from a captured snapshot."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main():
    directory = Path(__file__).resolve().parent
    data = json.loads((directory / 'loss_snapshot.json').read_text(encoding='utf-8'))
    colors = {'A_GBA': '#2563eb', 'B_GATE': '#e57d13', 'C_GBA_GATE': '#15803d'}
    plt.rcParams.update({'font.family': ['Microsoft YaHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), layout='constrained')
    panels = [('total_loss', 'Total loss'), ('task_loss', 'Task loss'),
              ('weighted_comm_loss', 'Communication penalty (0.05 x comm)')]
    rows, summaries = [], {}
    for name, model in data['models'].items():
        history = model['history']
        epochs = [row['epoch'] for row in history]
        for axis, (key, title) in zip(axes, panels):
            values = [row[key] for row in history]
            axis.plot(epochs, values, color=colors[name], marker='o', markersize=4,
                      linewidth=2, label=name)
            axis.annotate(format(values[-1], '.4f'), (epochs[-1], values[-1]),
                          xytext=(5, 5), textcoords='offset points',
                          color=colors[name], fontsize=9)
            axis.set_title(title, fontsize=11, fontweight='bold', pad=12)
            axis.set_xlabel('Completed epoch')
            axis.set_xlim(.6, 9.9)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(axis='y', color='#d9dfe6', linewidth=.7)
            axis.set_axisbelow(True)
        for row in history:
            if abs(row['total_loss'] - row['task_loss'] - row['weighted_comm_loss']) > 1e-6:
                raise AssertionError('Loss decomposition differs from the recorded objective')
            if abs(row['weighted_comm_loss'] - .05 * row['comm_loss']) > 1e-6:
                raise AssertionError('Communication coefficient changed')
            rows.append(dict(model=name, **row))
        first, last = history[0], history[-1]
        sixth = next(row for row in history if row['epoch'] == 6)
        summaries[name] = {
            'last_completed_epoch': last['epoch'], 'total_epoch1': first['total_loss'],
            'total_epoch6': sixth['total_loss'], 'total_last': last['total_loss'],
            'total_drop_percent': 100 * (first['total_loss'] - last['total_loss']) / first['total_loss'],
            'task_epoch1': first['task_loss'], 'task_epoch6': sixth['task_loss'],
            'task_last': last['task_loss'], 'comm_epoch1': first['comm_loss'],
            'comm_epoch6': sixth['comm_loss'], 'comm_last': last['comm_loss'],
            'nonfinite_batch_values': sum(row['nonfinite'] for row in model['component_means_by_epoch']),
            'duplicate_log_records': model['duplicate_records'],
            'late_total_loss_by_epoch': {str(row['epoch']): row['total_loss'] for row in history if row['epoch'] >= 3},
        }
    axes[0].legend(frameon=False, loc='upper right', fontsize=9)
    figure.suptitle('A / B / C training-loss trends', fontsize=16, fontweight='bold')
    figure.supxlabel('Frame-weighted epoch means; 7,362 training frames per epoch. Total = task + 0.05 x comm.\n'
                     'Partial epochs excluded. Snapshot: 2026-10-17 03:54 Asia/Shanghai.', fontsize=9)
    figure.savefig(directory / 'loss_trends.png', dpi=180, facecolor='white')
    figure.savefig(directory / 'loss_trends.svg', facecolor='white')
    plt.close(figure)
    with (directory / 'epoch_losses.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {'checked_utc': data['checked_utc'], 'averaging': 'per real frame, completed epochs only',
              'objective': 'task_loss + 0.05 * comm_loss', 'models': summaries,
              'limits': ['No comparable validation-loss curve is recorded; AMOTA/BPS are separate validation metrics.',
                         'Loss curves alone do not establish overfitting or final convergence.',
                         'Soft probability communication loss and hard thresholded validation BPS need not move together.']}
    (directory / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'plot': str(directory / 'loss_trends.png'), 'models': summaries}))


if __name__ == '__main__':
    main()
