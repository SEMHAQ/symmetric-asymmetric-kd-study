"""Analyze KD experiment results and generate comparison tables + figures."""
import os
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)


RESULTS_DIR = '../experiments/results'
OUTPUT_DIR = '../experiments/figures'

# Method display names and categories
METHOD_INFO = {
    'BASELINE': {'label': 'Baseline (No KD)', 'cat': 'baseline'},
    'KD':       {'label': 'KD (Hinton 2015)', 'cat': 'symmetric'},
    'FITNET':   {'label': 'FitNet (2015)',    'cat': 'symmetric'},
    'AT':       {'label': 'AT (2017)',        'cat': 'symmetric'},
    'SP':       {'label': 'SP (2019)',        'cat': 'asymmetric'},
    'CC':       {'label': 'CC (2020)',        'cat': 'asymmetric'},
}

CAT_COLORS = {
    'baseline':  '#95a5a6',   # gray
    'symmetric': '#3498db',   # blue
    'asymmetric': '#e74c3c',  # red
}


def load_results():
    """Load all experiment results."""
    results = {}
    for method_name in METHOD_INFO:
        path = os.path.join(RESULTS_DIR, f'KD_{method_name}', 'history.json')
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            # Determine if this is a teacher run
            if method_name == 'TEACHER':
                results['TEACHER'] = data
            else:
                results[method_name] = data
            print(f'  ✓ Loaded KD_{method_name}: best_acc={data.get("best_acc", "N/A"):.2f}%')
        else:
            print(f'  ✗ Missing: {path}')
    return results


def load_teacher_results():
    """Load teacher results."""
    teacher_path = os.path.join(RESULTS_DIR.replace('results', 'checkpoints'),
                                'teacher_history.json')
    # Try alternate locations
    for p in [
        '../experiments/checkpoints/teacher_history.json',
        os.path.join(RESULTS_DIR, 'teacher_history.json'),
    ]:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return None


def print_comparison_table(results):
    """Print a clean comparison table."""
    print('\n' + '='*85)
    print(f'{"Method":<30} {"Category":<15} {"Best Acc%":<12} {"Improvement":<15}')
    print('='*85)

    # Show teacher if available
    teacher_acc = None
    for rname, rdata in results.items():
        if 'TEACHER' in rname.upper() or rname == 'teacher':
            teacher_acc = rdata.get('best_acc', rdata.get('test_acc', [0])[-1])
            print(f'{"Teacher ResNet-34":<30} {"—":<15} {teacher_acc:<12.2f} {"—":<15}')

    print('-'*85)

    baseline_acc = None
    rows = []
    for method_name in ['BASELINE', 'KD', 'FITNET', 'AT', 'SP', 'CC']:
        if method_name not in results:
            continue
        data = results[method_name]
        best = data.get('best_acc', 0)
        cat = METHOD_INFO[method_name]['cat']
        label = METHOD_INFO[method_name]['label']

        if method_name == 'BASELINE':
            baseline_acc = best
            impr = '—'
        elif baseline_acc:
            impr = f'+{best - baseline_acc:.2f}%'
        else:
            impr = '—'

        rows.append((label, cat, best, impr))
        cat_display = {'baseline': '—', 'symmetric': '对称', 'asymmetric': '不对称'}[cat]

        print(f'{label:<30} {cat_display:<15} {best:<12.2f} {impr:<15}')

    print('='*85)

    # Category averages
    if baseline_acc:
        sym_accs = [r[2] for r in rows if METHOD_INFO.get(r[0].split('(')[0].strip().upper().split()[0], {}).get('cat') == 'symmetric']
        # simpler approach
        sym_accs = []
        asym_accs = []
        for label, cat, best, _ in rows:
            if cat == 'symmetric':
                sym_accs.append(best)
            elif cat == 'asymmetric':
                asym_accs.append(best)

        if sym_accs:
            print(f'\nSymmetric avg: {np.mean(sym_accs):.2f}% '
                  f'(vs baseline {baseline_acc:.2f}%, +{np.mean(sym_accs) - baseline_acc:.2f}%)')
        if asym_accs:
            print(f'Asymmetric avg: {np.mean(asym_accs):.2f}% '
                  f'(vs baseline {baseline_acc:.2f}%, +{np.mean(asym_accs) - baseline_acc:.2f}%)')

    return baseline_acc


def plot_learning_curves(results):
    """Plot test accuracy curves for all methods."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(12, 6))
    markers = ['o', 's', 'D', '^', 'v', '<', '>']

    for i, method_name in enumerate(['BASELINE', 'KD', 'FITNET', 'AT', 'SP', 'CC']):
        if method_name not in results:
            continue
        data = results[method_name]
        test_acc = data.get('test_acc', [])
        epochs = range(1, len(test_acc) + 1)
        label = METHOD_INFO[method_name]['label']
        cat = METHOD_INFO[method_name]['cat']
        color = CAT_COLORS[cat]
        marker = markers[i % len(markers)]

        plt.plot(epochs, test_acc, label=label, color=color,
                 marker=marker, markevery=max(1, len(test_acc)//8),
                 linewidth=1.5, markersize=6)

    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('CIFAR-100: Student ResNet-18 Learning Curves')
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'learning_curves.png'), dpi=200)
    plt.close()
    print(f'  ✓ Saved: {OUTPUT_DIR}/learning_curves.png')


def plot_bar_comparison(results, baseline_acc):
    """Bar chart comparing all methods."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    methods = ['BASELINE', 'KD', 'FITNET', 'AT', 'SP', 'CC']
    labels = [METHOD_INFO[m]['label'] for m in methods]
    accs = [results[m]['best_acc'] if m in results else 0 for m in methods]
    cats = [METHOD_INFO[m]['cat'] for m in methods]
    colors = [CAT_COLORS[c] for c in cats]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(methods)), accs, color=colors, edgecolor='gray', linewidth=0.5)

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add improvement arrows
    if baseline_acc:
        for i, acc in enumerate(accs):
            if i > 0 and acc > baseline_acc:
                plt.annotate(f'+{acc - baseline_acc:.2f}%',
                            xy=(i, acc), xytext=(i, acc + 1.2),
                            ha='center', fontsize=9, color='green')

    plt.xticks(range(len(methods)), labels, rotation=30, ha='right', fontsize=11)
    plt.ylabel('Best Test Accuracy (%)')
    plt.title('CIFAR-100: Symmetric vs Asymmetric KD Methods (ResNet-34 → ResNet-18)')
    plt.ylim(min(accs) - 2, max(accs) + 2)

    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CAT_COLORS['symmetric'], label='Symmetric KD'),
        Patch(facecolor=CAT_COLORS['asymmetric'], label='Asymmetric KD'),
        Patch(facecolor=CAT_COLORS['baseline'], label='Baseline'),
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'bar_comparison.png'), dpi=200)
    plt.close()
    print(f'  ✓ Saved: {OUTPUT_DIR}/bar_comparison.png')


def plot_category_comparison(results, baseline_acc):
    """Box/violin plot comparing symmetric vs asymmetric categories."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sym_accs = [results[m]['best_acc'] for m in ['KD', 'FITNET', 'AT']
                if m in results]
    asym_accs = [results[m]['best_acc'] for m in ['SP', 'CC']
                 if m in results]

    if not sym_accs or not asym_accs:
        print('  ⚠ Not enough data for category comparison')
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Data for violin
    data = [sym_accs, asym_accs, [baseline_acc]]
    labels = ['Symmetric KD\n(Instance-level)', 'Asymmetric KD\n(Structure-level)', 'Baseline\n(No KD)']
    colors = [CAT_COLORS['symmetric'], CAT_COLORS['asymmetric'], CAT_COLORS['baseline']]

    vp = ax.violinplot(data, showmeans=True, showmedians=True)
    for i, pc in enumerate(vp['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    # Scatter points
    for i, d in enumerate(data):
        y = d
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.8, s=60, color=colors[i], edgecolors='black', linewidth=0.5, zorder=5)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Best Test Accuracy (%)', fontsize=12)
    ax.set_title('CIFAR-100: Symmetric vs Asymmetric KD Comparison')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'category_comparison.png'), dpi=200)
    plt.close()
    print(f'  ✓ Saved: {OUTPUT_DIR}/category_comparison.png')


def print_analysis(results, baseline_acc):
    """Print analysis summary."""
    sym_methods = ['KD', 'FITNET', 'AT']
    asym_methods = ['SP', 'CC']

    sym_accs = [results[m]['best_acc'] for m in sym_methods if m in results]
    asym_accs = [results[m]['best_acc'] for m in asym_methods if m in results]

    print('\n' + '='*60)
    print('📊 ANALYSIS SUMMARY')
    print('='*60)

    print(f'\n  Teacher (ResNet-34): {results.get("TEACHER", {}).get("best_acc", "?")}%')

    if sym_accs:
        print(f'\n  Symmetric KD (average):  {np.mean(sym_accs):.2f}%')
        print(f'    - KD:     {results["KD"]["best_acc"]:.2f}%')
        print(f'    - FitNet: {results["FITNET"]["best_acc"]:.2f}%')
        print(f'    - AT:     {results["AT"]["best_acc"]:.2f}%')

    if asym_accs:
        print(f'\n  Asymmetric KD (average): {np.mean(asym_accs):.2f}%')
        print(f'    - SP: {results["SP"]["best_acc"]:.2f}%')
        print(f'    - CC: {results["CC"]["best_acc"]:.2f}%')

    if 'DELTA' not in results:
        print(f'\n  Baseline (ResNet-18):    {baseline_acc:.2f}%')
        if sym_accs and baseline_acc:
            print(f'  Symmetric vs Baseline:   +{np.mean(sym_accs) - baseline_acc:.2f}%')
        if asym_accs and baseline_acc:
            print(f'  Asymmetric vs Baseline:  +{np.mean(asym_accs) - baseline_acc:.2f}%')
            print(f'  Asymmetric vs Symmetric: {np.mean(asym_accs) - np.mean(sym_accs):+.2f}%')

    print('\n  💡 Key finding:')
    max_sym = max(sym_accs) if sym_accs else 0
    max_asym = max(asym_accs) if asym_accs else 0

    if max_asym > max_sym:
        print(f'    Asymmetric methods ({max_asym:.2f}%) outperformed symmetric methods ({max_sym:.2f}%)')
    elif max_sym > max_asym:
        print(f'    Symmetric methods ({max_sym:.2f}%) outperformed asymmetric methods ({max_asym:.2f}%)')
    else:
        print(f'    Comparable performance (~{max_sym:.2f}%)')

    all_improve = (sym_accs and asym_accs and
                   all(a > baseline_acc for a in sym_accs + asym_accs))
    if all_improve:
        print(f'    ✓ ALL KD methods improve over baseline — taxonomy is valid!')
    print('='*60)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('\nLoading results...')
    results = load_results()

    if not results:
        print('No results found. Run experiments first.')
        return

    # Add teacher results if available
    teacher_data = load_teacher_results()
    if teacher_data:
        results['TEACHER'] = teacher_data

    # Print table
    baseline_acc = print_comparison_table(results)

    # Generate plots
    print('\nGenerating figures...')
    plot_learning_curves(results)
    plot_bar_comparison(results, baseline_acc)
    plot_category_comparison(results, baseline_acc)

    # Analysis
    print_analysis(results, baseline_acc)

    print(f'\nAll figures saved to: {os.path.abspath(OUTPUT_DIR)}')


if __name__ == '__main__':
    main()
