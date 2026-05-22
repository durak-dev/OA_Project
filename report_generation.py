import os
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def compile_saved_results_to_pdf(data_path, figures_dir, configs, output_pdf="final_experiment_report.pdf"):
    """
    Generates a professional evaluation PDF using ReportLab.
    Dynamically maps configuration hyper-parameters and targets files using exact project strings.
    """
    # 1. Load Data
    if data_path.endswith('.csv'):
        test_df = pd.read_csv(data_path)
    elif data_path.endswith('.pkl'):
        test_df = pd.read_pickle(data_path)
    else:
        raise ValueError("Unsupported data file format. Provide a CSV or PKL file.")

    test_df.columns = test_df.columns.str.strip()

    # 2. Compute Performance and Tukey Statistics
    group_metrics = []
    for name, group in test_df.groupby('Algorithm'):
        scores = group['Score']
        group_metrics.append(
            [name, f"{scores.mean():.4f}", f"{scores.std():.4f}", f"{scores.min():.4f}", f"{scores.max():.4f}"])

    tukey = pairwise_tukeyhsd(endog=test_df['Score'], groups=test_df['Algorithm'], alpha=0.05)
    tukey_data = []
    for row in tukey._results_table.data[1:]:
        p_val = f"{row[3]:.4e}" if row[3] > 0 else "0.0000e+00"
        reject_status = "YES" if row[6] else "NO"
        tukey_data.append(
            [str(row[0]), str(row[1]), f"{row[2]:.4f}", p_val, f"{row[4]:.4f}", f"{row[5]:.4f}", reject_status])

    # 3. Setup PDF Layout Document
    doc = SimpleDocTemplate(output_pdf, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('ReportTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=22,
                                 textColor=colors.HexColor('#1e3a8a'), spaceAfter=4)
    meta_style = ParagraphStyle('ReportMeta', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=9,
                                textColor=colors.HexColor('#64748b'), spaceAfter=15)
    h2_style = ParagraphStyle('SectionHeader', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=13,
                              textColor=colors.HexColor('#0f172a'), spaceBefore=16, spaceAfter=8)
    body_style = ParagraphStyle('ReportBody', parent=styles['Normal'], fontName='Helvetica', fontSize=9.5, leading=14,
                                spaceAfter=10)
    param_header_style = ParagraphStyle('ParamHead', parent=body_style, fontName='Helvetica-Bold', fontSize=9.5,
                                        textColor=colors.white)
    caption_style = ParagraphStyle('ImgCaption', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=8,
                                   textColor=colors.HexColor('#475569'), alignment=1, spaceBefore=4)

    # Document Header
    story.append(Paragraph("Metaheuristic Optimization Analysis Report", title_style))
    story.append(Paragraph("Evaluation Scope: Hyperparameter Execution Results | Alpha = 0.05", meta_style))

    intro_txt = ("This multi-comparison evaluation captures structural performance distributions "
                 "and checks statistical significance across metaheuristic architectures using the Tukey HSD framework.")
    story.append(Paragraph(intro_txt, body_style))

    # --- Section: Dynamic Hyperparameter Configuration Summary ---
    story.append(Paragraph("Experimental Hyperparameter Configurations", h2_style))
    param_headers = [Paragraph(f"<b>{h}</b>", param_header_style) for h in ["Algorithm", "Key Optimization Parameters"]]
    param_table_data = [param_headers]

    for c in configs:
        p_strings = []
        for k, v in c['params'].items():
            # Extract readable names if values are function objects
            val_str = v.__name__ if hasattr(v, '__name__') else str(v)
            p_strings.append(f"{k}: {val_str}")

        param_table_data.append([
            Paragraph(f"<b>{c['name']}</b>", body_style),
            Paragraph(", ".join(p_strings), body_style)
        ])

    param_table = Table(param_table_data, colWidths=[100, 430])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#475569')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # Changed to MIDDLE for a cleaner look if text wraps
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),  # Tightened padding to save vertical space
        ('TOPPADDING', (0, 0), (-1, -1), 4),  # Tightened padding
        ('LEFTPADDING', (1, 1), (1, -1), 6),  # Ensures parameters start close to the line
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
    ]))
    story.append(param_table)

    # --- Section 1: Descriptive Stats Table & Parallel Learning Curve Visualizations---
    story.append(Paragraph("2. Descriptive Performance Space and Performance History", h2_style))
    desc_headers = [Paragraph(f"<b>{h}</b>", ParagraphStyle('W', parent=body_style, textColor=colors.white)) for h in
                    ["Optimization Subsystem", "Mean Score", "Std Dev", "Min Bound", "Max Bound"]]
    desc_table_data = [desc_headers]
    for row in group_metrics:
        desc_table_data.append([Paragraph(f"<b>{row[0]}</b>", body_style), row[1], row[2], row[3], row[4]])

    t1 = Table(desc_table_data, colWidths=[180, 85, 85, 85, 85])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
    ]))
    story.append(t1)

    # Exact dynamic file name string match for performance history chart
    history_filename = f'algorithm_performance{"_".join([c["name"].replace(" ", "_") for c in configs])}.png'
    history_path = os.path.join(figures_dir, history_filename)

    if os.path.exists(history_path):
        story.append(Image(history_path, width=420, height=300))
        story.append(
            Paragraph(f"Figure 3.1: Complete learning convergence trajectory history comparison.", caption_style))
    else:
        story.append(Paragraph(f"<i>[Performance history plot asset not found at {history_path}]</i>", body_style))

    # --- Section 2: Tukey HSD Table ---
    story.append(Paragraph("2. Pairwise Tukey Honestly Significant Difference (HSD) Matrix", h2_style))
    tukey_headers = [Paragraph(f"<b>{h}</b>", ParagraphStyle('W', parent=body_style, textColor=colors.white)) for h in
                     ["Group 1", "Group 2", "Mean Diff", "Adj p-Value", "Lower", "Upper", "Reject H0"]]
    tukey_table_data = [tukey_headers]
    for row in tukey_data:
        r_style = ParagraphStyle('R', parent=body_style,
                                 textColor=colors.HexColor('#991b1b') if row[6] == "YES" else colors.HexColor(
                                     '#166534'))
        tukey_table_data.append(
            [row[0], row[1], row[2], row[3], row[4], row[5], Paragraph(f"<b>{row[6]}</b>", r_style)])

    t2 = Table(tukey_table_data, colWidths=[110, 110, 65, 75, 55, 55, 55])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
    ]))
    story.append(t2)


    # --- Section 4: Tukey Distributions ---
    story.append(Paragraph("3. Pairwise Studentized Range Distribution Curves (q-Plots)", h2_style))

    # Helper to clean/check/load images according to your string formula template
    def load_tukey_fig(algo1, algo2):
        fn = f'tukey_q_dist_{algo1}_vs_{algo2}.png'.lower().replace(" ", "_")
        path = os.path.join(figures_dir, fn)
        caption = f"Pairwise Contrast: {algo1} vs {algo2}"
        if os.path.exists(path):
            return [Image(path, width=250, height=135), Paragraph(caption, caption_style)]
        return [Paragraph(f"<i>[Missing: {fn}]</i>", body_style), Spacer(1, 10)]

    # Collect unique names to find matching saved images
    unique_names = sorted(list(test_df['Algorithm'].unique()))
    tukey_figs = []
    for i in range(len(unique_names)):
        for j in range(i + 1, len(unique_names)):
            tukey_figs.append(load_tukey_fig(unique_names[i], unique_names[j]))

    # Structure pairwise plots into an image grid loop (2 plots per row)
    grid_data = []
    for idx in range(0, len(tukey_figs), 2):
        row = tukey_figs[idx:idx + 2]
        if len(row) == 1:
            row.append("")  # Pad odd entries
        grid_data.append(row)

    if grid_data:
        img_table = Table(grid_data, colWidths=[260, 260])
        img_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        story.append(KeepTogether([img_table]))

    # Build PDF
    doc.build(story)
    print(f"Report compiled successfully: '{output_pdf}'")