"""Make a pdf containing plots for each guideline ramp showing the space filled
by all the enumerated transitions filters from that ramp. Plots should show that
the space filled is the full domain (marked by the dark black box on the plot)
with no overlaps or gaps."""

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.backends.backend_pdf

import pandas as pd


def draw_rectangles(ax, state_filters, facecolor='r', edgecolor='b', alpha=0.4):
    filters = []
    for r in state_filters.iterrows():
        xy = (r[1].age_group_start, r[1].systolic_blood_pressure_start)
        rect = Rectangle(xy, r[1].width, r[1].height)
        filters.append(rect)

    pc = PatchCollection(filters, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor)

    ax.add_collection(pc)


def plot_guideline_ramp(data, figure_name, pdf):
    i = 1

    fig = plt.figure()

    for sex, cvd_risk_cat in itertools.product(('Male', 'Female'), (0, 1)):
        ax = fig.add_subplot(2, 2, i)
        draw_rectangles(ax, data[(data.sex == sex) & (data.cvd_risk_cat == cvd_risk_cat)])
        ax.set_ylim(0, 360)
        ax.set_xlim(-10, 135)

        container = Rectangle((0, 60), 125, 240, fill=False, edgecolor='black', lw=4)
        ax.add_patch(container)
        ax.title.set_text(f'({sex}, {cvd_risk_cat})')
        i += 1

    fig.suptitle(figure_name)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    pdf.savefig(fig)


def plot_ramp(data, pdf):
    if data.name[1] == 'initial' and {'mono_starter', 'combo_starter'}.issubset(set(data.to_ramp)):
        dn = data.name
        fig_name = (dn[0], f'{dn[1]}: to mono_starter only')
        plot_guideline_ramp(data[data.to_ramp != 'combo_starter'], fig_name, pdf)
        fig_name = (dn[0], f'{dn[1]}: to combo_starter only')
        plot_guideline_ramp(data[data.to_ramp != 'mono_starter'], fig_name, pdf)
    else:
        plot_guideline_ramp(data, data.name, pdf)


def main():
    data_file = Path(__file__).parent.parent / 'external_data/ramp_transition_filters.csv'
    ramp_transitions = pd.read_csv(data_file)
    ramp_transitions['width'] = ramp_transitions.age_group_end - ramp_transitions.age_group_start
    ramp_transitions['height'] = (ramp_transitions.systolic_blood_pressure_end
                                  - ramp_transitions.systolic_blood_pressure_start)

    pdf = matplotlib.backends.backend_pdf.PdfPages("filter_ramp_transitions_data_validation.pdf")
    ramp_transitions.groupby(['guideline', 'from_ramp']).apply(plot_ramp, pdf)
    pdf.close()


if __name__ == "__main__":
    main()

