from django.shortcuts import render
from Main.form import ParamForm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from Main.scripts import call_all_funcs
from settings.settings import BASE_DIR


def total_hist_by_protocol(data, save_to_file=False):
    grouped_by_protocol = data.groupby(['Protocol'])['No.'].count().sort_values()
    plt.figure(figsize=(10, 5))

    objects = grouped_by_protocol.keys()
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, grouped_by_protocol.values, align='center')
    plt.xticks(y_pos, objects, rotation=90)

    plt.ylabel('Requests')
    plt.title('Hist by protocol')
    if not save_to_file:
        plt.show()
    else:
        plt.savefig('static/total_by_protocol.png')
    return "TEXT"


def main(request):
    images = []
    if request.method == 'POST':
        form = ParamForm(request.POST)
        if form.is_valid():
            form_dataset = form.cleaned_data['data']
            data = pd.read_csv(BASE_DIR + '/datasets/' + form_dataset, error_bad_lines=False, nrows=450000)
            # text = total_hist_by_protocol(data, True)
            print("name", form_dataset.split('_')[0])
            images = call_all_funcs(data, form_dataset.split('_')[0])
            form = ParamForm()

            # img = ImgText(**{'title': 'first graph', 'img': 'total_by_protocol.png', 'text': 'here\'s the text',
            #                  'table_title': ["aasdgsd", "bb", "cc"], 'table': [[1, 2, 3], [2, 3, 4]]})
            # images.append(img)
            # img = ImgText(**{'title': 'second graph', 'img': '1.jpeg'})
            # images.append(img)

    else:
        form = ParamForm()

    return render(request, 'index.html', {'form': form, 'images': images})