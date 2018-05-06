from django.shortcuts import render
from Main.form import ParamForm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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


class ImgText:
    def __init__(self, img, text, is_table, table_title = None, table=None):
        self.img = img
        self.text = text
        self.is_table = is_table
        self.table_title = table_title
        self.table = table


def main(request):
    images = []
    text = ""
    if request.method == 'POST':
        form = ParamForm(request.POST)
        if form.is_valid():
            form_dataset = form.cleaned_data['data']
            if form_dataset == "1":
                print(form_dataset)
                data = pd.read_csv('Sality_data.csv', error_bad_lines=False, nrows=450000)
                text = total_hist_by_protocol(data, True)

            form = ParamForm()
            img = ImgText('total_by_protocol.png', 'text', True, ["aasdgsd", "bb", "cc"], [[1, 2, 3], [2, 3, 4]])
            images.append(img)
            img = ImgText('1.jpeg', 'TEXT2', False)
            images.append(img)
            # images.append('total_by_protocol.png')
            # images.append('1.jpeg')

    else:
        form = ParamForm()

    return render(request, 'index.html', {'form': form, 'images': images, 'text': text})