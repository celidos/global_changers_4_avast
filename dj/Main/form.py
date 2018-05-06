from django import forms


PROTS_CHOICES = ((1, 'TCP'), (2, 'SMTP'))
DATA_CHOICES = (("Sality_data.csv", 'Sality'), ("Bunitu_data.csv", 'Bunitu'))


class ParamForm(forms.Form):
    data = forms.ChoiceField(choices=DATA_CHOICES, label="Choose dataset")
    # prots = forms.MultipleChoiceField(choices=PROTS_CHOICES, label="Choose list of protocols",
    #                                   widget=forms.CheckboxSelectMultiple)
