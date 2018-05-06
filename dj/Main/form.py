from django import forms


PROTS_CHOICES = ((1, 'TCP'), (2, 'SMTP'))
DATA_CHOICES = ((1, 'Sality'), (2, 'Bunitu'))


class ParamForm(forms.Form):
    data = forms.ChoiceField(choices=DATA_CHOICES, label="Choose dataset")
    prots = forms.MultipleChoiceField(choices=PROTS_CHOICES, label="Choose list of protocols",
                                      widget=forms.CheckboxSelectMultiple)
