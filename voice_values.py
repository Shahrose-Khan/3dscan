import re

# Define the list of measures you are looking for
measures_list = [
    "CAL", "CBL", "CBL1", "CCL", "CDL", "CEL", "CFL", "CGL", "CYL", "LGL",
    "CAR", "CBR", "CBL1", "CCR", "CDR", "CER", "CFR", "CGR", "CYR", "LGR"
    ]

# The input text containing the values
input_text = """
value of CAL is 35.98 
value of CBL is 45.98
value of CB1L is 48.9 
value of good enough doctor can you go a bit slow CCR is 42.9 
i am fine are you good.

CDL is 37.89 
CEL is measured 89 cm the value is final.
value of CFL is 189 
value of CGL is 54
value of CYL is 42.1
value of LGL is 95.5
value of LGR is 105.5
I measure the CA which is 35.98
"""

# Initialize a dictionary to store the measures and their values
measures = {}

# Create a regex pattern that matches any of the measures in the list and their values, with word boundaries
pattern = r'\b(' + '|'.join(measures_list) + r')\b\D*([\d.]+)'

# Use re.findall to find all matches in the input text
matches = re.findall(pattern, input_text)

# Populate the dictionary with the matched measures and their values
for label, value in matches:
    measures[label] = float(value)

# Example usage
print(measures)
print(f"Measure of CA: {measures.get('CAL')}")
print(f"Measure of LG: {measures.get('LGL')}")