import re
import json
import numpy as np
import functools 


with open("per_section.json") as f:
    json_data = json.load(f)

with open("all_phr.json") as f:
    all_phrases = json.load(f)

COARSE_VIEWS=['A2C',
 'A3C',
 'A4C',
 'A5C',
 'Apical_Doppler',
 'Doppler_Parasternal_Long',
 'Doppler_Parasternal_Short',
 'Parasternal_Long',
 'Parasternal_Short',
 'SSN',
 'Subcostal']

ALL_SECTIONS=["Left Ventricle",
             "Resting Segmental Wall Motion Analysis",
             "Right Ventricle",
             "Left Atrium",
             "Right Atrium",
             "Atrial Septum",
             "Mitral Valve",
             "Aortic Valve",
             "Tricuspid Valve",
             "Pulmonic Valve",
             "Pericardium",
             "Aorta",
             "IVC",
             "Pulmonary Artery",
             "Pulmonary Veins",
             "Postoperative Findings"]

t_list = {k: [all_phrases[k][j] for j in all_phrases[k]] 
          for k in all_phrases}
phrases_per_section_list={k:functools.reduce(lambda a,b: a+b, v) for (k,v) in t_list.items()}
phrases_per_section_list_org={k:functools.reduce(lambda a,b: a+b, v) for (k,v) in t_list.items()}

numerical_pattern = r'(\\d+(\\.\\d+)?)'  # Escaped backslashes for integers or floats
string_pattern = r'\\b\\w+.*?(?=\\.)'

def isin(phrase,text):
    return phrase.lower() in (text.lower())

def extract_section(report, section_header):
    # Create a regex pattern that matches the section and anything up to the next [SEP]
    pattern = rf"{section_header}(.*?)(?=\[SEP\])"
    
    # Search for the pattern in the report
    match = re.search(pattern, report)
    
    # If a match is found, return the section including the header and the content up to [SEP]
    if match:
        # Include the trailing [SEP] if you need it as part of the output
        return f"{section_header}{match.group(1)}[SEP]"
    else:
        return "Section not found."

def extract_features(report: str) -> list:
    """
    Returns a list of 21 different features
    see json_data for a list of features
    """
    sorted_features=['impella',
    'ejection_fraction',
    'pacemaker',
    'rv_systolic_function_depressed',
    'right_ventricle_dilation',
    'left_atrium_dilation',
    'right_atrium_dilation',
    'mitraclip',
    'mitral_annular_calcification',
    'mitral_stenosis',
    'mitral_regurgitation',
    'tavr',
    'bicuspid_aov_morphology',
    'aortic_stenosis',
    'aortic_regurgitation',
    'tricuspid_stenosis',
    'tricuspid_valve_regurgitation',
    'pericardial_effusion',
    'aortic_root_dilation',
    'dilated_ivc',
    'pulmonary_artery_pressure_continuous']

    sorted_json_data = {k:json_data[k] for k in sorted_features}
    features=[]
    for key,value in sorted_json_data.items():
        if value['mode'] == "regression":
            match=None
            for phrase in value['label_sources']:
                pattern = re.compile((phrase.split("<#>")[0] + r"(\d{1,3}(?:\.\d{1,2})?)"), re.IGNORECASE)
                match = pattern.search(report)
                if match:
                    features.append(float(match.group(1)))
                    break
            if match is None:
                features.append(np.nan)

        elif value['mode'] == "binary":
            assigned=False
            for phrase in value['label_sources']:
                if isin(phrase,report):
                    features.append(1)
                    assigned=True
                    break
            if not assigned:
                features.append(0) 
    return features

def make_it_regex(sec):

    # replace numerical and string with corresponding regex
    for idx in range(len(sec)):
        sec[idx]=sec[idx].replace('(', '\(').replace(')', '\)').replace("+",'\+')
        sec[idx]=re.sub(r'<numerical>', numerical_pattern, sec[idx])
        sec[idx]=re.sub(r'<string>', string_pattern, sec[idx])

    regex_sec = re.compile('|'.join(sec), flags=re.IGNORECASE)
    return regex_sec


regex_per_section={k: make_it_regex(v)
                   for (k,v) in phrases_per_section_list.items()}
def remove_subsets(strings):
    result=[]
    for string in strings:
        if not any(string in res for res in result):
            result.append(string)  

    return list(result)

def structure_rep(rep):
    #remove double spaces
    rep = re.sub(r'\s{2,}', ' ', rep)
    structured_report = []
    for sec in ALL_SECTIONS:
        cur_section= extract_section(rep,sec)
        new_section=[sec+":"]
        
        # Find all matches using the combined pattern
        for match in re.finditer(regex_per_section[sec], cur_section):
            new_section.append(cur_section[match.start():match.end()])
            
        if len(new_section)>1:
            #remove phrases that are a subset of some other phrase
            new_section=remove_subsets(new_section)
            new_section.append("[SEP]")
            structured_report+=new_section
            
    # Join structured report parts
    structured_report = ' '.join(structured_report)
    return structured_report



def phrase_decode(phrase_ids):
    report = ""
    current_section = -1
    for sec_idx, phrase_idx, value in phrase_ids:
        section=list(phrases_per_section_list_org.keys())[sec_idx]
        if sec_idx!=current_section:
            if current_section!=-1:
                report+="[SEP] "
            report += section + ": "
            current_section=sec_idx

        # Get phrase template
        phr = phrases_per_section_list_org[section][phrase_idx]

        if '<numerical>' in phr:
            phr = phr.replace('<numerical>',str(value))
        elif '<string>' in phr:
            phr = phr.replace('<string>',str(value))
            
        report += phr + " "
    report += "[SEP]"
    return report

