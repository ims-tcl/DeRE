import csv
import re
import math
import itertools
import warnings
import shutil
import os

verbose = False
ignore_discontinuous_spans = True
maximal_number_of_sentences = math.inf

inconsitency_counter = 0
discontinuous_span_counter = 0
total_counter = 0

def get_original_sentence(formatted_sentence):
    # Construct original sentence in two steps
    original_sentence = re.sub(r"<span class=\"chemical\">([^<]+)</span>", r"\1", formatted_sentence)
    original_sentence = re.sub(r"<span class=\"disease\">([^<]+)</span>", r"\1", original_sentence)
    return original_sentence

def get_indices(formatted_sentence):
    chemical_pattern = r"<span class=\"chemical\">([^<]+)</span>"
    disease_pattern = r"<span class=\"disease\">([^<]+)</span>"

    chemical_matches = [e for e in re.finditer(chemical_pattern, formatted_sentence)]
    disease_matches = [e for e in re.finditer(disease_pattern, formatted_sentence)]

    offset = 0
    chemical_indices = []
    disease_indices = []

    # Merge-sort-like loop
    while (len(chemical_matches) > 0 or len(disease_matches) > 0):
        if len(chemical_matches) > 0:
            next_chemical_index = chemical_matches[0].span()[0]
        else:
            next_chemical_index = math.inf
        if len(disease_matches) > 0:
            next_disease_index = disease_matches[0].span()[0]
        else:
            next_disease_index = math.inf

        if next_chemical_index < next_disease_index:
            word_length = len(chemical_matches[0].group(1))
            start_index = next_chemical_index - offset
            stop_index = start_index + word_length
            # 30 character in <span class=\"chemical\">(.+)</span> will be removed
            offset += 30
            chemical_matches.pop(0)
            chemical_indices.append((start_index, stop_index))
        else:
            word_length = len(disease_matches[0].group(1))
            start_index = next_disease_index - offset
            stop_index = start_index + word_length
            # 29 character in <span class=\"disease\">(.+)</span> will be removed
            offset += 29
            disease_matches.pop(0)
            disease_indices.append((start_index, stop_index))
    return chemical_indices, disease_indices


def check_annotation_consistency(original_sentence, chemical_indices, disease_indices,
                                 chemical_names_from_csv, disease_names_from_csv):
    # The conversion to a set removes dublicates (if a chemical is mentioned twice in a sentence)
    chemical_names_from_indices = set([original_sentence[e[0]:e[1]].lower() for e in chemical_indices])
    chemical_names_permutations = ['/'.join(permutation) for permutation in itertools.permutations(chemical_names_from_indices)]
    if not chemical_names_from_csv.lower() in chemical_names_permutations:
        return False
        #raise Exception("Annotation inconsitency excpetion:\nSentence:"+original_sentence+"\nFound chemicals: "+
        #                '/'.join([original_sentence[e[0]:e[1]] for e in chemical_indices])+
        #                "\nAnnotated chemicals: "+chemical_names_from_csv)

    disease_names_from_indices = set([original_sentence[e[0]:e[1]].lower() for e in disease_indices])
    disease_names_permutations = ['/'.join(permutation) for permutation in
                                   itertools.permutations(disease_names_from_indices)]
    if not disease_names_from_csv.lower() in disease_names_permutations:
        return False
        #raise Exception("Annotation inconsitency excpetion:\nSentence:" + original_sentence + "\nFound diseases: " +
        #                '/'.join([original_sentence[e[0]:e[1]] for e in disease_indices]) +
        #                "\nAnnotated chemicals: " + disease_names_from_csv)
    return True

def generate_BRAT_files(id, original_sentence, chemical_indices, disease_indices, relation):
    # Build the annotation file step by step
    # 1.) Chemical
    annotation_text = "T1\tChemical {} {}".format(chemical_indices[0][0], chemical_indices[0][1])
    # If there are distributed spans
    for idxs in chemical_indices[1:]:
        annotation_text += ";{} {}".format(idxs[0], idxs[1])
    # BRAT uses spaces, in the original CSV they use slashes
    annotation_text += "\t" + ' '.join([original_sentence[e[0]:e[1]] for e in chemical_indices])
    # 2.) Disease
    annotation_text += "\nT2\tDisease {} {}".format(disease_indices[0][0], disease_indices[0][1])
    # If there are distributed spans
    for idxs in disease_indices[1:]:
        annotation_text += ";{} {}".format(idxs[0], idxs[1])
    annotation_text += "\t" + ' '.join([original_sentence[e[0]:e[1]] for e in disease_indices])
    # 3.) Relation
    if relation == "yes_direct":
        annotation_text += "\nE1\tDirectly_related_chemical:T1 Disease:T2"
    elif relation == "yes_indirect":
        annotation_text += "\nE1\tIndirectly_related_chemical:T1 Disease:T2"
    if not(ignore_discontinuous_spans and (len(disease_indices)>1 or len(chemical_indices)>1)):
        # Write the text file
        with open("BRAT/" + id + ".txt", "w") as text_file:
            text_file.write(original_sentence)
        # Write the annotation file
        with open("BRAT/" + id + ".a2", "w") as annotation_file:
            annotation_file.write(annotation_text)
    else:
        global discontinuous_span_counter
        discontinuous_span_counter += 1

# Delete folder
try:
    shutil.rmtree('BRAT')
except:
    pass
# Create folder
os.makedirs("BRAT")
with open('chemicals-and-disease-DFE.csv', 'r', encoding = "ISO-8859-1") as csvfile:
    # I started to fix the problems in the csv, anyway it seems to be full of errors, this is what i fixed:
    # I had to remove the sentence: The in vitro antistaphylococcal activity of <span class="chemical"></span><span class="disease">RP</span> 59500, a new streptogramin, was comparable to those of vancomycin and teicoplanin against Staphylococcus aureus, and <span class="disease">RP</span> 59500 was the most active agent against coagulase-negative staphylococci.
    # (Line 607)
    # I had to remove the sentence: We have investigated the effect of <span class="chemical"></span><span class="disease">retinoic acid</span> (<span class="disease">RA</span>) and retinyl acetate (RAc) on the production of reactive oxygen metabolites and the release of lysosomal enzymes by human polymorphonuclear leukocytes (PMN).
    # (Line 672), this also seems to be a dublicated sentence in the data set (also the next sentence)
    # I had to remove the sentence: Four patients on the G regimen were withdrawn due to <span class="chemical"></span><span class="disease">glucose</span> intolerance while none of the patients on GF developed <span class="chemical"></span><span class="disease">glucose</span> intolerance or hyperlipidaemia.
    # (Line 704)
    # I had to remove the sentence: We have investigated the effect of <span class="disease">retinoic acid</span> (<span class="disease">RA</span>) and retinyl acetate (RAc) on the production of reactive <span class="chemical">oxygen</span> metabolites and the release of lysosomal enzymes by human polymorphonuclear leukocytes (PMN).
    # (Line 671) as the annotation <span class="chemical">retinyl acetate</span> contradicts the sentence (this and the line before seem to be wrong anyway)
    # I removed lines 720 and 721 as they are obviously wrong and on top inconsistently annotated (e.g. Effect of Konjac food on <span class="disease">blood </span><span class="chemical">glucose</span> level in patients with diabetes.)
    # I removed the whitespace in the sentence <span class="disease">blood </span>
    # I removed line 719 because of inconsitency (The data analyzed by multiple F test indicate that the fasting <span class="disease">blood</span><span class="chemical">glucose</span> (FBG) and the 2-h postprandial <span class="disease">blood </span><span class="chemical">glucose</span> (PBG) on the 30th and the 65th days after the food was ingested were significantly reduced (P = 0.001, P less than 0.001, respectively), as was the glycosylated hemoglobin level at the end of the trial (P less than 0.05).)

    line_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    lines = []
    for row in line_reader:
        lines.append(row)
    for line in lines[1:]:
        id = line[0]
        # Get relation (this also includes the finalized annotations and not just the golden ones)
        relation = line[6]
        chemical_name_formatted = line[10]
        disease_name_formatted = line[12]
        formatted_sentence = line[13]

        original_sentence = get_original_sentence(formatted_sentence)
        chemical_indices, disease_indices = get_indices(formatted_sentence)

        if verbose:
            print('-' * 80)
            print("Formatted sentence: " + formatted_sentence)
            print("Original sentence: " + original_sentence)
            print("Chemicals recovered from indices: " + str([original_sentence[e[0]:e[1]] for e in chemical_indices]))
            print("Diseases recovered from indices: " + str([original_sentence[e[0]:e[1]] for e in disease_indices]))
            print("Relation: " + relation)


        # Check consistency
        # Extract chemical name
        p_c = re.compile(r"<span class=\"chemical\">(.+)</span>")
        match_chemical_name = p_c.match(chemical_name_formatted)
        chemical_name = match_chemical_name.group(1)
        # Extract disease name
        p_d = re.compile(r"<span class=\"disease\">(.+)</span>")
        match_disease_name = p_d.match(disease_name_formatted)
        disease_name = match_disease_name.group(1)

        if check_annotation_consistency(original_sentence, chemical_indices, disease_indices, chemical_name, disease_name):
            generate_BRAT_files(id, original_sentence, chemical_indices, disease_indices, relation)
        else:
            warnings.warn("Ignored inconsistent sentence: " + formatted_sentence)
            inconsitency_counter += 1
        total_counter += 1
        if total_counter == maximal_number_of_sentences:
            break

    print('-' * 80 + "\nIgnored {} inconsistent sentences from a total of {} sentences.\nIgnored {} sentences with discontinuous spans.".format(inconsitency_counter, total_counter, discontinuous_span_counter))