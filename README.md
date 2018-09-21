# KnowledgeScholar

## Steps to run KnowledgeScholar
### Step 1: Install software requirements
* Python 3.6.3 or above
* Python matplotlib package
* Python scipy package
* Python numpy package
* Python pandas package

Tip: You can simply install all required python packages by
<code>pip install -r requirements.txt</code>.

### Step 2: Download the dataset
Then you download the dataset from [here](https://www.dropbox.com/sh/h6kgo1nwjovh6j4/AACvGqeuPPEr9rTfNbfzMdqpa?dl=0) into your local machine.

### Step 3: Run the script to integrate data
Modify file paths in <code>data/data_path_file.txt</code> based on where you placed the dataset in Step 2. 
Then run the script <code>script_to_integrate_data.py</code> in the repo (after you clone it in your machine) with the following command <code>python script_to_integrate_data.py data/data_path_file.txt data/data_map.txt data/data_rules.xml data/inconsistency_rules.xml out.txt inconsistency.txt</code>. 

* <code>data/data_path_file.txt</code> is the file that lists locations of all the datasets to integrate. 
* <code>data/data_map.txt</code> is essentially the name map table (Fig. 1 in the Manuscript.docx) that lists all the synonyms.
* <code>data/data_rules.xml</code> is the data rule in xml format that lists logical statements to infer new facts from existing facts in the integrated dataset.
* <code>data/inconsistency_rules.xml</code> is the inconsistency rule in xml format that list logical statemetns to find conflictings facts in the integrated dataset.
* <code>out.txt</code> is the output file name that contains all the non-conflicting facts (or knowledge).
* <code>inconsistency.txt</code> is the name of another output filethat contains all the conflicting facts (or knowledge).

## Notes:
* There are many functions in the package (under modules directory) not are specifically used in the script (script_to_integrate_data). You might want to use them depending on your needs.
* I used AverageLog inconsistency corrector in this script but there are many more options in inconsistency_correctors directory. All the methods implemented in this dir are described in Manuscript.docx

## Information about the file formats
### 1. data_path_file
The first column represents the file path and the second column represent the name of the dataset. Note that there should be a header and the first column name being Path and the second column name being Source.

### 2. data_map
The first column represents the source name of an entity and the second column represent its target name. Note that there should be a header and the first column name being Source and the second column name being Target.

### 3. data_rules.xml and inconsistency_rules.xml
Please refer to the sections of "Rules to add knowledge" and "Rules to identify inconsistency" in Manuscript.docx.

### 4. out.txt
In this output file (a set of consistent knowledge), the knowledge is represented in triple format (Subject, Predicate, Object). There are 6 columns in the following order. 
* Subject:
* Predicate:
* Object:
* Belief: the confidence score of this specific fact measured by the inconsistency corrector (default is AverageLog).
* Source size: number of sources supporting this knowledge
* Sources: the list of sources supporting this knowledge

### inconsistency_out.txt
In this output file (a set of inconsistent knowledge), the knowledge is represented in triple format (Subject, Predicate, Object). The first six columns are for the fact with the highest belief (i.e. one that the inconsistency corrector think it is correct) among conflicting facts.
* Subject:
* Predicate:
* Object:
* Belief: the confidence score of this specific fact measured by the inconsistency corrector (default is AverageLog).
* Source size: number of sources supporting this knowledge
* Sources: the list of sources supporting this knowledge
* Total source size: the number of all sources (including the source size from the conflicting facts)
* Mean belief of conflicting tuples: the average of beliefs of all other tuples conflicting to one that is represented in the first 3 columns. 
* Belief difference: the difference between the Belief and the Mean belief of conflicting tuples
* Conflicting tuple info: the list of all the tuples conflicting to one that is represented in the first 3 columns. It is the list represented in "[(element1), (element2), ...]" and each element of the list is represented by "tuple, sources, belief" where tuple is "(subject, predicate, object)", and source is "[source1, source2, ...]"
