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
