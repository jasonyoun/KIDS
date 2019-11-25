
* <code>[./data/data_path_file.txt](./data/data_path_file.txt)</code>: is the file that lists locations of all the datasets to integrate.
* <code>[./data/name_map.txt](./data/name_map.txt)</code>: is essentially the name map table (Fig. 1 in the Manuscript.docx) that lists all the synonyms.
* <code>[./data/data_rules.xml](./data/data_rules.xml)</code>: is the data rule in xml format that lists logical statements to infer new facts from existing facts in the integrated dataset.
* <code>[./data/inconsistency_rules.xml](./data/inconsistency_rules.xml)</code>: is the inconsistency rule in xml format that list logical statemetns to find conflictings facts in the integrated dataset.
* <code>[./out.txt](./out.txt)</code>: is the output file name that contains all the non-conflicting facts (or knowledge).
* <code>[./inconsistency.txt](./inconsistency.txt)</code>: is the name of another output filethat contains all the conflicting facts (or knowledge).

## Information about the file formats
### 1. data_path_file
The first column represents the file path and the second column represent the name of the dataset. Note that there should be a header and the first column name being Path and the second column name being Source.

### 2. name_map
The first column represents the source name of an entity and the second column represent its target name. Note that there should be a header and the first column name being Source and the second column name being Target.

### 3. data_rules.xml and inconsistency_rules.xml
Please refer to the sections of "Rules to add knowledge" and "Rules to identify inconsistency" in Manuscript.docx.