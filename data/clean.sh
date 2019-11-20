sed  -i -E 's|(.+:.+:.+)(:)(.+)|\1#SEMICOLON#\3|g' entity_full_names.txt
sed  -i -E 's| |#SPACE#|g' entity_full_names.txt
sed  -i -E 's|,|#COMMA#|g' entity_full_names.txt

sed  -i -E 's| |#SPACE#|g' data.txt
sed  -i -E 's|,|#COMMA#|g' data.txt
sed  -i -E 's|:|#SEMICOLON#|g' data.txt

sed  -i -E 's| |#SPACE#|g' domain_range.txt
sed  -i -E 's|,|#COMMA#|g' domain_range.txt
sed  -i -E 's|:|#SEMICOLON#|g' domain_range.txt
