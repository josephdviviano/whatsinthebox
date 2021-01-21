pip freeze | cut -d = -f 1 | cut -d @ -f 1 | sed '/-e/ d' > requirements.txt
