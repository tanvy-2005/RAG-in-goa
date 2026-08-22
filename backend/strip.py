import os
import io
import tokenize
import glob

def remove_comments_and_docstrings(source):
    io_obj = io.StringIO(source)
    out = ""
    last_lineno = -1
    last_col = 0
    prev_toktype = tokenize.ENCODING
    
    try:
        tokens = list(tokenize.generate_tokens(io_obj.readline))
    except Exception:
        return source
    
    for tok in tokens:
        token_type = tok.type
        token_string = tok.string
        start_line, start_col = tok.start
        end_line, end_col = tok.end
        
        # Check if docstring
        is_docstring = False
        if token_type == tokenize.STRING:
            # simplistic docstring check
            if prev_toktype in (tokenize.INDENT, tokenize.NEWLINE, tokenize.NL, tokenize.ENCODING):
                is_docstring = True
                
        # Space alignment
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            # We don't want to add whitespace if we are deleting the token and it's alone on the line?
            # We will just preserve whitespace to keep indentation identical.
            out += (" " * (start_col - last_col))
            
        if token_type == tokenize.COMMENT:
            pass # delete
        elif is_docstring:
            out += '""' # Replace docstring with empty string to preserve syntax validity
        else:
            out += token_string
            
        if token_type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT) or (not is_docstring and token_type == tokenize.STRING):
            prev_toktype = token_type
        if is_docstring:
            prev_toktype = tokenize.STRING
            
        last_lineno = end_line
        last_col = end_col
        
    # Clean up empty lines that had only comments
    cleaned_out = []
    for line in out.splitlines():
        if line.strip() == "" and line != "":
            # it might be just whitespace left from a deleted comment, but we want to preserve blank lines that were already there.
            pass
    # We will just leave the whitespace, it's safer.
    return out

for filepath in glob.glob("*.py"):
    if filepath == "strip.py":
        continue
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    
    new_source = remove_comments_and_docstrings(source)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_source)

print("Done stripping comments.")
