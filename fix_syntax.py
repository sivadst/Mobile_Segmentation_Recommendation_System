"""Fix multiline f-string syntax errors in src files."""
import re
import os

src_dir = 'src'
files = ['eda_analysis.py', 'recommendation_system.py', 'clustering_model.py']

for fname in files:
    fpath = os.path.join(src_dir, fname)
    with open(fpath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if this line has an unterminated print(f" or print(" with no closing quote
        # Pattern: line ends with print(f"\n or print("\n (the string continues on next line)
        if i + 1 < len(lines):
            stripped = line.rstrip('\n').rstrip('\r')
            # Case 1: print(f"  or print("  at end of line (string continues on next line)
            if re.match(r'^\s*print\(f?"$', stripped):
                # Merge with next line by replacing the newline with \n in the string
                next_line = lines[i + 1]
                # The current line is like: print(f"
                # Next line is like: Something here")
                merged = stripped + '\\n' + next_line.lstrip()
                fixed_lines.append(merged)
                i += 2
                continue
        fixed_lines.append(line)
        i += 1
    
    with open(fpath, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    print(f'Fixed {fname}')

print('All files fixed!')
