"""Remove all write_image calls entirely and uninstall kaleido to prevent hangs."""
import re
import os

src_dir = 'src'
files = ['eda_analysis.py', 'clustering_model.py']

for fname in files:
    fpath = os.path.join(src_dir, fname)
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove try/except blocks wrapping write_image (from previous fix)
    content = re.sub(
        r'    try:\n        (?:fig\w*|segment_fig)\.write_image\(.+\)\n        except Exception:\n        pass  # kaleido not available\n',
        '',
        content
    )
    
    # Also remove any remaining bare write_image lines
    content = re.sub(r'^.*\.write_image\(.+\).*\n', '', content, flags=re.MULTILINE)
    
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'Cleaned {fname}')

print('All write_image calls removed!')
