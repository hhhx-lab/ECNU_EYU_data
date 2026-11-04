"""Parse observed Slurm compound keys without changing frozen submission helpers."""
import re


def job_fields(text):
    pattern=r'(?:^|\s)(?:(?P<key>[A-Za-z][A-Za-z0-9_/:]*)=|(?P<legacy>NtasksPerTRES):)'
    tokens=list(re.finditer(pattern,text))
    result={}
    for index,token in enumerate(tokens):
        key=token.group('key') or token.group('legacy')
        assert key not in result, 'duplicate Slurm field: '+key
        end=tokens[index+1].start() if index+1<len(tokens) else len(text)
        result[key]=text[token.end():end].strip()
    return result
