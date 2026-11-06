"""One bounded read-only inventory of official data metadata, never MRI/GT voxels."""
import ast
import json
import os
from pathlib import Path
import shlex

from session import session, access_module


CODE = r'''
import csv,io,json,os,re,zipfile
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

CLINICAL=re.compile(r'pre.?op|post.?op|pre.?treat|post.?treat|treatment|surgery|clinical|time.?point|pre.?post',re.I)
TEXT_EXT={'.csv','.tsv','.json','.txt','.md','.yaml','.yml','.xlsx','.xls','.pdf'}
LIMIT=512*1024

def metadata_name(name):
    return Path(name).suffix.lower() in TEXT_EXT and not any(part.startswith('.') for part in Path(name).parts)

def inspect_metadata(name,body):
    suffix=Path(name).suffix.lower()
    if suffix in {'.csv','.tsv'}:
        rows=csv.DictReader(io.StringIO(body.decode('utf-8-sig')),delimiter='\t' if suffix=='.tsv' else ',')
        columns=rows.fieldnames or []
        fields=[c for c in columns if CLINICAL.search(c)]
        count=0
        for _ in rows:count+=1
        return dict(columns=columns,clinical_candidate_columns=fields,rows=count,patient_rows_returned=False)
    if suffix=='.json':
        data=json.loads(body)
        if isinstance(data,dict):keys=list(data)[:60]
        elif isinstance(data,list) and data and isinstance(data[0],dict):keys=list(data[0])[:60]
        else:keys=[]
        return dict(schema_keys=keys,clinical_candidate_keys=[k for k in keys if CLINICAL.search(str(k))],patient_values_returned=False)
    text=body.decode('utf-8-sig',errors='replace')
    lines=[line[:500] for line in text.splitlines() if CLINICAL.search(line)]
    return dict(clinical_matching_line_count=len(lines),clinical_matching_excerpt=lines[:8])

def inspect_root(root):
    root=Path(root)
    result=dict(path=str(root),exists=root.is_dir(),directories_visited=0,entries_seen=0,
                metadata=[],archives=[],clinical_directory_names=[],skipped_symlinks=0,
                limits_reached=[],errors=[],root_noncase_entries=[])
    if not root.is_dir():return result
    pending=[(root,0)];read_bytes=0
    while pending:
        directory,depth=pending.pop()
        if result['directories_visited']>=3000 or result['entries_seen']>=30000:
            result['limits_reached'].append('directory_inventory_budget');break
        result['directories_visited']+=1
        try:
            entries=sorted(os.scandir(directory),key=lambda e:e.name)
        except OSError as error:
            result['errors'].append(dict(path=str(directory),error=type(error).__name__));continue
        for entry in entries:
            result['entries_seen']+=1
            if entry.name.startswith('.'):continue
            if entry.is_symlink():result['skipped_symlinks']+=1;continue
            relative=str(Path(entry.path).relative_to(root))
            if depth==0 and not re.fullmatch(r'BraTS-MET-\d{5}-\d{3}',entry.name):
                if len(result['root_noncase_entries'])<60:result['root_noncase_entries'].append(entry.name)
            if entry.is_dir(follow_symlinks=False):
                if CLINICAL.search(entry.name):result['clinical_directory_names'].append(relative)
                if depth<3:pending.append((Path(entry.path),depth+1))
                else:result['limits_reached'].append('depth:'+relative)
                continue
            if not entry.is_file(follow_symlinks=False):continue
            if metadata_name(entry.name):
                if len(result['metadata'])>=64:
                    result['limits_reached'].append('metadata_file_budget');continue
                size=entry.stat(follow_symlinks=False).st_size
                item=dict(path=relative,bytes=size,content_read=False)
                if Path(entry.name).suffix.lower() in {'.pdf','.xlsx','.xls'}:
                    item['not_parsed']='binary_document_inventory_only'
                elif size>LIMIT or read_bytes+size>4*1024**2:
                    item['not_parsed']='metadata_byte_budget'
                else:
                    try:
                        body=Path(entry.path).read_bytes();read_bytes+=len(body)
                        item.update(inspect_metadata(entry.name,body),content_read=True)
                    except (OSError,ValueError,UnicodeError) as error:item['error']=type(error).__name__
                result['metadata'].append(item)
            elif entry.name.lower().endswith('.zip') and 'brats' in entry.name.lower() and len(result['archives'])<4:
                item=dict(path=relative,central_directory_only=True,metadata=[],errors=[])
                try:
                    with zipfile.ZipFile(entry.path) as archive:
                        members=archive.infolist();item['member_count']=len(members)
                        item['clinical_directory_names']=sorted({str(Path(x.filename).parent) for x in members if any(CLINICAL.search(p) for p in Path(x.filename).parts[:-1])})[:60]
                        for member in members:
                            if not metadata_name(member.filename) or member.is_dir():continue
                            record=dict(path=member.filename,bytes=member.file_size,content_read=False)
                            suffix=Path(member.filename).suffix.lower()
                            if len(item['metadata'])>=64:break
                            if suffix in {'.pdf','.xlsx','.xls'}:record['not_parsed']='binary_document_inventory_only'
                            elif member.file_size<=LIMIT and read_bytes+member.file_size<=4*1024**2:
                                body=archive.read(member);read_bytes+=len(body)
                                record.update(inspect_metadata(member.filename,body),content_read=True)
                            else:record['not_parsed']='metadata_byte_budget'
                            item['metadata'].append(record)
                except (OSError,ValueError,zipfile.BadZipFile) as error:item['errors'].append(type(error).__name__)
                result['archives'].append(item)
    result['metadata_bytes_read']=read_bytes
    result['limits_reached']=sorted(set(result['limits_reached']))[:40]
    return result

def main():
    started=datetime.now(timezone.utc).isoformat()
    roots=[Path('/public/home/zqchen/data'),Path('/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/data/raw')]
    discovery=[]
    for parent in [Path('/public/home/zqchen'),Path('/public/home/zqchen/projects/ECNU_EYU_data')]:
        if not parent.is_dir():continue
        for entry in sorted(os.scandir(parent),key=lambda e:e.name):
            if re.search(r'brats|miccai|mets|dataset',entry.name,re.I) and not entry.name.startswith('.') and not entry.is_symlink():
                discovery.append(str(Path(entry.path)))
                if entry.is_dir(follow_symlinks=False) and len(roots)<6:roots.append(Path(entry.path))
    output=dict(remote_started_utc=started,roots=[inspect_root(p) for p in dict.fromkeys(roots)],
                related_top_level_discovery=discovery,remote_finished_utc=datetime.now(timezone.utc).isoformat(),
                source_images_read=False,GT_voxels_read=False,checkpoint_read=False,remote_writes=False,
                jobs_queried_or_modified=False,archive_MRI_members_extracted=False,
                boundary='current_retained_directory_and_available_archive_metadata_only_not_original_release_completeness_proof')
    print(json.dumps(output))

if __name__=='__main__':main()
'''


def main():
    expected = '/Users/hwaigc/比赛+课题/服务器登陆/服务器登陆的env'
    if os.environ.get('ECNU_SERVER_ENV_FILE') != expected:
        raise RuntimeError('Explicit authorized credential env path required')
    ast.parse(CODE, feature_version=(3, 10))
    with session('inspect_official_clinical_metadata') as access:
        access_module.exclusive(access.audit / 'EXECUTED_SOURCE.json', dict(source=CODE))
        command = 'env PYTHONDONTWRITEBYTECODE=1 ' + shlex.quote(os.environ['ECNU_REMOTE_PYTHON'])
        output = access.ssh(command + ' -B -c ' + shlex.quote(CODE), timeout=120)
        result = json.loads(output.strip().splitlines()[-1])
        access_module.exclusive(access.audit / 'RESULT.json', dict(access_module.META, **result))
        print(json.dumps(dict(audit=str(access.audit), roots=result['roots'],
                              remote_utc=result['remote_finished_utc']), ensure_ascii=False))


if __name__ == '__main__':
    main()
